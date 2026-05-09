# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch


@dataclass
class PrefixLearningAsyncStepJob:
    step_seq: int
    epoch: int
    req_ids: list[str]
    req_lengths: list[int]
    topk_by_layer: dict[int, np.ndarray]


@dataclass
class PrefixLearningPendingCopyJob:
    step_seq: int
    epoch: int
    req_ids: list[str]
    req_lengths: list[int]
    topk_by_layer_cpu: dict[int, torch.Tensor]
    gpu_refs: list[torch.Tensor]
    ready_event: torch.cuda.Event | None


class AsyncPrefixLearningOwnerLearner:
    def __init__(
        self,
        num_ranks: int,
        trace_enabled: bool,
    ) -> None:
        self._num_ranks = num_ranks
        self._trace_enabled = trace_enabled
        self._work_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._lock = threading.Lock()
        self._owner_counts_by_req: dict[str, np.ndarray] = {}
        self._processed_step_seq = 0
        self._epoch: int | None = None
        self._owner_lookup: np.ndarray | None = None
        self._owner_rank_lookup: np.ndarray | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="PrefixLearningAsyncOwnerLearner",
            daemon=True,
        )
        self._thread.start()

    def update_owner_state(
        self,
        epoch: int,
        owner_lookup: np.ndarray,
        owner_rank_lookup: np.ndarray | None,
    ) -> None:
        self._work_queue.put(
            ("owner_state", (epoch, owner_lookup, owner_rank_lookup))
        )

    def enqueue_step(self, job: PrefixLearningAsyncStepJob) -> None:
        self._work_queue.put(("step", job))

    def drop_requests(self, req_ids: Iterable[str]) -> None:
        req_id_list = list(req_ids)
        if req_id_list:
            self._work_queue.put(("drop", req_id_list))

    def get_processed_step_seq(self) -> int:
        with self._lock:
            return self._processed_step_seq

    def take_owner_if_ready(
        self,
        req_id: str,
        required_step_seq: int,
        epoch: int,
    ) -> tuple[dict[str, int] | None, bool]:
        with self._lock:
            processed = self._processed_step_seq >= required_step_seq
            if not processed or self._epoch != epoch:
                return None, processed
            scores = self._owner_counts_by_req.pop(req_id, None)

        if scores is None or scores.size == 0:
            return None, True
        best_score = float(scores.max(initial=0.0))
        if best_score <= 0:
            return None, True
        target_rank = int(np.argmax(scores))
        return {
            "target_rank": target_rank,
            "epoch": epoch,
        }, True

    def _run(self) -> None:
        while True:
            kind, payload = self._work_queue.get()
            if kind == "owner_state":
                epoch, owner_lookup, owner_rank_lookup = cast(
                    tuple[int, np.ndarray, np.ndarray | None], payload
                )
                with self._lock:
                    self._epoch = epoch
                    self._owner_lookup = owner_lookup
                    self._owner_rank_lookup = owner_rank_lookup
                    self._owner_counts_by_req.clear()
                continue

            if kind == "drop":
                req_ids = cast(list[str], payload)
                with self._lock:
                    for req_id in req_ids:
                        self._owner_counts_by_req.pop(req_id, None)
                continue

            if kind != "step":
                continue

            job = cast(PrefixLearningAsyncStepJob, payload)
            with self._lock:
                owner_lookup = self._owner_lookup
                owner_rank_lookup = self._owner_rank_lookup
                epoch = self._epoch

            if (
                epoch is None
                or job.epoch != epoch
                or owner_lookup is None
            ):
                with self._lock:
                    self._processed_step_seq = max(
                        self._processed_step_seq, job.step_seq
                    )
                continue

            num_reqs = len(job.req_ids)
            if num_reqs <= 0:
                with self._lock:
                    self._processed_step_seq = max(
                        self._processed_step_seq, job.step_seq
                    )
                continue

            req_lengths = np.asarray(job.req_lengths, dtype=np.int64)
            token_req_indices = np.repeat(
                np.arange(num_reqs, dtype=np.int64),
                req_lengths,
            )
            batch_counts = np.zeros(
                (num_reqs, self._num_ranks),
                dtype=np.float32,
            )
            flat_req_indices: np.ndarray | None = None
            flat_req_topk = -1
            for layer_id, captured in job.topk_by_layer.items():
                if captured.size == 0:
                    continue
                if captured.ndim != 2 or captured.shape[0] != token_req_indices.size:
                    continue
                topk = int(captured.shape[1])
                if topk <= 0:
                    continue
                if flat_req_indices is None or flat_req_topk != topk:
                    flat_req_indices = np.repeat(token_req_indices, topk)
                    flat_req_topk = topk
                experts = captured.reshape(-1)
                if owner_rank_lookup is not None:
                    rank_lookup_layer = owner_rank_lookup[int(layer_id)]
                    valid_mask = (
                        (experts >= 0)
                        & (experts < rank_lookup_layer.shape[0])
                    )
                    if not np.any(valid_mask):
                        continue
                    owner_ranks = rank_lookup_layer[experts[valid_mask]]
                    owner_mask = owner_ranks >= 0
                    if not np.any(owner_mask):
                        continue
                    linear_indices = (
                        flat_req_indices[valid_mask][owner_mask] * self._num_ranks
                        + owner_ranks[owner_mask].astype(np.int64, copy=False)
                    )
                    batch_counts += np.bincount(
                        linear_indices,
                        minlength=num_reqs * self._num_ranks,
                    ).reshape(num_reqs, self._num_ranks).astype(
                        np.float32,
                        copy=False,
                    )
                else:
                    owner_lookup_layer = owner_lookup[int(layer_id)]
                    valid_mask = (
                        (experts >= 0)
                        & (experts < owner_lookup_layer.shape[0])
                    )
                    if not np.any(valid_mask):
                        continue
                    valid_req_indices = flat_req_indices[valid_mask]
                    owner_rows = owner_lookup_layer[experts[valid_mask]]
                    for rank_idx in range(self._num_ranks):
                        batch_counts[:, rank_idx] += np.bincount(
                            valid_req_indices,
                            weights=owner_rows[:, rank_idx],
                            minlength=num_reqs,
                        ).astype(np.float32, copy=False)

            with self._lock:
                if self._epoch == job.epoch:
                    nonzero_req_indices = np.flatnonzero(
                        np.any(batch_counts > 0, axis=1)
                    )
                    for req_idx in nonzero_req_indices:
                        req_id = job.req_ids[int(req_idx)]
                        counts = batch_counts[int(req_idx)]
                        existing = self._owner_counts_by_req.get(req_id)
                        if existing is None:
                            self._owner_counts_by_req[req_id] = counts.copy()
                        else:
                            existing += counts
                self._processed_step_seq = max(self._processed_step_seq, job.step_seq)

