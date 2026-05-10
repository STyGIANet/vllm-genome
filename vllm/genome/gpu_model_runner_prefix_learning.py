# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.prefix_router import build_owner_cache_from_physical_to_logical_map

from .prefix_learning import (
    AsyncPrefixLearningOwnerLearner,
    PrefixLearningAsyncStepJob,
    PrefixLearningPendingCopyJob,
)

logger = init_logger(__name__)


class GenomePrefixLearningCaptureRunnerMixin:

    def _init_genome_prefix_learning_state(self) -> None:
        self._current_prefill_capture_ranges = []
        self._current_prefill_capture_req_ids = []
        self._last_prefix_router_placement_epoch = None
        self._last_prefix_router_physical_to_logical_map = None
        self._prefix_learning_owner_cache_epoch = None
        self._prefix_learning_owner_lookup = None
        self._prefix_learning_owner_rank_lookup = None
        self._prefix_learning_num_layers = 0
        self._prefix_learning_num_experts = 0
        self._prefix_learning_seen_pairs_buffer = None
        self._prefix_learning_owner_counts_buffer = None
        self._prefix_learning_req_slot_by_id = {}
        self._prefix_learning_req_id_by_slot = []
        self._prefix_learning_free_slots = []
        self._prefix_learning_step_req_ids = []
        self._prefix_learning_step_req_lengths = []
        self._prefix_learning_step_total_prompt_tokens = 0
        self._prefix_learning_step_topk_by_layer = {}
        self._prefix_learning_async_step_seq = 0
        self._prefix_learning_last_step_seq_by_req = {}
        self._prefix_learning_pending_async_req_steps = {}
        self._prefix_learning_pending_copy_jobs = []
        self._prefix_learning_copy_stream = None
        self._prefix_learning_async_learner = None
        self._prefix_learning_trace_debug = bool(
            getattr(self.model_config, "load_balancer_debug", False)
        )
        self._prefix_learning_dummy_owner_enabled = bool(
            int(os.getenv("VLLM_PREFIX_LEARNING_DUMMY_OWNER", "0"))
        )
        self._skip_prefix_learning_capture_for_current_forward = False
        self._placement_routing_dump_session_dir = None
        if self.model_config.placement_routing_dump_dir:
            self._placement_routing_dump_session_dir = (
                self._make_placement_routing_dump_session_dir(
                    self.model_config.placement_routing_dump_dir
                )
            )

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts=%s "
            "enable_prefix_affinity_routing=%s",
            self.model_config.enable_return_routed_experts,
            self.model_config.enable_prefix_affinity_routing,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        self.routed_experts_attn_gid = self._get_attention_kv_cache_gid()
        min_block_size = min(
            [
                group.kv_cache_spec.block_size
                for group in self.kv_cache_config.kv_cache_groups
            ]
        )
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // num_groups
        ) * min_block_size
        dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size * dcp_size > 1:
            self.max_num_kv_tokens *= pcp_size * dcp_size

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_router_capture_hooks(routed_experts_capturer)
        self.routed_experts_initialized = True

    def init_prefix_learning_capture(self) -> None:
        self._bind_router_capture_hooks(None)

    def _uses_placement_snapshot_for_prefix_learning(self) -> bool:
        return False

    def _get_primary_prefix_learning_capture_inputs(
        self,
        prefill_capture_ranges: list[tuple[int, int]],
        prefill_capture_req_ids: list[str],
    ) -> tuple[list[tuple[int, int]], list[str]]:
        self._update_prefix_learning_freeze_state()
        return prefill_capture_ranges, prefill_capture_req_ids

    def _get_or_alloc_prefix_learning_slot(self, req_id: str) -> int | None:
        slot = self._prefix_learning_req_slot_by_id.get(req_id)
        if slot is not None:
            return slot
        if (self._prefix_learning_owner_counts_buffer is None
                or not self._prefix_learning_free_slots):
            return None
        slot = self._prefix_learning_free_slots.pop()
        self._prefix_learning_req_slot_by_id[req_id] = slot
        self._prefix_learning_req_id_by_slot[slot] = req_id
        self._prefix_learning_owner_counts_buffer[slot].zero_()
        return slot

    def _free_prefix_learning_slot(self, req_id: str) -> None:
        slot = self._prefix_learning_req_slot_by_id.pop(req_id, None)
        if slot is None:
            return
        if self._prefix_learning_owner_counts_buffer is not None:
            self._prefix_learning_owner_counts_buffer[slot].zero_()
        if 0 <= slot < len(self._prefix_learning_req_id_by_slot):
            self._prefix_learning_req_id_by_slot[slot] = None
        self._prefix_learning_free_slots.append(slot)

    def _clear_all_prefix_learning_state(self) -> None:
        stale_req_ids = list(self._prefix_learning_req_slot_by_id.keys())
        stale_req_ids.extend(self._prefix_learning_pending_async_req_steps.keys())
        if self._prefix_learning_owner_counts_buffer is not None:
            self._prefix_learning_owner_counts_buffer.zero_()
        self._prefix_learning_req_slot_by_id.clear()
        if self._prefix_learning_req_id_by_slot:
            self._prefix_learning_req_id_by_slot = [None] * len(
                self._prefix_learning_req_id_by_slot
            )
        self._prefix_learning_free_slots = list(
            range(self.max_num_reqs - 1, -1, -1)
        )
        self._prefix_learning_step_req_ids = []
        self._prefix_learning_step_req_lengths = []
        self._prefix_learning_step_total_prompt_tokens = 0
        self._prefix_learning_step_topk_by_layer = {}
        self._prefix_learning_last_step_seq_by_req.clear()
        self._prefix_learning_pending_async_req_steps.clear()
        if self._prefix_learning_async_learner is not None and stale_req_ids:
            self._prefix_learning_async_learner.drop_requests(stale_req_ids)

    def _ensure_prefix_learning_async_learner(self) -> AsyncPrefixLearningOwnerLearner:
        learner = self._prefix_learning_async_learner
        if learner is None:
            learner = AsyncPrefixLearningOwnerLearner(
                num_ranks=self.parallel_config.data_parallel_size,
                trace_enabled=self._prefix_learning_trace_debug,
            )
            self._prefix_learning_async_learner = learner
        return learner

    def _take_async_prefix_learning_owner_updates(
        self,
    ) -> dict[str, dict[str, int]] | None:
        self._drain_prefix_learning_pending_copy_jobs()
        learner = self._prefix_learning_async_learner
        epoch = self._prefix_learning_owner_cache_epoch
        if learner is None or epoch is None or not self._prefix_learning_pending_async_req_steps:
            return None

        results: dict[str, dict[str, int]] = {}
        ready_req_ids: list[str] = []
        for req_id, required_step_seq in list(
            self._prefix_learning_pending_async_req_steps.items()
        ):
            owner, processed = learner.take_owner_if_ready(
                req_id=req_id,
                required_step_seq=required_step_seq,
                epoch=epoch,
            )
            if owner is not None:
                results[req_id] = owner
                ready_req_ids.append(req_id)
            elif processed:
                ready_req_ids.append(req_id)

        for req_id in ready_req_ids:
            self._prefix_learning_pending_async_req_steps.pop(req_id, None)
            self._prefix_learning_last_step_seq_by_req.pop(req_id, None)

        return results or None

    def _drain_prefix_learning_pending_copy_jobs(self) -> None:
        if not self._prefix_learning_pending_copy_jobs:
            return

        learner = self._prefix_learning_async_learner
        if learner is None:
            return

        current_epoch = self._prefix_learning_owner_cache_epoch
        remaining_jobs: list[PrefixLearningPendingCopyJob] = []
        for job in self._prefix_learning_pending_copy_jobs:
            if job.ready_event is not None and not job.ready_event.query():
                remaining_jobs.append(job)
                continue
            if current_epoch is None or job.epoch != current_epoch:
                continue
            learner.enqueue_step(
                PrefixLearningAsyncStepJob(
                    step_seq=job.step_seq,
                    epoch=job.epoch,
                    req_ids=job.req_ids,
                    req_lengths=job.req_lengths,
                    topk_by_layer={
                        layer_id: tensor.numpy()
                        for layer_id, tensor in job.topk_by_layer_cpu.items()
                    },
                )
            )
        self._prefix_learning_pending_copy_jobs = remaining_jobs

    def _force_drain_prefix_learning_pending_copy_jobs(
        self,
        required_step_seq: int,
    ) -> None:
        if not self._prefix_learning_pending_copy_jobs:
            return

        learner = self._prefix_learning_async_learner
        if learner is None:
            return

        current_epoch = self._prefix_learning_owner_cache_epoch
        if current_epoch is None:
            return

        remaining_jobs: list[PrefixLearningPendingCopyJob] = []
        max_enqueued_step_seq = 0
        for job in self._prefix_learning_pending_copy_jobs:
            if job.epoch != current_epoch or job.step_seq > required_step_seq:
                remaining_jobs.append(job)
                continue
            if job.ready_event is not None:
                job.ready_event.synchronize()
            learner.enqueue_step(
                PrefixLearningAsyncStepJob(
                    step_seq=job.step_seq,
                    epoch=job.epoch,
                    req_ids=job.req_ids,
                    req_lengths=job.req_lengths,
                    topk_by_layer={
                        layer_id: tensor.numpy()
                        for layer_id, tensor in job.topk_by_layer_cpu.items()
                    },
                )
            )
            max_enqueued_step_seq = max(max_enqueued_step_seq, job.step_seq)
        self._prefix_learning_pending_copy_jobs = remaining_jobs

        if max_enqueued_step_seq <= 0:
            return

        deadline = time.perf_counter() + 0.5
        while learner.get_processed_step_seq() < max_enqueued_step_seq:
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.0005)

    def _update_prefix_learning_freeze_state(self) -> bool:
        return False

    def _get_prefix_learning_capture_cpu_dtype(self) -> torch.dtype:
        num_experts = self._prefix_learning_num_experts
        if num_experts > 0:
            if num_experts <= 127:
                return torch.int8
            if num_experts <= 32767:
                return torch.int16
        return torch.int32

    def _refresh_prefix_learning_owner_state(
        self,
        physical_to_logical_map: Sequence[Sequence[int]],
        epoch: int,
    ) -> None:
        if (
            not self.model_config.enable_prefix_affinity_routing
            or self._prefix_learning_num_layers <= 0
            or self._prefix_learning_num_experts <= 0
        ):
            return

        num_ranks = self.parallel_config.data_parallel_size
        old_owner_lookup = self._prefix_learning_owner_lookup
        old_owner_rank_lookup = self._prefix_learning_owner_rank_lookup
        old_owner_epoch = self._prefix_learning_owner_cache_epoch
        owner_cache = build_owner_cache_from_physical_to_logical_map(
            physical_to_logical_map,
            num_ranks,
        )
        owner_lookup = torch.zeros(
            (
                self._prefix_learning_num_layers,
                self._prefix_learning_num_experts,
                num_ranks,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        use_single_owner_fast_path = (
            self.parallel_config.eplb_config.num_redundant_experts == 0
        )
        owner_rank_lookup = (
            torch.full(
                (
                    self._prefix_learning_num_layers,
                    self._prefix_learning_num_experts,
                ),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            if use_single_owner_fast_path
            else None
        )
        for layer_idx, layer_owner_cache in enumerate(
                owner_cache[:self._prefix_learning_num_layers]):
            for expert_id, owner_ranks in enumerate(
                    layer_owner_cache[:self._prefix_learning_num_experts]):
                for owner_rank in owner_ranks:
                    if 0 <= owner_rank < num_ranks:
                        owner_lookup[layer_idx, expert_id, owner_rank] = 1
                if owner_rank_lookup is not None:
                    if len(owner_ranks) == 1 and 0 <= owner_ranks[0] < num_ranks:
                        owner_rank_lookup[layer_idx, expert_id] = owner_ranks[0]
                    elif len(owner_ranks) > 1:
                        owner_rank_lookup = None

        epoch_changed = (
            old_owner_lookup is None
            or old_owner_epoch != epoch
            or not torch.equal(old_owner_lookup, owner_lookup)
            or (
                (old_owner_rank_lookup is None) != (owner_rank_lookup is None)
                or (
                    old_owner_rank_lookup is not None
                    and owner_rank_lookup is not None
                    and not torch.equal(old_owner_rank_lookup, owner_rank_lookup)
                )
            )
        )
        self._prefix_learning_owner_lookup = owner_lookup
        self._prefix_learning_owner_rank_lookup = owner_rank_lookup
        self._prefix_learning_owner_cache_epoch = epoch
        owner_lookup_cpu = owner_lookup.to(device="cpu").numpy()
        owner_rank_lookup_cpu = (
            owner_rank_lookup.to(device="cpu").numpy()
            if owner_rank_lookup is not None
            else None
        )
        self._ensure_prefix_learning_async_learner().update_owner_state(
            epoch=epoch,
            owner_lookup=owner_lookup_cpu,
            owner_rank_lookup=owner_rank_lookup_cpu,
        )
        if epoch_changed and (
            self._prefix_learning_req_slot_by_id
            or self._prefix_learning_pending_async_req_steps
        ):
            self._clear_all_prefix_learning_state()

    def _begin_prefix_learning_step_capture(
        self,
        prefill_capture_ranges: list[tuple[int, int]],
        prefill_capture_req_ids: list[str],
    ) -> None:
        trace_enabled = self._prefix_learning_trace_debug
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        self._prefix_learning_step_req_ids = []
        self._prefix_learning_step_req_lengths = []
        self._prefix_learning_step_total_prompt_tokens = 0
        self._prefix_learning_step_topk_by_layer = {}

        if (
            self._uses_placement_snapshot_for_prefix_learning()
            or
            not self.model_config.enable_prefix_affinity_routing
            or self._prefix_learning_num_layers <= 0
            or self._prefix_learning_num_experts <= 0
            or self._prefix_learning_owner_counts_buffer is None
        ):
            return

        active_req_ids: list[str] = []
        req_lengths: list[int] = []
        for req_id, (start, end) in zip(prefill_capture_req_ids, prefill_capture_ranges):
            if end <= start:
                continue
            slot = self._get_or_alloc_prefix_learning_slot(req_id)
            if slot is None:
                continue
            active_req_ids.append(req_id)
            req_lengths.append(end - start)

        total_prompt_tokens = sum(req_lengths)
        if not active_req_ids or total_prompt_tokens <= 0:
            return

        self._prefix_learning_step_req_ids = active_req_ids
        self._prefix_learning_step_req_lengths = req_lengths
        self._prefix_learning_step_total_prompt_tokens = total_prompt_tokens

    def _accumulate_prefix_learning_step_capture(self) -> None:
        trace_enabled = self._prefix_learning_trace_debug
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        self._drain_prefix_learning_pending_copy_jobs()

        epoch = self._prefix_learning_owner_cache_epoch
        step_topk_by_layer = self._prefix_learning_step_topk_by_layer
        if (
            not self.model_config.enable_prefix_affinity_routing
            or epoch is None
            or not step_topk_by_layer
        ):
            self._prefix_learning_step_req_ids = []
            self._prefix_learning_step_req_lengths = []
            self._prefix_learning_step_total_prompt_tokens = 0
            self._prefix_learning_step_topk_by_layer = {}
            return

        total_captured_tokens = 0
        total_layers = 0
        total_copy_enqueue_ms = 0.0
        req_count = len(self._prefix_learning_step_req_ids)
        topk_by_layer_cpu: dict[int, torch.Tensor] = {}
        gpu_refs: list[torch.Tensor] = []
        capture_cpu_dtype = self._get_prefix_learning_capture_cpu_dtype()
        copy_stream = None
        current_stream = None
        pin_memory = is_pin_memory_available()
        if self.device.type == "cuda":
            current_stream = torch.cuda.current_stream(device=self.device)
            if self._prefix_learning_copy_stream is None:
                self._prefix_learning_copy_stream = torch.cuda.Stream(
                    device=self.device
                )
            copy_stream = self._prefix_learning_copy_stream
        if copy_stream is not None:
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(current_stream)
                for layer_id, captured_parts in list(step_topk_by_layer.items()):
                    if not captured_parts:
                        continue
                    first_part = captured_parts[0]
                    topk = int(first_part.shape[1]) if first_part.ndim >= 2 else 0
                    if topk <= 0:
                        continue
                    layer_tokens = sum(int(part.shape[0]) for part in captured_parts)
                    if self._prefix_learning_step_total_prompt_tokens != layer_tokens:
                        continue
                    if trace_enabled:
                        enqueue_start_ns = time.perf_counter_ns()
                    cpu_buffer = torch.empty(
                        (layer_tokens, topk),
                        dtype=capture_cpu_dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    offset = 0
                    for part in captured_parts:
                        part_len = int(part.shape[0])
                        if part_len <= 0:
                            continue
                        src = part.to(dtype=capture_cpu_dtype)
                        cpu_buffer[offset:offset + part_len].copy_(
                            src,
                            non_blocking=True,
                        )
                        gpu_refs.append(src)
                        offset += part_len
                    topk_by_layer_cpu[int(layer_id)] = cpu_buffer
                    if trace_enabled:
                        total_copy_enqueue_ms += (
                            time.perf_counter_ns() - enqueue_start_ns
                        ) / 1e6
                    total_captured_tokens += layer_tokens
                    total_layers += 1
        else:
            for layer_id, captured_parts in list(step_topk_by_layer.items()):
                if not captured_parts:
                    continue
                first_part = captured_parts[0]
                topk = int(first_part.shape[1]) if first_part.ndim >= 2 else 0
                if topk <= 0:
                    continue
                layer_tokens = sum(int(part.shape[0]) for part in captured_parts)
                if self._prefix_learning_step_total_prompt_tokens != layer_tokens:
                    continue
                if trace_enabled:
                    enqueue_start_ns = time.perf_counter_ns()
                cpu_buffer = torch.empty(
                    (layer_tokens, topk),
                    dtype=capture_cpu_dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                offset = 0
                for part in captured_parts:
                    part_len = int(part.shape[0])
                    if part_len <= 0:
                        continue
                    src = part.to(dtype=capture_cpu_dtype)
                    cpu_buffer[offset:offset + part_len].copy_(src)
                    gpu_refs.append(src)
                    offset += part_len
                topk_by_layer_cpu[int(layer_id)] = cpu_buffer
                if trace_enabled:
                    total_copy_enqueue_ms += (
                        time.perf_counter_ns() - enqueue_start_ns
                    ) / 1e6
                total_captured_tokens += layer_tokens
                total_layers += 1

        ready_event: torch.Event | None = None
        if copy_stream is not None and topk_by_layer_cpu:
            ready_event = torch.cuda.Event()
            ready_event.record(copy_stream)

        if topk_by_layer_cpu:
            self._ensure_prefix_learning_async_learner()
            self._prefix_learning_async_step_seq += 1
            step_seq = self._prefix_learning_async_step_seq
            self._prefix_learning_pending_copy_jobs.append(
                PrefixLearningPendingCopyJob(
                    step_seq=step_seq,
                    epoch=epoch,
                    req_ids=list(self._prefix_learning_step_req_ids),
                    req_lengths=list(self._prefix_learning_step_req_lengths),
                    topk_by_layer_cpu=topk_by_layer_cpu,
                    gpu_refs=gpu_refs,
                    ready_event=ready_event,
                )
            )
            for req_id in self._prefix_learning_step_req_ids:
                self._prefix_learning_last_step_seq_by_req[req_id] = step_seq

        self._prefix_learning_step_req_ids = []
        self._prefix_learning_step_req_lengths = []
        self._prefix_learning_step_total_prompt_tokens = 0
        self._prefix_learning_step_topk_by_layer = {}

    def _accumulate_prefix_learning_from_routing_snapshot(
        self,
        routing_snapshot: dict[int, list[dict[str, Any]]],
    ) -> None:
        trace_enabled = self._prefix_learning_trace_debug
        start_ns = time.perf_counter_ns() if trace_enabled else 0

        if (
            not self._uses_placement_snapshot_for_prefix_learning()
            or not self.model_config.enable_prefix_affinity_routing
            or self._prefix_learning_owner_counts_buffer is None
            or self._prefix_learning_num_experts <= 0
            or self._prefix_learning_num_layers <= 0
        ):
            return

        updated_reqs = 0
        updated_pairs = 0
        for layer_id, captures in routing_snapshot.items():
            if layer_id < 0 or layer_id >= self._prefix_learning_num_layers:
                continue
            for capture in captures:
                prefill_ranges = capture.get("prefill_ranges")
                prefill_req_ids = capture.get("prefill_req_ids")
                if (not prefill_ranges or not prefill_req_ids
                        or len(prefill_ranges) != len(prefill_req_ids)):
                    continue
                topk_ids = capture["topk_ids"]
                for req_id, (start, end) in zip(prefill_req_ids, prefill_ranges):
                    if end <= start:
                        continue
                    if start < 0 or end > topk_ids.shape[0]:
                        continue
                    slot = self._get_or_alloc_prefix_learning_slot(req_id)
                    if slot is None:
                        continue
                    req_topk_ids = topk_ids[start:end]
                    if req_topk_ids.numel() == 0:
                        continue
                    flat_experts = req_topk_ids.reshape(-1).to(torch.int64)
                    valid_experts = flat_experts[
                        (flat_experts >= 0)
                        & (flat_experts < self._prefix_learning_num_experts)
                    ]
                    if (
                        valid_experts.numel() == 0
                        or self._prefix_learning_owner_lookup is None
                    ):
                        continue
                    self._prefix_learning_owner_counts_buffer[slot] += (
                        self._prefix_learning_owner_lookup[
                            int(layer_id), valid_experts
                        ].sum(dim=0)
                    )
                    updated_reqs += 1
                    updated_pairs += int(valid_experts.numel())

    def _extract_prefix_learning_pairs_for_slot(
        self, slot: int
    ) -> list[list[int]]:
        return []

    def _capture_prefix_learning_pairs(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> None:
        trace_enabled = self._prefix_learning_trace_debug
        stage_ms: dict[str, float] = {}
        if trace_enabled and self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        last_ns = start_ns

        def mark_stage(name: str) -> None:
            nonlocal last_ns
            if not trace_enabled:
                return
            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            now_ns = time.perf_counter_ns()
            stage_ms[name] = (now_ns - last_ns) / 1e6
            last_ns = now_ns

        if (
            self._skip_prefix_learning_capture_for_current_forward
            or self._uses_placement_snapshot_for_prefix_learning()
            or not self.model_config.enable_prefix_affinity_routing
            or layer_id < 0
            or layer_id >= self._prefix_learning_num_layers
            or self._prefix_learning_num_experts <= 0
        ):
            return
        if (not self._current_prefill_capture_ranges
                or not self._current_prefill_capture_req_ids
                or self._prefix_learning_step_total_prompt_tokens <= 0):
            return
        captured_parts = [
            topk_ids[start:end]
            for start, end in self._current_prefill_capture_ranges
            if end > start
        ]
        mark_stage("slice")
        if not captured_parts:
            return

        if len(captured_parts) == 1:
            captured = captured_parts[0]
        else:
            captured = torch.cat(captured_parts, dim=0)
        mark_stage("concat")

        if self._prefix_learning_step_total_prompt_tokens != captured.shape[0]:
            return

        self._prefix_learning_step_topk_by_layer.setdefault(
            int(layer_id), []).append(captured)
        mark_stage("store")

    def _get_prefix_learning_pairs_for_requests(
        self, req_ids: list[str]
    ) -> dict[str, list[list[int]]] | None:
        trace_enabled = self._prefix_learning_trace_debug
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        if not req_ids:
            return None

        results: dict[str, list[list[int]]] = {}
        for req_id in req_ids:
            slot = self._prefix_learning_req_slot_by_id.get(req_id)
            if slot is None:
                continue
            pairs = self._extract_prefix_learning_pairs_for_slot(slot)
            if not pairs:
                self._free_prefix_learning_slot(req_id)
                continue
            results[req_id] = pairs
            self._free_prefix_learning_slot(req_id)
        return results or None

    def _get_prefix_learning_owners_for_requests(
        self, req_ids: list[str]
    ) -> dict[str, dict[str, int]] | None:
        self._drain_prefix_learning_pending_copy_jobs()
        if (
            not req_ids
            or self._prefix_learning_owner_cache_epoch is None
        ):
            return None

        num_ranks = self.parallel_config.data_parallel_size
        if self._prefix_learning_dummy_owner_enabled:
            epoch = self._prefix_learning_owner_cache_epoch
            results: dict[str, dict[str, int]] = {}
            for req_id in req_ids:
                target_rank = sum(req_id.encode("utf-8")) % max(num_ranks, 1)
                results[req_id] = {
                    "target_rank": target_rank,
                    "epoch": epoch,
                }
                self._free_prefix_learning_slot(req_id)
            return results or None

        learner = self._prefix_learning_async_learner
        if learner is None:
            return None

        required_step_seq = max(
            (
                self._prefix_learning_last_step_seq_by_req.get(
                    req_id, self._prefix_learning_async_step_seq
                )
                for req_id in req_ids
            ),
            default=0,
        )
        if learner.get_processed_step_seq() < required_step_seq:
            self._force_drain_prefix_learning_pending_copy_jobs(required_step_seq)

        epoch = self._prefix_learning_owner_cache_epoch
        results: dict[str, dict[str, int]] = {}
        for req_id in req_ids:
            required_step_seq = self._prefix_learning_last_step_seq_by_req.get(
                req_id, self._prefix_learning_async_step_seq
            )
            owner, processed = learner.take_owner_if_ready(
                req_id=req_id,
                required_step_seq=required_step_seq,
                epoch=epoch,
            )
            self._free_prefix_learning_slot(req_id)
            if owner is not None:
                results[req_id] = owner
                self._prefix_learning_pending_async_req_steps.pop(req_id, None)
                self._prefix_learning_last_step_seq_by_req.pop(req_id, None)
            elif not processed:
                self._prefix_learning_pending_async_req_steps[req_id] = (
                    required_step_seq
                )
            else:
                self._prefix_learning_pending_async_req_steps.pop(req_id, None)
                self._prefix_learning_last_step_seq_by_req.pop(req_id, None)
        return results or None

    def _bind_router_capture_hooks(
        self, capturer: RoutedExpertsCapturer | None
    ) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        use_snapshot_prefix_learning = (
            self._uses_placement_snapshot_for_prefix_learning()
        )
        enable_prefix_learning = bool(
            self.model_config.enable_prefix_affinity_routing
        )

        if capturer is None and not enable_prefix_learning:
            for module in self.compilation_config.static_forward_context.values():
                if isinstance(module, FusedMoE) and isinstance(
                    module.router, BaseRouter
                ):
                    module.router.set_capture_fn(None)

            self._prefix_learning_num_layers = 0
            self._prefix_learning_num_experts = 0
            self._prefix_learning_seen_pairs_buffer = None
            self._prefix_learning_owner_counts_buffer = None
            self._prefix_learning_req_slot_by_id = {}
            self._prefix_learning_req_id_by_slot = []
            self._prefix_learning_free_slots = []
            return

        self._prefix_learning_num_layers = 0
        self._prefix_learning_num_experts = 0
        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id
                self._prefix_learning_num_layers = max(
                    self._prefix_learning_num_layers,
                    int(layer_id) + 1,
                )
                self._prefix_learning_num_experts = max(
                    self._prefix_learning_num_experts,
                    int(module.logical_num_experts),
                )

                if capturer is not None or not use_snapshot_prefix_learning:
                    def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                        if _capturer is not None:
                            _capturer.capture(_layer_id, topk_ids)
                        if not use_snapshot_prefix_learning:
                            self._capture_prefix_learning_pairs(_layer_id, topk_ids)

                    module.router.set_capture_fn(_capture_fn)
                else:
                    module.router.set_capture_fn(None)

        if (self._prefix_learning_num_layers > 0
                and self._prefix_learning_num_experts > 0):
            self._prefix_learning_seen_pairs_buffer = None
            self._prefix_learning_owner_counts_buffer = torch.zeros(
                (
                    self.max_num_reqs,
                    self.parallel_config.data_parallel_size,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            self._prefix_learning_req_slot_by_id = {}
            self._prefix_learning_req_id_by_slot = [None] * self.max_num_reqs
            self._prefix_learning_free_slots = list(
                range(self.max_num_reqs - 1, -1, -1))
        else:
            self._prefix_learning_seen_pairs_buffer = None
            self._prefix_learning_owner_counts_buffer = None
            self._prefix_learning_req_slot_by_id = {}
            self._prefix_learning_req_id_by_slot = []
            self._prefix_learning_free_slots = []
