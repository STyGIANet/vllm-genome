# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)


class _DispatchTrafficWriter:

    def __init__(self, dump_dir: str, global_rank: int):
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_dir = dump_dir
        self.global_rank = global_rank
        self.event_idx = 0
        self.path = os.path.join(dump_dir, f"traffic_{global_rank}.jsonl")
        self._fp = open(self.path, "w", encoding="utf-8", buffering=1)

    def write(self, payload: dict) -> None:
        payload["event_idx"] = self.event_idx
        self._fp.write(json.dumps(payload) + "\n")
        self.event_idx += 1

    def close(self) -> None:
        self._fp.close()


_WRITER_LOCK = threading.Lock()
_TRAFFIC_WRITERS: dict[tuple[str, int], _DispatchTrafficWriter] = {}
_FAILED_WRITER_KEYS: set[tuple[str, int]] = set()


def _get_writer(
    dump_dir: str,
    global_rank: int,
) -> _DispatchTrafficWriter | None:
    key = (dump_dir, global_rank)
    with _WRITER_LOCK:
        if key in _FAILED_WRITER_KEYS:
            return None
        writer = _TRAFFIC_WRITERS.get(key)
        if writer is not None:
            return writer
        try:
            writer = _DispatchTrafficWriter(dump_dir=dump_dir, global_rank=global_rank)
        except OSError as exc:
            logger.warning_once(
                "Failed to initialize MoE dispatch traffic dump at %s: %s",
                dump_dir,
                exc,
                scope="local",
            )
            _FAILED_WRITER_KEYS.add(key)
            return None
        _TRAFFIC_WRITERS[key] = writer
        return writer


def reset_moe_dispatch_traffic_writer(
    dump_dir: str | None,
    global_rank: int,
) -> None:
    if not dump_dir:
        return
    key = (dump_dir, global_rank)
    with _WRITER_LOCK:
        writer = _TRAFFIC_WRITERS.pop(key, None)
        if writer is not None:
            writer.close()


def _resolve_dump_dir(layer: "FusedMoE") -> str | None:
    model_config = getattr(getattr(layer, "vllm_config", None), "model_config", None)
    if model_config is None:
        return None
    dump_dir = getattr(model_config, "moe_dispatch_traffic_dump_dir", None)
    if not isinstance(dump_dir, str):
        return None
    normalized = dump_dir.strip()
    return normalized or None


def is_moe_dispatch_traffic_dump_enabled(layer: "FusedMoE") -> bool:
    return _resolve_dump_dir(layer) is not None


def _linear_owner_ranks(
    expert_ids: torch.Tensor,
    ep_size: int,
    global_num_experts: int,
) -> torch.Tensor:
    base = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    threshold = (base + 1) * remainder
    if base == 0:
        return expert_ids
    return torch.where(
        expert_ids < threshold,
        torch.div(expert_ids, base + 1, rounding_mode="floor"),
        remainder
        + torch.div(expert_ids - threshold, base, rounding_mode="floor"),
    )


def _destination_ep_ranks(
    layer: "FusedMoE",
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    expert_ids = topk_ids.to(dtype=torch.int64)
    if getattr(layer, "enable_eplb", False):
        local_physical_experts = layer.global_num_experts // layer.ep_size
        return torch.div(
            expert_ids,
            local_physical_experts,
            rounding_mode="floor",
        )

    if getattr(layer, "expert_placement_strategy", "linear") == "round_robin":
        return torch.remainder(expert_ids, layer.ep_size)

    return _linear_owner_ranks(
        expert_ids=expert_ids,
        ep_size=layer.ep_size,
        global_num_experts=layer.global_num_experts,
    )


def _counts_per_destination_rank(
    layer: "FusedMoE",
    topk_ids: torch.Tensor,
) -> list[int]:
    if topk_ids.numel() == 0:
        return [0] * layer.ep_size

    if topk_ids.dim() == 1:
        topk_ids = topk_ids.unsqueeze(-1)
    dest_ranks = _destination_ep_ranks(layer, topk_ids)
    valid = (dest_ranks >= 0) & (dest_ranks < layer.ep_size)
    clipped = dest_ranks.clamp(min=0, max=layer.ep_size - 1)
    dest_mask = F.one_hot(clipped, num_classes=layer.ep_size)
    dest_mask = dest_mask * valid.unsqueeze(-1)
    per_token_mask = dest_mask.amax(dim=1)
    counts = per_token_mask.sum(dim=0)
    return [int(x) for x in counts.tolist()]


def maybe_dump_moe_dispatch_traffic(
    layer: "FusedMoE",
    topk_ids: torch.Tensor,
) -> None:
    dump_dir = _resolve_dump_dir(layer)
    if dump_dir is None:
        return

    ep_group = get_ep_group()
    writer = _get_writer(dump_dir=dump_dir, global_rank=int(ep_group.rank))
    if writer is None:
        return

    forward_pass_idx = None
    moe_layer_ordinal_in_pass = None
    batch_signature = None
    if is_forward_context_available():
        forward_context = get_forward_context()
        forward_pass_idx = int(forward_context.forward_pass_idx)
        moe_layer_ordinal_in_pass = int(forward_context.moe_layer_index - 1)
        batch_signature = forward_context.additional_kwargs.get(
            "moe_dispatch_traffic_batch_signature"
        )

    counts = _counts_per_destination_rank(layer, topk_ids)
    payload = {
        "layer_id": int(layer.layer_id),
        "layer_name": layer.layer_name,
        "batch_signature": batch_signature,
        "forward_pass_idx": forward_pass_idx,
        "moe_layer_ordinal_in_pass": moe_layer_ordinal_in_pass,
        "source_global_rank": int(ep_group.rank),
        "source_ep_rank": int(ep_group.rank_in_group),
        "dest_global_ranks": [int(rank) for rank in ep_group.ranks],
        "dest_ep_ranks": list(range(int(ep_group.world_size))),
        "dest_token_counts": counts,
        "num_tokens": int(topk_ids.shape[0]),
        "top_k": int(topk_ids.shape[1] if topk_ids.dim() > 1 else 1),
        "num_token_copies": int(sum(counts)),
        "ep_world_size": int(ep_group.world_size),
        "enable_eplb": bool(getattr(layer, "enable_eplb", False)),
        "expert_placement_strategy": getattr(
            layer, "expert_placement_strategy", None
        ),
    }
    writer.write(payload)
