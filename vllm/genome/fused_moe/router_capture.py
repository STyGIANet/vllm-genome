# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.forward_context import get_forward_context


def record_expert_load(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
) -> None:
    """Record expert load metrics for the selected physical expert ids."""
    topk_ids_flatten = topk_ids.flatten()
    expert_load_view.scatter_add_(
        dim=0,
        index=topk_ids_flatten.long(),
        src=torch.ones_like(topk_ids_flatten).to(expert_load_view),
    )


def record_expert_load_ranges(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_ranges: list[tuple[int, int]],
) -> None:
    """Record expert load metrics for selected token ranges only."""
    for start, end in record_ranges:
        if end <= start:
            continue
        record_expert_load(topk_ids[start:end], expert_load_view)


def maybe_capture_routing(
    layer_id: int | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
) -> None:
    """Capture logical routed expert IDs for offline tracking/placement."""
    if layer_id is None:
        return

    from vllm.model_executor.layers.fused_moe.layer import (
        _ROUTING_DATA,
        _is_tracking_enabled,
    )

    ctx = get_forward_context()
    capture_for_tracking = _is_tracking_enabled()
    capture_prefill_only = bool(
        ctx is not None
        and ctx.additional_kwargs.get(
            "routing_capture_prefill_only_for_placement", False
        )
    )
    capture_for_placement = bool(
        ctx is not None
        and (
            capture_prefill_only
            or "routing_capture_prefill_ranges" in ctx.additional_kwargs
        )
    )
    if not capture_for_tracking and not capture_for_placement:
        return
    if ctx is not None and bool(
        ctx.additional_kwargs.get("skip_routed_experts_capture", False)
    ):
        return

    capture_ids = topk_ids
    capture_weights = topk_weights if capture_for_tracking else None
    capture_num = int(topk_ids.shape[0])
    capture_prefill_ranges: list[tuple[int, int]] | None = None
    capture_prefill_req_ids: list[str] | None = None

    if ctx is not None:
        capture_prefill_ranges = list(
            ctx.additional_kwargs.get("routing_capture_prefill_ranges", [])
        )
        capture_prefill_req_ids = list(
            ctx.additional_kwargs.get("routing_capture_prefill_req_ids", [])
        )

    if capture_prefill_only and capture_prefill_ranges is not None:
        prefill_id_parts: list[torch.Tensor] = []
        prefill_weight_parts: list[torch.Tensor] = []
        normalized_prefill_ranges: list[tuple[int, int]] = []
        normalized_prefill_req_ids: list[str] | None = None
        if (
            capture_prefill_req_ids is not None
            and len(capture_prefill_req_ids) == len(capture_prefill_ranges)
        ):
            normalized_prefill_req_ids = []
        offset = 0
        for idx, (start, end) in enumerate(capture_prefill_ranges):
            start = max(0, int(start))
            end = min(int(end), int(topk_ids.shape[0]))
            if end <= start:
                continue
            token_count = end - start
            prefill_id_parts.append(topk_ids[start:end].clone())
            if topk_weights is not None and capture_weights is not None:
                prefill_weight_parts.append(topk_weights[start:end].clone())
            normalized_prefill_ranges.append((offset, offset + token_count))
            if normalized_prefill_req_ids is not None:
                normalized_prefill_req_ids.append(capture_prefill_req_ids[idx])
            offset += token_count
        if prefill_id_parts:
            capture_ids = torch.cat(prefill_id_parts, dim=0)
            if capture_weights is not None:
                if prefill_weight_parts:
                    capture_weights = torch.cat(prefill_weight_parts, dim=0)
                else:
                    capture_weights = None
            capture_num = int(capture_ids.shape[0])
            capture_prefill_ranges = normalized_prefill_ranges
            capture_prefill_req_ids = normalized_prefill_req_ids
        else:
            capture_prefill_ranges = []
            capture_prefill_req_ids = []
    elif ctx is not None and ctx.dp_metadata is not None:
        from vllm.distributed import get_dp_group

        try:
            dp_rank = get_dp_group().rank_in_group
            sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()
            start = sum(sizes[:dp_rank])
            end = start + sizes[dp_rank]
            if end <= topk_ids.shape[0] and end > start:
                capture_ids = topk_ids[start:end].clone()
                if capture_weights is not None:
                    capture_weights = topk_weights[start:end].clone()
                capture_num = int(capture_ids.shape[0])
                if capture_prefill_ranges is not None:
                    local_prefill_ranges: list[tuple[int, int]] = []
                    local_prefill_req_ids: list[str] | None = None
                    if (
                        capture_prefill_req_ids is not None
                        and len(capture_prefill_req_ids)
                        == len(capture_prefill_ranges)
                    ):
                        local_prefill_req_ids = []
                    for idx, (req_start, req_end) in enumerate(
                        capture_prefill_ranges
                    ):
                        local_start = max(int(req_start), start)
                        local_end = min(int(req_end), end)
                        if local_end <= local_start:
                            continue
                        local_prefill_ranges.append(
                            (local_start - start, local_end - start)
                        )
                        if local_prefill_req_ids is not None:
                            local_prefill_req_ids.append(
                                capture_prefill_req_ids[idx]
                            )
                    capture_prefill_ranges = local_prefill_ranges
                    capture_prefill_req_ids = local_prefill_req_ids
        except Exception:
            capture_prefill_ranges = None
            capture_prefill_req_ids = None

    if layer_id not in _ROUTING_DATA:
        _ROUTING_DATA[layer_id] = []
    capture_record = {
        "topk_ids": capture_ids.detach(),
        "num_tokens": capture_num,
        "prefill_ranges": capture_prefill_ranges,
        "prefill_req_ids": capture_prefill_req_ids,
    }
    if capture_weights is not None:
        capture_record["topk_weights"] = capture_weights.detach()
    _ROUTING_DATA[layer_id].append(capture_record)
