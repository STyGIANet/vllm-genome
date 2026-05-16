# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


@dataclass
class NcclAllToAllOverlapUnit:
    send_hidden_states: torch.Tensor
    sent_token_indices: torch.Tensor
    sent_topk_weights: torch.Tensor
    send_tokens_per_expert: torch.Tensor
    send_a1q_scale: torch.Tensor | None
    dispatch_is_local: bool
    dispatch_send_peer_rank: int | None
    dispatch_recv_peer_rank: int | None
    dispatch_recv_token_count: int
    comm_estimate: float
    comp_estimate: float


@dataclass
class NcclAllToAllDispatchTask:
    unit: NcclAllToAllOverlapUnit
    recv_hidden_states: torch.Tensor
    recv_tokens_per_expert: torch.Tensor
    recv_topk_weights: torch.Tensor | None
    recv_a1q_scale: torch.Tensor | None
    work_handles: list[dist.Work]


@dataclass
class NcclAllToAllCombineTask:
    unit: NcclAllToAllOverlapUnit
    combined_output: torch.Tensor
    work_handles: list[dist.Work]


def hungarian_max_assignment(weight_matrix: torch.Tensor) -> tuple[int, ...]:
    """Return a max-weight permutation for a square matrix."""
    matrix = weight_matrix.detach().to(dtype=torch.float64, device="cpu")
    n = matrix.size(0)
    max_weight = matrix.max().item() if n > 0 else 0.0
    cost = (max_weight - matrix).tolist()

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return tuple(assignment)


def build_residual_schedule(
    rank_dispatch_token_matrix: torch.Tensor,
) -> list[tuple[tuple[int, ...], int, tuple[int, ...]]]:
    """
    Greedy max-weight decomposition of the source->destination token matrix.

    Each unit is a residual permutation with a uniform edge count across
    active rows. This is the decomposition used as the overlap "jobs".
    """
    if (
        rank_dispatch_token_matrix.numel() == 0
        or rank_dispatch_token_matrix.max().item() == 0
    ):
        return []

    residual = rank_dispatch_token_matrix.detach().to(
        dtype=torch.int64, device="cpu"
    ).clone()
    schedule: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []

    while residual.max().item() > 0:
        support = (residual > 0).to(dtype=torch.float64)
        priority_scale = float(residual.sum().item()) + 1.0
        permutation = hungarian_max_assignment(
            support * priority_scale + residual.to(dtype=torch.float64)
        )
        active_rows = tuple(
            row_idx
            for row_idx, col_idx in enumerate(permutation)
            if residual[row_idx, col_idx].item() > 0
        )
        if not active_rows:
            raise RuntimeError(
                "Failed to build a valid residual nccl_alltoall overlap schedule."
            )
        unit_edge_count = min(
            int(residual[row_idx, permutation[row_idx]].item())
            for row_idx in active_rows
        )
        schedule.append((permutation, unit_edge_count, active_rows))
        for row_idx in active_rows:
            residual[row_idx, permutation[row_idx]] -= unit_edge_count

    return schedule


def estimate_unit_cost(
    *,
    unit_edge_count: int,
    num_active_rows: int,
    hidden_size: int,
    hidden_element_size: int,
    intermediate_size: int,
    scale_element_size: int,
) -> tuple[float, float]:
    """
    Simple two-machine flowshop cost model for Johnson ordering.

    We treat dispatch+combine as the communication machine and local expert
    compute as the compute machine.
    """
    total_edges = unit_edge_count * num_active_rows
    bytes_per_edge = hidden_size * hidden_element_size + scale_element_size
    comm_cost = float(total_edges * bytes_per_edge * 2)
    comp_cost = float(total_edges * hidden_size * intermediate_size)
    return comm_cost, comp_cost


def apply_johnsons_rule(
    schedule_units: list[
        tuple[tuple[int, ...], int, tuple[int, ...], float, float]
    ],
) -> list[tuple[tuple[int, ...], int, tuple[int, ...], float, float]]:
    if len(schedule_units) <= 1:
        return list(schedule_units)

    set_u = [item for item in schedule_units if item[3] <= item[4]]
    set_v = [item for item in schedule_units if item[3] > item[4]]
    set_u.sort(key=lambda item: item[3])
    set_v.sort(key=lambda item: item[4], reverse=True)
    return set_u + set_v


def make_expert_tokens_meta(
    expert_num_tokens: torch.Tensor,
    device: torch.device,
) -> mk.ExpertTokensMetadata:
    expert_num_tokens = expert_num_tokens.to(device=device, dtype=torch.int32)
    return mk.ExpertTokensMetadata(
        expert_num_tokens=expert_num_tokens,
        expert_num_tokens_cpu=None,
    )


def prepare_scales_for_moe(
    a1q_scale: torch.Tensor | None,
    quant_config: FusedMoEQuantConfig | None,
    prepare_swizzled_scale: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor | None:
    if a1q_scale is None or quant_config is None:
        return a1q_scale
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        return prepare_swizzled_scale(a1q_scale)
    return a1q_scale


def maybe_make_unit_scales(
    a1q_scale: torch.Tensor | None,
    num_rows: int,
    token_indices: torch.Tensor,
) -> torch.Tensor | None:
    if a1q_scale is None or a1q_scale.ndim == 0:
        return a1q_scale
    if a1q_scale.size(0) != num_rows:
        return a1q_scale
    return a1q_scale.index_select(0, token_indices)


def unit_topk_weights(
    recv_topk_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones_like(recv_topk_ids, dtype=dtype)


def reconstruct_local_topk_ids_from_tokens_per_expert(
    *,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    counts = tokens_per_expert.to(dtype=torch.long)
    expert_ids = torch.repeat_interleave(
        torch.arange(
            num_local_experts,
            device=counts.device,
            dtype=torch.long,
        ),
        counts,
    )
    return expert_ids.to(dtype=dtype).view(-1, 1)


def block_current_stream_on_work_handles(work_handles: list[dist.Work]) -> None:
    for handle in work_handles:
        if hasattr(handle, "block_current_stream"):
            handle.block_current_stream()
        else:
            handle.wait()


def reduce_local_fused_output(
    *,
    weight_and_reduce_impl: mk.TopKWeightAndReduce,
    fused_expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    local_output = weight_and_reduce_impl.apply(
        output=None,
        fused_expert_output=fused_expert_output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    if local_output.ndim == 3 and local_output.size(1) == 1:
        local_output = local_output.squeeze(1)
    return local_output
