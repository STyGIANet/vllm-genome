# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Author: Vamsi Addanki @STyGIANet
# Nothing fancy, just plain torch distributed alltoall
# In view of being agnostic to Pcie,nvlink,rdma... NCCL takes care of it.

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    nccl_alltoall_overlap as overlap,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.triton_utils import tl
from vllm.utils.flashinfer import nvfp4_block_scale_interleave

_OVERLAP_COMBINE_GROUPS: dict[tuple[int, ...], dist.ProcessGroup] = {}


@dataclass(frozen=True)
class _DirectTritonOverlapContext:
    kernel_impl: mk.FusedMoEKernelModularImpl
    fused_experts: TritonExperts


@dataclass
class _NcclAllToAllHandle:
    send_counts: list[int]
    recv_counts: list[int]
    sent_token_indices: torch.Tensor
    sent_topk_weights: torch.Tensor


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig | None,
    defer_input_quant: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if defer_input_quant or quant_config is None:
        return a1, None

    input_sf = quant_config.a1_scale
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=False,
    )
    return a1q, a1q_scale


def _prepare_scales_for_moe(
    a1q_scale: torch.Tensor | None,
    quant_config: FusedMoEQuantConfig | None,
) -> torch.Tensor | None:
    return overlap.prepare_scales_for_moe(
        a1q_scale,
        quant_config,
        prepare_swizzled_scale=lambda scale: nvfp4_block_scale_interleave(
            scale.view(torch.uint8) if scale.element_size() == 1 else scale
        ),
    )


class NcclAllToAllPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Real routed all2all implemented with torch.distributed.all_to_all_single.

    This keeps the integration local to the modular MoE path and avoids the
    extra connection/bootstrap machinery used by custom transports.
    """

    def __init__(
        self,
        num_dispatchers: int,
        global_to_physical: torch.Tensor | None = None,
        enable_overlap: bool = False,
        tp_size: int = 1,
        overlap_decomposition_reorder: str | None = "johnson",
        overlap_johnson_estimate: str | None = None,
        overlap_comm_alpha: float | None = None,
        overlap_comm_beta: float | None = None,
        overlap_comp_mfu: float | None = None,
        overlap_comp_tflops: float | None = None,
        overlap_comp_mem_bw: float | None = None,
        overlap_johnson_simple_scaler: float | None = None,
    ) -> None:
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.global_to_physical = global_to_physical
        self.enable_overlap = enable_overlap
        self.tp_size = tp_size
        self.overlap_decomposition_reorder = overlap_decomposition_reorder
        self.overlap_johnson_estimate = overlap_johnson_estimate
        self.overlap_comm_alpha = overlap_comm_alpha
        self.overlap_comm_beta = overlap_comm_beta
        self.overlap_comp_mfu = overlap_comp_mfu
        self.overlap_comp_tflops = overlap_comp_tflops
        self.overlap_comp_mem_bw = overlap_comp_mem_bw
        self.overlap_johnson_simple_scaler = overlap_johnson_simple_scaler
        self.handle: _NcclAllToAllHandle | None = None
        self._validated_num_experts: int | None = None
        self._cached_expert_map_key: tuple[int, int] | None = None
        self._cached_local_num_experts: int | None = None
        self._cached_local_to_global_key: tuple[int, int] | None = None
        self._cached_local_to_global: torch.Tensor | None = None
        self._overlap_combine_group: dist.ProcessGroup | None = None
        self._overlap_dispatch_stream: torch.cuda.Stream | None = None
        self._overlap_combine_stream: torch.cuda.Stream | None = None
        self._overlap_output_stream: torch.cuda.Stream | None = None
        self._overlap_compute_streams: list[torch.cuda.Stream] = []
        if self.enable_overlap:
            self._overlap_combine_group = self._get_overlap_combine_group()

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_overlap_execution(self) -> bool:
        return self.enable_overlap

    def _get_overlap_combine_group(self) -> dist.ProcessGroup:
        if self._overlap_combine_group is not None:
            return self._overlap_combine_group

        ep_group = get_ep_group()
        key = tuple(ep_group.ranks)
        combine_group = _OVERLAP_COMBINE_GROUPS.get(key)
        if combine_group is None:
            backend = dist.get_backend(ep_group.device_group)
            combine_group = dist.new_group(ep_group.ranks, backend=backend)
            _OVERLAP_COMBINE_GROUPS[key] = combine_group
        self._overlap_combine_group = combine_group
        return combine_group

    def _get_overlap_dispatch_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._overlap_dispatch_stream is None:
            self._overlap_dispatch_stream = torch.cuda.Stream(device=device)
        return self._overlap_dispatch_stream

    def _get_overlap_combine_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._overlap_combine_stream is None:
            self._overlap_combine_stream = torch.cuda.Stream(device=device)
        return self._overlap_combine_stream

    def _get_overlap_output_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._overlap_output_stream is None:
            self._overlap_output_stream = torch.cuda.Stream(device=device)
        return self._overlap_output_stream

    def _get_overlap_compute_streams(
        self,
        device: torch.device,
        num_units: int,
    ) -> list[torch.cuda.Stream]:
        stream_count = 1 if self.tp_size > 1 else max(num_units, 1)
        if len(self._overlap_compute_streams) < stream_count:
            self._overlap_compute_streams.extend(
                torch.cuda.Stream(device=device)
                for _ in range(stream_count - len(self._overlap_compute_streams))
            )
        return self._overlap_compute_streams[:stream_count]

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids.long()].to(dtype=topk_ids.dtype)

    def _physical_expert_to_rank(
        self, physical_ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        world_size = self.num_dispatchers_
        base = num_experts // world_size
        remainder = num_experts % world_size
        if base == 0:
            return physical_ids.long()
        if remainder == 0:
            return torch.div(physical_ids.long(), base, rounding_mode="floor")
        split = (base + 1) * remainder
        physical_ids_long = physical_ids.long()
        return torch.where(
            physical_ids_long < split,
            torch.div(physical_ids_long, base + 1, rounding_mode="floor"),
            remainder
            + torch.div(physical_ids_long - split, base, rounding_mode="floor"),
        )

    def _validate_static_configuration(self, num_experts: int) -> None:
        if self._validated_num_experts == num_experts:
            return
        if self.global_to_physical is not None:
            max_physical = int(self.global_to_physical.max().item())
            if max_physical >= num_experts:
                raise NotImplementedError(
                    "nccl_alltoall does not support physical expert ids outside "
                    "the [0, num_experts) range. Redundant EPLB experts are not "
                    "supported by this backend yet."
                )
        self._validated_num_experts = num_experts

    def _local_num_experts(
        self, expert_map: torch.Tensor | None, num_experts: int
    ) -> int:
        if expert_map is None:
            return num_experts
        key = (expert_map.data_ptr(), expert_map.numel())
        if key != self._cached_expert_map_key:
            self._cached_expert_map_key = key
            self._cached_local_num_experts = int(
                torch.count_nonzero(expert_map >= 0).item()
            )
        assert self._cached_local_num_experts is not None
        return self._cached_local_num_experts

    def _num_physical_local_experts_for_rank(
        self, rank_in_group: int, num_experts: int
    ) -> int:
        world_size = self.num_dispatchers_
        base = num_experts // world_size
        remainder = num_experts % world_size
        return base + 1 if rank_in_group < remainder else base

    def _physical_expert_to_local(
        self, physical_ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        world_size = self.num_dispatchers_
        base = num_experts // world_size
        remainder = num_experts % world_size
        if base == 0:
            return torch.zeros_like(physical_ids, dtype=torch.long)

        owner_ranks = self._physical_expert_to_rank(physical_ids, num_experts)
        if remainder == 0:
            offsets = owner_ranks.long() * base
        else:
            split = (base + 1) * remainder
            owner_ranks_long = owner_ranks.long()
            offsets = torch.where(
                owner_ranks_long < remainder,
                owner_ranks_long * (base + 1),
                split + (owner_ranks_long - remainder) * base,
            )
        return physical_ids.long() - offsets

    def _local_to_global_expert_ids(
        self,
        expert_map: torch.Tensor | None,
        num_experts: int,
        local_num_experts: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if expert_map is None:
            assert self.global_to_physical is None, (
                "Expected expert_map when global_to_physical remapping is active."
            )
            rank = get_ep_group().rank_in_group
            base = num_experts // self.num_dispatchers_
            remainder = num_experts % self.num_dispatchers_
            if rank < remainder:
                start = rank * (base + 1)
            else:
                start = remainder * (base + 1) + (rank - remainder) * base
            return torch.arange(
                start,
                start + local_num_experts,
                device=device,
                dtype=dtype,
            )

        key = (expert_map.data_ptr(), expert_map.numel())
        if key != self._cached_local_to_global_key:
            local_to_global = torch.full(
                (local_num_experts,),
                -1,
                dtype=torch.long,
                device=expert_map.device,
            )
            valid_mask = expert_map >= 0
            global_ids = torch.arange(
                expert_map.numel(), device=expert_map.device, dtype=torch.long
            )[valid_mask]
            local_ids = expert_map[valid_mask].long()
            local_to_global[local_ids] = global_ids
            self._cached_local_to_global_key = key
            self._cached_local_to_global = local_to_global

        assert self._cached_local_to_global is not None
        return self._cached_local_to_global.to(device=device, dtype=dtype)

    def _all_to_all_counts(self, send_counts_tensor: torch.Tensor) -> torch.Tensor:
        ep_group = get_ep_group()
        group = ep_group.device_group
        assert group is not None
        recv_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=group)
        return recv_counts_tensor

    def _all_to_all_tensor(
        self,
        input_tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        async_op: bool = False,
    ) -> tuple[torch.Tensor, dist.Work | None]:
        group = get_ep_group().device_group
        assert group is not None
        output_shape = (sum(recv_counts),) + input_tensor.shape[1:]
        output = torch.empty(
            output_shape,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        work = dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
            async_op=async_op,
        )
        return output, work

    def _maybe_get_direct_triton_overlap_context(
        self,
        run_fused_experts: Callable[..., torch.Tensor],
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> _DirectTritonOverlapContext | None:
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            return None

        closure = run_fused_experts.__closure__
        if closure is None:
            return None

        closure_vars = {
            name: cell.cell_contents
            for name, cell in zip(run_fused_experts.__code__.co_freevars, closure)
        }
        kernel_impl = closure_vars.get("self")
        if not isinstance(kernel_impl, mk.FusedMoEKernelModularImpl):
            return None

        fused_experts = kernel_impl.fused_experts
        if type(fused_experts) is not TritonExperts:
            return None

        if getattr(fused_experts, "_lora_context", None) is not None:
            return None

        return _DirectTritonOverlapContext(
            kernel_impl=kernel_impl,
            fused_experts=fused_experts,
        )

    @staticmethod
    def _build_grouped_triton_assignment(
        *,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_tokens == 0:
            empty = torch.empty(
                (0,), device=tokens_per_expert.device, dtype=torch.int32
            )
            return (
                empty,
                empty,
                torch.zeros(
                    (1,), device=tokens_per_expert.device, dtype=torch.int32
                ),
            )

        counts = tokens_per_expert.to(dtype=torch.long)
        num_local_experts = counts.numel()
        padded_counts = ((counts + block_size - 1) // block_size) * block_size
        padded_ends = torch.cumsum(padded_counts, dim=0)
        padded_starts = padded_ends - padded_counts
        token_starts = torch.cumsum(counts, dim=0) - counts
        num_tokens_post_padded = padded_counts.sum().to(dtype=torch.int32).view(1)

        max_num_tokens_padded = num_tokens + num_local_experts * (block_size - 1)
        sorted_token_ids = torch.full(
            (max_num_tokens_padded,),
            num_tokens,
            device=tokens_per_expert.device,
            dtype=torch.int32,
        )
        positions = torch.arange(
            max_num_tokens_padded, device=tokens_per_expert.device, dtype=torch.long
        )
        expert_for_position = torch.bucketize(positions, padded_ends, right=True)
        valid_positions = expert_for_position < num_local_experts
        safe_expert_for_position = expert_for_position.clamp(max=num_local_experts - 1)
        local_offsets = positions - padded_starts.index_select(
            0, safe_expert_for_position
        )
        expert_counts = counts.index_select(0, safe_expert_for_position)
        source_offsets = token_starts.index_select(0, safe_expert_for_position)
        valid_tokens = valid_positions & (local_offsets < expert_counts)
        sorted_token_ids[valid_tokens] = (
            source_offsets[valid_tokens] + local_offsets[valid_tokens]
        ).to(dtype=torch.int32)

        max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        expert_ids = torch.full(
            (max_num_blocks,),
            -1,
            device=tokens_per_expert.device,
            dtype=torch.int32,
        )
        block_positions = torch.arange(
            max_num_blocks, device=tokens_per_expert.device, dtype=torch.long
        ) * block_size
        expert_for_block = torch.bucketize(block_positions, padded_ends, right=True)
        valid_blocks = expert_for_block < num_local_experts
        expert_ids[valid_blocks] = expert_for_block[valid_blocks].to(
            dtype=torch.int32
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    def _run_direct_triton_overlap_unit(
        self,
        *,
        context: _DirectTritonOverlapContext,
        recv_hidden_states: torch.Tensor,
        recv_tokens_per_expert: torch.Tensor,
        recv_topk_weights: torch.Tensor | None,
        recv_a1q_scale: torch.Tensor | None,
        activation: mk.MoEActivation,
        w1: torch.Tensor,
        w2: torch.Tensor,
        local_num_experts: int,
        apply_router_weight_on_input: bool,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if recv_hidden_states.size(0) == 0:
            return torch.empty(
                (0, recv_hidden_states.size(-1)),
                device=recv_hidden_states.device,
                dtype=output_dtype,
            )

        fused_experts = context.fused_experts
        kernel_impl = context.kernel_impl
        top_k_num = 1
        num_tokens = recv_hidden_states.size(0)
        n_dim = w1.size(1)
        k_dim = recv_hidden_states.size(-1)

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            fused_experts.quant_config.config_name(recv_hidden_states.dtype),
            num_tokens,
            block_shape=fused_experts.block_shape,
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            self._build_grouped_triton_assignment(
                tokens_per_expert=recv_tokens_per_expert,
                num_tokens=num_tokens,
                block_size=config["BLOCK_SIZE_M"],
            )
        )

        if recv_hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif recv_hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif recv_hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif recv_hidden_states.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(
                f"Unsupported compute_type: {recv_hidden_states.dtype}"
            )

        expert_tokens_meta = overlap.make_expert_tokens_meta(
            recv_tokens_per_expert, recv_hidden_states.device
        )
        workspace13, workspace2, fused_out = kernel_impl._allocate_buffers(
            output_dtype,
            recv_hidden_states.device,
            num_tokens,
            num_tokens,
            n_dim,
            k_dim,
            top_k_num,
            local_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, n_dim))
        cache2_dim = fused_experts.adjust_N_for_activation(n_dim, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, k_dim))

        invoke_fused_moe_triton_kernel(
            recv_hidden_states,
            w1,
            intermediate_cache1,
            recv_a1q_scale if recv_a1q_scale is not None else fused_experts.a1_scale,
            fused_experts.w1_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=fused_experts.quant_config.use_fp8_w8a8,
            use_int8_w8a8=fused_experts.quant_config.use_int8_w8a8,
            use_int8_w8a16=fused_experts.quant_config.use_int8_w8a16,
            use_int4_w4a16=fused_experts.quant_config.use_int4_w4a16,
            per_channel_quant=fused_experts.per_act_token_quant,
            block_shape=fused_experts.block_shape,
            B_bias=fused_experts.w1_bias,
        )

        fused_experts.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, n_dim)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            fused_experts.a2_scale,
            fused_experts.quant_dtype,
            fused_experts.per_act_token_quant,
            fused_experts.block_shape,
            quantization_emulation=fused_experts.quantization_emulation,
        )

        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            fused_experts.w2_scale,
            recv_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=fused_experts.quant_config.use_fp8_w8a8,
            use_int8_w8a8=fused_experts.quant_config.use_int8_w8a8,
            use_int8_w8a16=fused_experts.quant_config.use_int8_w8a16,
            use_int4_w4a16=fused_experts.quant_config.use_int4_w4a16,
            per_channel_quant=fused_experts.per_act_token_quant,
            block_shape=fused_experts.block_shape,
            B_bias=fused_experts.w2_bias,
        )
        fused_experts.moe_sum(intermediate_cache3, fused_out)
        return fused_out

    @staticmethod
    def _needs_scale_dispatch(
        scale_tensor: torch.Tensor | None,
        num_rows: int,
    ) -> bool:
        return (
            scale_tensor is not None
            and scale_tensor.ndim != 0
            and scale_tensor.size(0) == num_rows
        )

    def _all_gather_send_counts(self, send_counts_tensor: torch.Tensor) -> torch.Tensor:
        group = get_ep_group().device_group
        assert group is not None
        gathered = [
            torch.empty_like(send_counts_tensor)
            for _ in range(self.num_dispatchers_)
        ]
        dist.all_gather(gathered, send_counts_tensor, group=group)
        return torch.stack(gathered, dim=0)

    def _batch_p2p_ops(
        self,
        p2p_ops: list[dist.P2POp],
    ) -> list[dist.Work]:
        if not p2p_ops:
            return []
        return dist.batch_isend_irecv(p2p_ops)

    def _estimate_residual_schedule_johnson_cost(
        self,
        *,
        unit_edge_count: int,
        active_rows: tuple[int, ...],
        hidden_size: int,
        hidden_element_size: int,
        intermediate_size: int,
    ) -> tuple[float, float]:
        total_tokens = unit_edge_count * len(active_rows)
        total_token_bytes = total_tokens * hidden_size * hidden_element_size

        alpha = self.overlap_comm_alpha or 0.0
        beta = self.overlap_comm_beta or 0.0
        comm_cost = alpha * 1e-3 + beta * total_token_bytes * 1e3

        match self.overlap_johnson_estimate:
            case "simple":
                scaler = self.overlap_johnson_simple_scaler or 0.0
                return comm_cost, total_token_bytes * scaler
            case "paper":
                if total_tokens == 0:
                    return comm_cost, 0.0
                mfu = self.overlap_comp_mfu or 1.0
                flops_peak = float(self.overlap_comp_tflops or 0.0) * 1e12
                d_model = hidden_size
                d_ff = intermediate_size
                comp_cost = (
                    total_tokens * d_model * d_ff / (flops_peak * mfu)
                    if flops_peak > 0
                    else 0.0
                )
                return comm_cost, comp_cost
            case _:
                raise RuntimeError(
                    "Unsupported overlap Johnson estimate strategy: "
                    f"{self.overlap_johnson_estimate!r}"
                )

    @staticmethod
    def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
        inverse = [-1] * len(permutation)
        for src_rank, dst_rank in enumerate(permutation):
            inverse[dst_rank] = src_rank
        return tuple(inverse)

    @staticmethod
    def _empty_like_prefix(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:0]

    def _build_overlap_units(
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        route_topk_ids: torch.Tensor,
        num_experts: int,
        intermediate_size: int,
    ) -> list[overlap.NcclAllToAllOverlapUnit]:
        ep_group = get_ep_group()
        world_size = self.num_dispatchers_
        flat_route_topk_ids = route_topk_ids.reshape(-1)
        flat_topk_weights = topk_weights.reshape(-1)
        token_indices = torch.arange(a1q.size(0), device=a1q.device, dtype=torch.int64)
        flat_token_indices = token_indices.view(-1, 1).expand_as(topk_ids).reshape(-1)
        dest_ranks = self._physical_expert_to_rank(flat_route_topk_ids, num_experts)
        flat_local_expert_ids = self._physical_expert_to_local(
            flat_route_topk_ids, num_experts
        )
        order = torch.argsort(dest_ranks)
        ordered_token_indices = flat_token_indices.index_select(0, order)
        ordered_local_expert_ids = flat_local_expert_ids.index_select(0, order)
        ordered_topk_weights = flat_topk_weights.index_select(0, order)
        ordered_hidden_states = a1q.index_select(0, ordered_token_indices)
        ordered_a1q_scale = overlap.maybe_make_unit_scales(
            a1q_scale,
            a1q.size(0),
            ordered_token_indices,
        )

        send_counts_tensor = torch.bincount(dest_ranks, minlength=world_size)
        rank_dispatch_token_matrix = self._all_gather_send_counts(send_counts_tensor)
        send_offsets = torch.cumsum(send_counts_tensor, dim=0) - send_counts_tensor
        consumed = torch.zeros_like(send_counts_tensor)

        scale_element_size = (
            0
            if ordered_a1q_scale is None or ordered_a1q_scale.ndim == 0
            else ordered_a1q_scale.element_size()
        )
        schedule = []
        for permutation, unit_edge_count, active_rows in overlap.build_residual_schedule(
            rank_dispatch_token_matrix
        ):
            if (
                self.overlap_decomposition_reorder == "johnson"
                and self.overlap_johnson_estimate is not None
            ):
                comm_estimate, comp_estimate = (
                    self._estimate_residual_schedule_johnson_cost(
                        unit_edge_count=unit_edge_count,
                        active_rows=active_rows,
                        hidden_size=a1q.size(1),
                        hidden_element_size=a1q.element_size(),
                        intermediate_size=intermediate_size,
                    )
                )
            else:
                comm_estimate, comp_estimate = overlap.estimate_unit_cost(
                    unit_edge_count=unit_edge_count,
                    num_active_rows=len(active_rows),
                    hidden_size=a1q.size(1),
                    hidden_element_size=a1q.element_size(),
                    intermediate_size=intermediate_size,
                    scale_element_size=scale_element_size,
                )
            schedule.append(
                (permutation, unit_edge_count, active_rows, comm_estimate, comp_estimate)
            )
        if self.overlap_decomposition_reorder == "johnson":
            schedule = overlap.apply_johnsons_rule(schedule)

        units: list[overlap.NcclAllToAllOverlapUnit] = []
        rank = ep_group.rank_in_group
        for permutation, unit_edge_count, active_rows, comm_estimate, comp_estimate in schedule:
            local_send_dest = permutation[rank]
            local_send_count = unit_edge_count if rank in active_rows else 0
            inverse = self._inverse_permutation(permutation)
            local_recv_src = inverse[rank]
            if local_recv_src not in active_rows:
                local_recv_src = None
            local_recv_count = unit_edge_count if local_recv_src is not None else 0
            dispatch_is_local = local_send_count > 0 and local_send_dest == rank

            send_counts = [0] * world_size
            recv_counts = [0] * world_size
            if local_send_count > 0:
                send_counts[local_send_dest] = local_send_count
            if local_recv_src is not None:
                recv_counts[local_recv_src] = local_recv_count

            if local_send_count > 0:
                start = int(
                    send_offsets[local_send_dest].item()
                    + consumed[local_send_dest].item()
                )
                end = start + local_send_count
                unit_hidden_states = ordered_hidden_states[start:end]
                unit_local_expert_ids = ordered_local_expert_ids[start:end]
                unit_token_indices = ordered_token_indices[start:end]
                unit_topk_weights = ordered_topk_weights[start:end]
                unit_a1q_scale = (
                    None if ordered_a1q_scale is None or ordered_a1q_scale.ndim == 0
                    else ordered_a1q_scale[start:end]
                )
                if unit_local_expert_ids.numel() > 1:
                    unit_order = unit_local_expert_ids.argsort(stable=True)
                    unit_hidden_states = unit_hidden_states.index_select(0, unit_order)
                    unit_local_expert_ids = unit_local_expert_ids.index_select(
                        0, unit_order
                    )
                    unit_token_indices = unit_token_indices.index_select(
                        0, unit_order
                    )
                    unit_topk_weights = unit_topk_weights.index_select(0, unit_order)
                    if unit_a1q_scale is not None and unit_a1q_scale.ndim != 0:
                        unit_a1q_scale = unit_a1q_scale.index_select(0, unit_order)
                send_hidden_states = unit_hidden_states
                sent_token_indices = unit_token_indices
                sent_topk_weights = unit_topk_weights
                send_a1q_scale = unit_a1q_scale
                send_tokens_per_expert = torch.bincount(
                    unit_local_expert_ids,
                    minlength=self._num_physical_local_experts_for_rank(
                        local_send_dest, num_experts
                    ),
                ).to(dtype=torch.long, device=a1q.device)
                consumed[local_send_dest] += local_send_count
            else:
                send_hidden_states = self._empty_like_prefix(ordered_hidden_states)
                sent_token_indices = self._empty_like_prefix(ordered_token_indices)
                sent_topk_weights = self._empty_like_prefix(ordered_topk_weights)
                send_tokens_per_expert = torch.zeros(
                    (
                        self._num_physical_local_experts_for_rank(
                            local_send_dest, num_experts
                        )
                        if local_send_dest is not None
                        else 0
                    ),
                    dtype=torch.long,
                    device=a1q.device,
                )
                send_a1q_scale = (
                    ordered_a1q_scale
                    if ordered_a1q_scale is None or ordered_a1q_scale.ndim == 0
                    else self._empty_like_prefix(ordered_a1q_scale)
                )

            units.append(
                overlap.NcclAllToAllOverlapUnit(
                    send_hidden_states=send_hidden_states,
                    sent_token_indices=sent_token_indices,
                    sent_topk_weights=sent_topk_weights,
                    send_tokens_per_expert=send_tokens_per_expert,
                    send_a1q_scale=send_a1q_scale,
                    dispatch_is_local=dispatch_is_local,
                    dispatch_send_peer_rank=(
                        ep_group.ranks[local_send_dest]
                        if local_send_count > 0 and not dispatch_is_local
                        else None
                    ),
                    dispatch_recv_peer_rank=(
                        ep_group.ranks[local_recv_src]
                        if local_recv_src is not None and not dispatch_is_local
                        else None
                    ),
                    dispatch_recv_token_count=(
                        local_send_count if dispatch_is_local else local_recv_count
                    ),
                    comm_estimate=comm_estimate,
                    comp_estimate=comp_estimate,
                )
            )

        return units

    def _launch_dispatch(
        self,
        unit: overlap.NcclAllToAllOverlapUnit,
        dispatch_stream: torch.cuda.Stream,
        local_num_experts: int,
        dispatch_topk_weights: bool,
    ) -> overlap.NcclAllToAllDispatchTask:
        if unit.dispatch_is_local:
            return overlap.NcclAllToAllDispatchTask(
                unit=unit,
                recv_hidden_states=unit.send_hidden_states,
                recv_tokens_per_expert=unit.send_tokens_per_expert,
                recv_topk_weights=(
                    unit.sent_topk_weights if dispatch_topk_weights else None
                ),
                recv_a1q_scale=unit.send_a1q_scale,
                work_handles=[],
            )

        ep_group = get_ep_group()
        group = ep_group.device_group
        assert group is not None
        recv_token_count = unit.dispatch_recv_token_count

        with torch.cuda.stream(dispatch_stream):
            recv_hidden_states = unit.send_hidden_states.new_empty(
                (recv_token_count, unit.send_hidden_states.size(-1))
            )
            recv_tokens_per_expert = unit.send_tokens_per_expert.new_zeros(
                (local_num_experts,)
            )
            recv_topk_weights = (
                unit.sent_topk_weights.new_empty((recv_token_count,))
                if dispatch_topk_weights
                else None
            )
            recv_a1q_scale = unit.send_a1q_scale
            p2p_ops: list[dist.P2POp] = []

            if unit.dispatch_send_peer_rank is not None:
                p2p_ops.append(
                    dist.P2POp(
                        dist.isend,
                        unit.send_hidden_states,
                        unit.dispatch_send_peer_rank,
                        group,
                    )
                )
            if unit.dispatch_recv_peer_rank is not None and recv_token_count > 0:
                p2p_ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_hidden_states,
                        unit.dispatch_recv_peer_rank,
                        group,
                    )
                )

            if unit.dispatch_send_peer_rank is not None:
                p2p_ops.append(
                    dist.P2POp(
                        dist.isend,
                        unit.send_tokens_per_expert,
                        unit.dispatch_send_peer_rank,
                        group,
                    )
                )
            if unit.dispatch_recv_peer_rank is not None and recv_token_count > 0:
                p2p_ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_tokens_per_expert,
                        unit.dispatch_recv_peer_rank,
                        group,
                    )
                )

            if dispatch_topk_weights:
                if unit.dispatch_send_peer_rank is not None:
                    p2p_ops.append(
                        dist.P2POp(
                            dist.isend,
                            unit.sent_topk_weights,
                            unit.dispatch_send_peer_rank,
                            group,
                        )
                    )
                if unit.dispatch_recv_peer_rank is not None and recv_token_count > 0:
                    assert recv_topk_weights is not None
                    p2p_ops.append(
                        dist.P2POp(
                            dist.irecv,
                            recv_topk_weights,
                            unit.dispatch_recv_peer_rank,
                            group,
                        )
                    )

            if self._needs_scale_dispatch(
                unit.send_a1q_scale, unit.send_hidden_states.size(0)
            ):
                assert unit.send_a1q_scale is not None
                recv_a1q_scale = unit.send_a1q_scale.new_empty(
                    (recv_token_count,) + unit.send_a1q_scale.shape[1:]
                )
                if unit.dispatch_send_peer_rank is not None:
                    p2p_ops.append(
                        dist.P2POp(
                            dist.isend,
                            unit.send_a1q_scale,
                            unit.dispatch_send_peer_rank,
                            group,
                        )
                    )
                if unit.dispatch_recv_peer_rank is not None and recv_token_count > 0:
                    p2p_ops.append(
                        dist.P2POp(
                            dist.irecv,
                            recv_a1q_scale,
                            unit.dispatch_recv_peer_rank,
                            group,
                        )
                    )

            work_handles = self._batch_p2p_ops(p2p_ops)
        return overlap.NcclAllToAllDispatchTask(
            unit=unit,
            recv_hidden_states=recv_hidden_states,
            recv_tokens_per_expert=recv_tokens_per_expert,
            recv_topk_weights=recv_topk_weights,
            recv_a1q_scale=recv_a1q_scale,
            work_handles=work_handles,
        )

    def _launch_combine(
        self,
        unit: overlap.NcclAllToAllOverlapUnit,
        local_output: torch.Tensor,
        combine_stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, list[dist.Work]]:
        if unit.dispatch_is_local:
            return local_output, []

        group = self._get_overlap_combine_group()
        assert group is not None
        recv_token_count = unit.sent_token_indices.numel()

        with torch.cuda.stream(combine_stream):
            recv_output = local_output.new_empty(
                (recv_token_count, local_output.size(-1))
            )
            p2p_ops: list[dist.P2POp] = []
            if (
                unit.dispatch_recv_peer_rank is not None
                and local_output.size(0) > 0
            ):
                p2p_ops.append(
                    dist.P2POp(
                        dist.isend,
                        local_output,
                        unit.dispatch_recv_peer_rank,
                        group,
                    )
                )
            if (
                unit.dispatch_send_peer_rank is not None
                and recv_token_count > 0
            ):
                p2p_ops.append(
                    dist.P2POp(
                        dist.irecv,
                        recv_output,
                        unit.dispatch_send_peer_rank,
                        group,
                    )
                )
            work_handles = self._batch_p2p_ops(p2p_ops)

        return recv_output, work_handles

    def apply_overlap(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        activation: mk.MoEActivation,
        w1: torch.Tensor,
        w2: torch.Tensor,
        local_num_experts: int,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
        run_fused_experts: Callable[..., torch.Tensor],
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> torch.Tensor:
        self._validate_static_configuration(num_experts)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        a1q, a1q_scale = _quantize_and_setup_dispatch(
            hidden_states, quant_config, defer_input_quant
        )
        route_topk_ids = self._map_global_to_physical_ids(topk_ids)

        units = self._build_overlap_units(
            a1q=a1q,
            a1q_scale=a1q_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_topk_ids=route_topk_ids,
            num_experts=num_experts,
            intermediate_size=w2.shape[2],
        )
        output.zero_()
        if not units:
            return output

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        current_stream = torch.cuda.current_stream(device=hidden_states.device)
        dispatch_stream = self._get_overlap_dispatch_stream(hidden_states.device)
        combine_stream = self._get_overlap_combine_stream(hidden_states.device)
        output_stream = self._get_overlap_output_stream(hidden_states.device)
        compute_streams = self._get_overlap_compute_streams(
            hidden_states.device, len(units)
        )
        dispatch_stream.wait_stream(current_stream)
        combine_stream.wait_stream(current_stream)
        output_stream.wait_stream(current_stream)
        output.record_stream(output_stream)
        dispatch_tasks: list[overlap.NcclAllToAllDispatchTask | None] = [
            None
        ] * len(units)
        dispatch_events: list[torch.cuda.Event | None] = [None] * len(units)
        combine_tasks: list[overlap.NcclAllToAllCombineTask | None] = [None] * len(
            units
        )
        combine_events: list[torch.cuda.Event | None] = [None] * len(units)
        dispatch_topk_weights = not apply_router_weight_on_input
        direct_triton_context = self._maybe_get_direct_triton_overlap_context(
            run_fused_experts, weight_and_reduce_impl
        )

        def launch_dispatch(unit_idx: int) -> None:
            with torch.cuda.stream(dispatch_stream):
                dispatch_tasks[unit_idx] = self._launch_dispatch(
                    units[unit_idx],
                    dispatch_stream,
                    local_num_experts,
                    dispatch_topk_weights,
                )
                dispatch_events[unit_idx] = dispatch_stream.record_event()

        for unit_idx in range(len(units)):
            launch_dispatch(unit_idx)

        for unit_idx, unit in enumerate(units):
            compute_stream = compute_streams[unit_idx % len(compute_streams)]
            dispatch_event = dispatch_events[unit_idx]
            assert dispatch_event is not None
            compute_stream.wait_event(dispatch_event)
            with torch.cuda.stream(compute_stream):
                pending_dispatch = dispatch_tasks[unit_idx]
                assert pending_dispatch is not None
                overlap.block_current_stream_on_work_handles(
                    pending_dispatch.work_handles
                )
                recv_hidden_states = pending_dispatch.recv_hidden_states
                recv_tokens_per_expert = pending_dispatch.recv_tokens_per_expert
                recv_hidden_states.record_stream(compute_stream)
                recv_tokens_per_expert.record_stream(compute_stream)
                if pending_dispatch.recv_topk_weights is not None:
                    pending_dispatch.recv_topk_weights.record_stream(compute_stream)
                if (
                    pending_dispatch.recv_a1q_scale is not None
                    and pending_dispatch.recv_a1q_scale.ndim != 0
                ):
                    pending_dispatch.recv_a1q_scale.record_stream(compute_stream)
                recv_a1q_scale = _prepare_scales_for_moe(
                    pending_dispatch.recv_a1q_scale,
                    quant_config,
                )
                recv_topk_weights = (
                    None
                    if pending_dispatch.recv_topk_weights is None
                    else pending_dispatch.recv_topk_weights.to(
                        dtype=topk_weights.dtype
                    ).view(-1, 1)
                )

                if direct_triton_context is not None:
                    local_output = self._run_direct_triton_overlap_unit(
                        context=direct_triton_context,
                        recv_hidden_states=recv_hidden_states,
                        recv_tokens_per_expert=recv_tokens_per_expert,
                        recv_topk_weights=recv_topk_weights,
                        recv_a1q_scale=recv_a1q_scale,
                        activation=activation,
                        w1=w1,
                        w2=w2,
                        local_num_experts=local_num_experts,
                        apply_router_weight_on_input=apply_router_weight_on_input,
                        output_dtype=output.dtype,
                    )
                else:
                    recv_topk_ids = (
                        overlap.reconstruct_local_topk_ids_from_tokens_per_expert(
                            tokens_per_expert=recv_tokens_per_expert,
                            num_local_experts=local_num_experts,
                            dtype=topk_ids.dtype,
                        )
                    )
                    expert_tokens_meta = overlap.make_expert_tokens_meta(
                        recv_tokens_per_expert, recv_hidden_states.device
                    )
                    recv_topk_weights_fallback = (
                        overlap.unit_topk_weights(recv_topk_ids, topk_weights.dtype)
                        if recv_topk_weights is None
                        else recv_topk_weights
                    )
                    fused_expert_output = run_fused_experts(
                        a1q=recv_hidden_states,
                        a1q_scale=recv_a1q_scale,
                        unit_topk_weights=recv_topk_weights_fallback,
                        unit_topk_ids=recv_topk_ids,
                        expert_tokens_meta=expert_tokens_meta,
                        unit_expert_map=None,
                        unit_global_num_experts=local_num_experts,
                    )
                    local_output = overlap.reduce_local_fused_output(
                        weight_and_reduce_impl=weight_and_reduce_impl,
                        fused_expert_output=fused_expert_output,
                        topk_weights=recv_topk_weights_fallback,
                        topk_ids=recv_topk_ids,
                        apply_router_weight_on_input=apply_router_weight_on_input,
                    )
                local_output.record_stream(compute_stream)
                compute_done_event = compute_stream.record_event()

            combine_stream.wait_event(compute_done_event)
            with torch.cuda.stream(combine_stream):
                recv_output, combine_work_handles = self._launch_combine(
                    unit,
                    local_output,
                    combine_stream,
                )
                combine_tasks[unit_idx] = overlap.NcclAllToAllCombineTask(
                    unit=unit,
                    combined_output=recv_output,
                    work_handles=combine_work_handles,
                )
                combine_events[unit_idx] = combine_stream.record_event()

            combine_event = combine_events[unit_idx]
            assert combine_event is not None
            output_stream.wait_event(combine_event)
            with torch.cuda.stream(output_stream):
                combine_task = combine_tasks[unit_idx]
                assert combine_task is not None
                overlap.block_current_stream_on_work_handles(
                    combine_task.work_handles
                )
                recv_output = combine_task.combined_output
                recv_output.record_stream(output_stream)
                output.index_add_(0, unit.sent_token_indices, recv_output)

        current_stream.wait_stream(output_stream)
        return output

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig | None,
        defer_input_quant: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        mk.ExpertTokensMetadata,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._validate_static_configuration(num_experts)

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        route_topk_ids = self._map_global_to_physical_ids(topk_ids)
        flat_route_topk_ids = route_topk_ids.reshape(-1)
        flat_topk_ids = topk_ids.reshape(-1, 1)
        flat_topk_weights = topk_weights.reshape(-1, 1)
        topk = topk_ids.size(1)

        if topk == 1:
            dest_ranks = self._physical_expert_to_rank(
                route_topk_ids.view(-1), num_experts
            )
            order = torch.argsort(dest_ranks)
            send_hidden_states = a1q.index_select(0, order)
            send_topk_ids = flat_topk_ids.index_select(0, order)
            sent_token_indices = order
            sent_topk_weights = flat_topk_weights.index_select(0, order)
            send_a1q_scale = overlap.maybe_make_unit_scales(
                a1q_scale, a1q.size(0), order
            )
        else:
            token_indices = torch.arange(
                a1.size(0), device=a1.device, dtype=torch.int64
            )
            flat_token_indices = token_indices.view(-1, 1).expand_as(topk_ids)
            flat_token_indices = flat_token_indices.reshape(-1)
            dest_ranks = self._physical_expert_to_rank(
                flat_route_topk_ids, num_experts
            )
            order = torch.argsort(dest_ranks)
            ordered_token_indices = flat_token_indices.index_select(0, order)
            send_hidden_states = a1q.index_select(0, ordered_token_indices)
            send_topk_ids = flat_topk_ids.index_select(0, order)
            sent_token_indices = ordered_token_indices
            sent_topk_weights = flat_topk_weights.index_select(0, order)
            send_a1q_scale = overlap.maybe_make_unit_scales(
                a1q_scale, a1q.size(0), ordered_token_indices
            )

        send_counts_tensor = torch.bincount(
            dest_ranks, minlength=self.num_dispatchers_
        )
        recv_counts_tensor = self._all_to_all_counts(send_counts_tensor)
        send_counts = send_counts_tensor.cpu().tolist()
        recv_counts = recv_counts_tensor.cpu().tolist()

        recv_hidden_states, hidden_work = self._all_to_all_tensor(
            send_hidden_states, send_counts, recv_counts, async_op=True
        )
        recv_topk_ids, topk_ids_work = self._all_to_all_tensor(
            send_topk_ids, send_counts, recv_counts, async_op=True
        )
        recv_a1q_scale = None
        scales_work = None
        if self._needs_scale_dispatch(send_a1q_scale, send_hidden_states.size(0)):
            recv_a1q_scale, scales_work = self._all_to_all_tensor(
                send_a1q_scale, send_counts, recv_counts, async_op=True
            )
        else:
            recv_a1q_scale = a1q_scale

        if hidden_work is not None:
            hidden_work.wait()
        if topk_ids_work is not None:
            topk_ids_work.wait()
        if scales_work is not None:
            scales_work.wait()

        recv_a1q_scale = _prepare_scales_for_moe(recv_a1q_scale, quant_config)

        local_num_experts = self._local_num_experts(expert_map, num_experts)
        if expert_map is None:
            local_expert_ids = recv_topk_ids.view(-1)
        else:
            local_expert_ids = expert_map[recv_topk_ids.view(-1).long()].long()

        expert_num_tokens = torch.bincount(
            local_expert_ids, minlength=local_num_experts
        )
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens.to(
                device=recv_hidden_states.device, dtype=torch.int32
            ),
            expert_num_tokens_cpu=expert_num_tokens.to(dtype=torch.int32).cpu(),
        )

        self.handle = _NcclAllToAllHandle(
            send_counts=send_counts,
            recv_counts=recv_counts,
            sent_token_indices=sent_token_indices,
            sent_topk_weights=sent_topk_weights,
        )

        return (
            recv_hidden_states,
            recv_a1q_scale,
            expert_tokens_meta,
            recv_topk_ids,
            (
                torch.ones_like(recv_topk_ids, dtype=topk_weights.dtype)
                if apply_router_weight_on_input
                else torch.ones_like(recv_topk_ids, dtype=topk_weights.dtype)
            ),
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduceDelegate,
    ) -> None:
        assert self.handle is not None

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        local_output = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if local_output.ndim == 3 and local_output.size(1) == 1:
            local_output = local_output.squeeze(1)

        recv_output, reverse_work = self._all_to_all_tensor(
            local_output,
            self.handle.recv_counts,
            self.handle.send_counts,
            async_op=True,
        )
        output.zero_()
        if reverse_work is not None:
            reverse_work.wait()

        if not apply_router_weight_on_input:
            recv_output.mul_(self.handle.sent_topk_weights.view(-1, 1))

        output.index_add_(0, self.handle.sent_token_indices, recv_output)
        self.handle = None
