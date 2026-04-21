# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

# Routing tracking — globals live in layer.py to avoid circular imports.
from vllm.model_executor.layers.fused_moe.layer import (
    _ROUTING_DATA,
    _is_any_routing_capture_enabled,
    _is_tracking_enabled,
)
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
        layer_id: int | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self._layer_id = layer_id

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )
        
        # Capture routing data if offline tracking or placement capture is enabled.
        if _is_any_routing_capture_enabled() and self._layer_id is not None:
            # In DP+EP mode with do_naive_dispatch_combine=True, hidden_states
            # is the all-gathered tensor from ALL DP ranks.  We must slice out
            # only the tokens that belong to this rank to avoid N-fold counting.
            #
            # dp_metadata is only set when dp_size > 1.  When it is set the
            # router always runs inside the sp_local_sizes() context manager
            # (see default_moe_runner.py), so local_sizes is always valid here.
            #
            # The gathered tensor is contiguous by EP rank:
            #   [rank0 tokens | rank1 tokens | … | rank(N-1) tokens]
            # get_ep_group().rank_in_group is the index into local_sizes.
            capture_ids = topk_ids
            capture_weights = topk_weights if _is_tracking_enabled() else None
            capture_num = hidden_states.shape[0]
            capture_prefill_ranges: list[tuple[int, int]] | None = None

            ctx = get_forward_context()
            if ctx is not None:
                capture_prefill_ranges = list(
                    ctx.additional_kwargs.get("routing_capture_prefill_ranges", [])
                )
            if ctx is not None and ctx.dp_metadata is not None:
                try:
                    ep_rank = get_ep_group().rank_in_group
                    sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()
                    start = sum(sizes[:ep_rank])
                    end = start + sizes[ep_rank]
                    # Only slice when the gathered tensor is larger than local.
                    # .clone() makes an independent copy so the full gathered
                    # tensor can be freed rather than being kept alive by a view.
                    if end <= topk_ids.shape[0] and end > start:
                        capture_ids = topk_ids[start:end].clone()
                        if capture_weights is not None:
                            capture_weights = topk_weights[start:end].clone()
                        capture_num = sizes[ep_rank]
                except Exception:
                    capture_prefill_ranges = None

            if self._layer_id not in _ROUTING_DATA:
                _ROUTING_DATA[self._layer_id] = []
            capture_record = {
                'topk_ids': capture_ids.detach(),
                'num_tokens': capture_num,
                'prefill_ranges': capture_prefill_ranges,
            }
            if capture_weights is not None:
                capture_record['topk_weights'] = capture_weights.detach()
            _ROUTING_DATA[self._layer_id].append(capture_record)

        return topk_weights, topk_ids
