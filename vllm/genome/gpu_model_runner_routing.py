# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import time
from typing import Any, cast

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.models.interfaces import is_mixture_of_experts

logger = init_logger(__name__)


class GenomeRoutingPlacementRunnerMixin:
    # Optional user callback: compute_placement(routing_snapshot) -> dict
    # Set this on the model runner to enable custom placement on EPLB
    # rearrangement steps. The forward path does not compute placement; it only
    # contributes prefill-only co-activation edge counts to the current EPLB
    # window, and the callback runs later from StaticPlacementPolicy during the
    # normal rearrange() cadence.
    _compute_placement_callback: object = None

    def _uses_graph_only_callback_placement(self) -> bool:
        return bool(
            self._compute_placement_callback
            and self.parallel_config.enable_eplb
            and self.parallel_config.eplb_config.policy == "custom"
        )

    def _make_placement_routing_dump_session_dir(
        self,
        dump_dir: str,
        session_name: str | None = None,
    ) -> str:
        if session_name is None:
            session_name = f"session_{time.time_ns()}"
        return os.path.join(dump_dir, session_name)

    def get_runtime_placement_routing_dump_state(self) -> dict[str, Any]:
        dp_rank = getattr(self.parallel_config, "data_parallel_rank", None)
        ep_rank = None
        dump_state: dict[str, Any] = {}
        try:
            ep_rank = int(get_ep_group().rank_in_group)
        except Exception:
            ep_rank = None
        try:
            from vllm.distributed.eplb.policy.custom_policy import (
                StaticPlacementPolicy,
            )

            dump_state = cast(
                dict[str, Any],
                StaticPlacementPolicy.get_placement_routing_dump_state(),
            )
        except Exception:
            dump_state = {}
        return {
            "dump_dir": self.model_config.placement_routing_dump_dir,
            "enabled": bool(dump_state.get("enabled", False)),
            "session_dir": dump_state.get(
                "session_dir", self._placement_routing_dump_session_dir
            ),
            "trigger_index": dump_state.get("trigger_index"),
            "step_in_trigger": None,
            "global_rank": int(getattr(self.parallel_config, "rank", -1)),
            "data_parallel_rank": (
                None if dp_rank is None else int(dp_rank)
            ),
            "ep_rank": ep_rank,
        }

    def get_runtime_moe_dispatch_traffic_dump_state(self) -> dict[str, Any]:
        dp_rank = getattr(self.parallel_config, "data_parallel_rank", None)
        ep_rank = None
        try:
            ep_rank = int(get_ep_group().rank_in_group)
        except Exception:
            ep_rank = None
        return {
            "dump_dir": self.model_config.moe_dispatch_traffic_dump_dir,
            "enabled": bool(self.model_config.moe_dispatch_traffic_dump_dir),
            "global_rank": int(getattr(self.parallel_config, "rank", -1)),
            "data_parallel_rank": (
                None if dp_rank is None else int(dp_rank)
            ),
            "ep_rank": ep_rank,
        }

    def set_runtime_placement_routing_dump_dir(
        self,
        dump_dir: str | None,
        session_name: str | None = None,
    ) -> dict[str, Any]:
        normalized = dump_dir.strip() if isinstance(dump_dir, str) else None
        if normalized == "":
            normalized = None
        self.model_config.placement_routing_dump_dir = normalized
        self._placement_routing_dump_session_dir = (
            self._make_placement_routing_dump_session_dir(
                normalized,
                session_name=session_name,
            )
            if normalized
            else None
        )
        try:
            from vllm.distributed.eplb.policy.custom_policy import (
                StaticPlacementPolicy,
            )

            StaticPlacementPolicy.set_placement_routing_dump_session_dir(
                self._placement_routing_dump_session_dir
            )
        except Exception:
            pass
        return self.get_runtime_placement_routing_dump_state()

    def set_runtime_moe_dispatch_traffic_dump_dir(
        self,
        dump_dir: str | None,
    ) -> dict[str, Any]:
        normalized = dump_dir.strip() if isinstance(dump_dir, str) else None
        if normalized == "":
            normalized = None

        old_dir = self.model_config.moe_dispatch_traffic_dump_dir
        if old_dir != normalized:
            from vllm.model_executor.layers.fused_moe.dispatch_traffic_trace import (
                reset_moe_dispatch_traffic_writer,
            )

            reset_moe_dispatch_traffic_writer(
                old_dir,
                int(getattr(self.parallel_config, "rank", -1)),
            )

        self.model_config.moe_dispatch_traffic_dump_dir = normalized
        return self.get_runtime_moe_dispatch_traffic_dump_state()

    def _filter_routing_snapshot_to_prefill(self, routing_snapshot: dict) -> dict:
        """Build a prefill-only routing snapshot from full-step captures."""
        filtered_snapshot: dict[int, list[dict[str, Any]]] = {}
        for layer_id, captures in routing_snapshot.items():
            filtered_captures: list[dict[str, Any]] = []
            for capture in captures:
                topk_ids = capture["topk_ids"]
                prefill_ranges = capture.get("prefill_ranges")
                if not prefill_ranges:
                    continue

                topk_ids_parts = []
                num_tokens = 0
                for start, end in prefill_ranges:
                    if end <= start:
                        continue
                    start = max(0, int(start))
                    end = min(int(end), int(topk_ids.shape[0]))
                    if end <= start:
                        continue
                    topk_ids_parts.append(topk_ids[start:end])
                    num_tokens += end - start

                if not topk_ids_parts:
                    continue

                filtered_captures.append(
                    {
                        "topk_ids": torch.cat(topk_ids_parts, dim=0),
                        "num_tokens": num_tokens,
                    }
                )

            if filtered_captures:
                filtered_snapshot[layer_id] = filtered_captures
        return filtered_snapshot

    def _build_local_coactivation_edges(
        self,
        routing_snapshot: dict,
        num_layers: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
        """Build compact upper-triangular edge counts for local token co-activation.

        Returns:
            edge_weights: Tensor[num_nodes * (num_nodes - 1) // 2]
            node_activation_counts: Tensor[num_nodes]
            layer_ids:    Sorted layer ids present in routing_snapshot.
            num_nodes:    num_layers * num_experts
        """
        num_nodes = num_layers * num_experts
        num_edges = num_nodes * (num_nodes - 1) // 2
        edge_weights = torch.zeros(num_edges, dtype=torch.int64, device=self.device)
        node_activation_counts = torch.zeros(
            num_nodes, dtype=torch.int64, device=self.device
        )

        if not routing_snapshot:
            return edge_weights, node_activation_counts, list(range(num_layers)), num_nodes

        layer_ids = sorted(routing_snapshot.keys())
        per_layer_ids: list[torch.Tensor] = []
        total_tokens = None
        top_k = None

        for layer_id in layer_ids:
            captures = routing_snapshot[layer_id]
            if not captures:
                return edge_weights, node_activation_counts, layer_ids, num_nodes
            layer_tensor = torch.cat(
                [cap["topk_ids"] for cap in captures], dim=0
            ).to(torch.int64)
            if total_tokens is None:
                total_tokens = layer_tensor.shape[0]
                top_k = layer_tensor.shape[1]
            elif (layer_tensor.shape[0] != total_tokens
                  or layer_tensor.shape[1] != top_k):
                logger.warning(
                    "Skipping co-activation graph build due to inconsistent "
                    "routing tensor shapes across layers."
                )
                return edge_weights, node_activation_counts, layer_ids, num_nodes
            per_layer_ids.append(layer_tensor)

        if total_tokens is None or total_tokens == 0:
            return edge_weights, node_activation_counts, layer_ids, num_nodes

        expert_ids = torch.stack(per_layer_ids, dim=1)  # [tokens, layers, top_k]
        layer_offsets = (
            torch.tensor(layer_ids, dtype=torch.int64, device=self.device)
            .view(len(layer_ids), 1) * num_experts
        )
        flat_nodes = (expert_ids + layer_offsets).reshape(total_tokens, -1)
        if flat_nodes.shape[1] < 2:
            valid_nodes = flat_nodes.reshape(-1)
            valid_nodes = valid_nodes[(valid_nodes >= 0) & (valid_nodes < num_nodes)]
            if valid_nodes.numel() > 0:
                node_activation_counts = torch.bincount(
                    valid_nodes, minlength=num_nodes
                )
            return edge_weights, node_activation_counts, layer_ids, num_nodes

        valid_nodes = flat_nodes.reshape(-1)
        valid_nodes = valid_nodes[(valid_nodes >= 0) & (valid_nodes < num_nodes)]
        if valid_nodes.numel() > 0:
            node_activation_counts = torch.bincount(
                valid_nodes, minlength=num_nodes
            )

        row_idx, col_idx = torch.triu_indices(
            flat_nodes.shape[1], flat_nodes.shape[1], offset=1, device=self.device
        )
        pair_count = row_idx.numel()
        if pair_count == 0:
            return edge_weights, node_activation_counts, layer_ids, num_nodes

        # Chunk pairwise edge construction to cap peak GPU memory usage.
        # Each chunk materializes src/dst/lo/hi/edge_ids as int64 tensors of
        # shape [total_tokens, chunk_pairs], so keep the chunk size small
        # enough to fit comfortably alongside the model forward.
        target_temp_bytes = 32 * 1024 * 1024
        tensors_per_chunk = 5
        bytes_per_pair = max(total_tokens, 1) * 8 * tensors_per_chunk
        chunk_pairs = max(
            1,
            min(pair_count, target_temp_bytes // max(bytes_per_pair, 1)),
        )

        for start in range(0, pair_count, chunk_pairs):
            end = min(start + chunk_pairs, pair_count)
            src_nodes = flat_nodes[:, row_idx[start:end]]
            dst_nodes = flat_nodes[:, col_idx[start:end]]
            lo = torch.minimum(src_nodes, dst_nodes)
            hi = torch.maximum(src_nodes, dst_nodes)
            edge_ids = lo * (2 * num_nodes - lo - 1) // 2 + (hi - lo - 1)
            edge_weights.add_(
                torch.bincount(edge_ids.reshape(-1), minlength=num_edges)
            )

        return edge_weights, node_activation_counts, layer_ids, num_nodes

    def _on_routing_step(self) -> None:
        """Called after each model forward pass, immediately before eplb_step().

        1. Reads the current step's routing data.
        2. Optionally converts prefill-only top-k ids into compact graph edge
           increments for the current EPLB window.
        3. Optionally snapshots the full-step routing data for offline tracking.
        4. Clears _ROUTING_DATA so the next forward pass starts fresh.
        """
        from vllm.model_executor.layers.fused_moe.layer import (
            _is_any_routing_capture_enabled,
            _is_placement_capture_enabled,
            _is_tracking_enabled,
            clear_routing_data,
            get_routing_data,
            push_step_snapshot,
        )

        self._accumulate_prefix_learning_step_capture()
        if not (
            _is_any_routing_capture_enabled()
            or self._uses_graph_only_callback_placement()
        ):
            return

        callback = self._compute_placement_callback
        routing_snapshot = get_routing_data()
        placement_routing_snapshot = routing_snapshot
        if _is_placement_capture_enabled() or callback is not None:
            placement_routing_snapshot = self._filter_routing_snapshot_to_prefill(
                routing_snapshot
            )

        if self._uses_graph_only_callback_placement():
            try:
                if self.eplb_state is not None and self.eplb_state.model_states:
                    first = next(iter(self.eplb_state.model_states.values()))
                    num_layers = first.model.num_moe_layers
                    num_experts = first.model.num_logical_experts
                    (
                        coactivation_edges,
                        node_activation_counts,
                        _,
                        _,
                    ) = self._build_local_coactivation_edges(
                        placement_routing_snapshot, num_layers, num_experts
                    )
                    self.eplb_state.record_coactivation_graph(
                        coactivation_edges,
                        node_activation_counts,
                    )
            except Exception as exc:
                logger.warning(
                    "local co-activation recording raised an exception: %s",
                    exc,
                )

        if _is_tracking_enabled():
            push_step_snapshot()

        clear_routing_data()

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=(
                self.parallel_config.eplb_config.log_balancedness
                and not self._uses_graph_only_callback_placement()
            ),
        )

    def _take_prefix_router_placement_update(self) -> dict[str, object] | None:
        if not self.model_config.enable_prefix_affinity_routing:
            return None
        if self.eplb_state is None or not self.eplb_state.model_states:
            return None

        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        epoch = StaticPlacementPolicy.get_prefix_router_epoch()
        model_state = next(iter(self.eplb_state.model_states.values()))
        physical_to_logical_map = model_state.physical_to_logical_map.detach().cpu()
        physical_to_logical_map_list = physical_to_logical_map.tolist()
        self._refresh_prefix_learning_owner_state(
            physical_to_logical_map_list,
            epoch,
        )

        if self._last_prefix_router_physical_to_logical_map is None:
            self._last_prefix_router_physical_to_logical_map = (
                physical_to_logical_map.clone()
            )
            self._last_prefix_router_placement_epoch = epoch
        elif self._last_prefix_router_placement_epoch == epoch:
            return None
        elif torch.equal(
            self._last_prefix_router_physical_to_logical_map, physical_to_logical_map
        ):
            return None
        else:
            self._last_prefix_router_physical_to_logical_map = (
                physical_to_logical_map.clone()
            )
            self._last_prefix_router_placement_epoch = epoch

        return {
            "epoch": epoch,
            "physical_to_logical_map": physical_to_logical_map_list,
        }

    def _needs_genome_model_runner_output_fields(self) -> bool:
        return bool(
            self.model_config.enable_prefix_affinity_routing
            or self.model_config.enable_return_routed_experts
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _build_genome_model_runner_output_fields(
        self,
        prefill_capture_ranges: list[tuple[int, int]],
        prefill_capture_req_ids: list[str],
    ) -> dict[str, object]:
        routed_experts_step = None
        routed_experts_step_indices = None
        prefix_learning_pairs_by_req = None
        prefix_learning_owner_by_req = None
        async_prefix_learning_owner_by_req = None
        (
            prefix_learning_capture_ranges,
            prefix_learning_capture_req_ids,
        ) = self._get_primary_prefix_learning_capture_inputs(
            prefill_capture_ranges, prefill_capture_req_ids
        )
        if (
            self.model_config.enable_prefix_affinity_routing
            and prefix_learning_capture_req_ids
        ):
            prefix_learning_owner_by_req = (
                self._get_prefix_learning_owners_for_requests(
                    prefix_learning_capture_req_ids
                )
            )
        unresolved_prefix_learning_req_ids = prefix_learning_capture_req_ids
        if prefix_learning_owner_by_req is not None:
            unresolved_prefix_learning_req_ids = [
                req_id for req_id in prefix_learning_capture_req_ids
                if req_id not in prefix_learning_owner_by_req
            ]
        if (
            unresolved_prefix_learning_req_ids
            and self.model_config.enable_prefix_affinity_routing
        ):
            prefix_learning_pairs_by_req = (
                self._get_prefix_learning_pairs_for_requests(
                    unresolved_prefix_learning_req_ids
                )
            )
        if (
            prefix_learning_pairs_by_req is None
            and self.model_config.enable_prefix_affinity_routing
            and self.routed_experts_initialized
        ):
            capturer = RoutedExpertsCapturer.get_instance()
            if (
                capturer is not None
                and prefix_learning_capture_ranges
                and prefix_learning_capture_req_ids
            ):
                compact_pairs = capturer.get_unique_layer_expert_pairs_for_ranges(
                    prefix_learning_capture_ranges
                )
                if compact_pairs is not None:
                    prefix_learning_pairs_by_req = {
                        req_id: pairs.tolist()
                        for req_id, pairs in zip(
                            prefix_learning_capture_req_ids, compact_pairs
                        )
                        if pairs is not None and len(pairs) > 0
                    }
        if (
            self.model_config.enable_return_routed_experts
            and self.routed_experts_initialized
        ):
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                routed_experts_step = capturer.get_captured_experts_for_ranges(
                    prefill_capture_ranges
                )
                if self.slot_mapping is not None and prefill_capture_ranges:
                    routed_experts_step_indices = np.concatenate(
                        [
                            self.slot_mapping[start:end]
                            for start, end in prefill_capture_ranges
                            if end > start
                        ]
                    ).copy()
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        prefix_router_placement_update = self._take_prefix_router_placement_update()
        async_prefix_learning_owner_by_req = (
            self._take_async_prefix_learning_owner_updates()
            if self.model_config.enable_prefix_affinity_routing
            else None
        )
        self._current_prefill_capture_ranges = []
        self._current_prefill_capture_req_ids = []
        return {
            "routed_experts_step": routed_experts_step,
            "routed_experts_step_indices": routed_experts_step_indices,
            "prefix_learning_pairs_by_req": prefix_learning_pairs_by_req,
            "prefix_learning_owner_by_req": prefix_learning_owner_by_req,
            "async_prefix_learning_owner_by_req": (
                async_prefix_learning_owner_by_req
            ),
            "prefix_router_placement_update": (
                prefix_router_placement_update
            ),
        }

    def _build_moe_dispatch_traffic_batch_signature(
        self,
        scheduler_output: Any,
    ) -> str:
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids[:num_reqs]

        hasher = hashlib.sha256()
        hasher.update(b"scheduled-batch-v1")
        hasher.update(len(req_ids).to_bytes(8, "little", signed=False))

        for req_id in req_ids:
            req_index = self.input_batch.req_id_to_index[req_id]
            num_scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])

            req_id_bytes = req_id.encode("utf-8")
            hasher.update(len(req_id_bytes).to_bytes(8, "little", signed=False))
            hasher.update(req_id_bytes)
            hasher.update(num_scheduled.to_bytes(8, "little", signed=False))
            hasher.update(num_computed.to_bytes(8, "little", signed=False))

        return hasher.hexdigest()

    def _build_profile_moe_dispatch_traffic_batch_signature(
        self,
        num_scheduled_tokens: np.ndarray,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"profile-batch-v1")
        hasher.update(len(num_scheduled_tokens).to_bytes(8, "little", signed=False))
        for num_scheduled in num_scheduled_tokens.tolist():
            hasher.update(int(num_scheduled).to_bytes(8, "little", signed=False))
        return hasher.hexdigest()

    def _needs_prefill_capture_metadata(self) -> bool:
        return bool(
            self.model_config.enable_prefix_affinity_routing
            or self.model_config.enable_return_routed_experts
            or self._compute_placement_callback is not None
        )

    def _build_prefill_capture_metadata(
        self,
        common_attn_metadata: Any,
    ) -> tuple[list[tuple[int, int]], list[str]]:
        prefill_capture_ranges: list[tuple[int, int]] = []
        prefill_capture_req_ids: list[str] = []
        if (
            self._needs_prefill_capture_metadata()
            and common_attn_metadata.is_prefilling is not None
        ):
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            num_capture_reqs = min(
                len(common_attn_metadata.is_prefilling),
                query_start_loc_cpu.shape[0] - 1,
                len(self.input_batch.req_ids),
            )
            for req_idx in range(num_capture_reqs):
                if not bool(common_attn_metadata.is_prefilling[req_idx].item()):
                    continue
                start = int(query_start_loc_cpu[req_idx].item())
                end = int(query_start_loc_cpu[req_idx + 1].item())
                if end > start:
                    prefill_capture_ranges.append((start, end))
                    prefill_capture_req_ids.append(self.input_batch.req_ids[req_idx])
        return prefill_capture_ranges, prefill_capture_req_ids

    def _maybe_clear_current_step_routing_data(self) -> None:
        # Clear routing data from the previous step so _ROUTING_DATA always
        # reflects only the current step's captures (prefill or one decode step).
        from vllm.model_executor.layers.fused_moe.layer import (
            _is_any_routing_capture_enabled,
            clear_routing_data,
        )

        if (
            _is_any_routing_capture_enabled()
            or self._uses_graph_only_callback_placement()
        ):
            clear_routing_data()

    def _begin_current_forward_prefix_learning_capture(
        self,
        prefill_capture_ranges: list[tuple[int, int]],
        prefill_capture_req_ids: list[str],
    ) -> None:
        (
            prefix_learning_capture_ranges,
            prefix_learning_capture_req_ids,
        ) = self._get_primary_prefix_learning_capture_inputs(
            prefill_capture_ranges, prefill_capture_req_ids
        )
        self._current_prefill_capture_ranges = list(
            prefix_learning_capture_ranges
        )
        self._current_prefill_capture_req_ids = list(
            prefix_learning_capture_req_ids
        )
        self._begin_prefix_learning_step_capture(
            prefix_learning_capture_ranges,
            prefix_learning_capture_req_ids,
        )

    def _build_additional_forward_context(
        self,
        scheduler_output: Any,
        prefill_capture_ranges: list[tuple[int, int]],
        prefill_capture_req_ids: list[str],
    ) -> dict[str, object]:
        additional_forward_context: dict[str, object] = {
            "moe_dispatch_traffic_batch_signature": (
                self._build_moe_dispatch_traffic_batch_signature(
                    scheduler_output
                )
            ),
        }
        if self._needs_prefill_capture_metadata():
            additional_forward_context.update({
                "routing_capture_prefill_ranges": prefill_capture_ranges,
                "routing_capture_prefill_req_ids": prefill_capture_req_ids,
                "routing_capture_prefill_only_for_placement": bool(
                    self._compute_placement_callback
                ),
                "routing_capture_skip_expert_load": (
                    self._uses_graph_only_callback_placement()
                ),
            })
        return additional_forward_context

    def _build_profile_additional_forward_context(
        self,
        num_scheduled_tokens: np.ndarray,
        prefill_capture_ranges: list[tuple[int, int]],
    ) -> dict[str, object]:
        profile_additional_forward_context: dict[str, object] = {
            "moe_dispatch_traffic_batch_signature": (
                self._build_profile_moe_dispatch_traffic_batch_signature(
                    num_scheduled_tokens
                )
            ),
            "skip_routed_experts_capture": True,
        }
        if self._needs_prefill_capture_metadata():
            profile_additional_forward_context.update({
                "routing_capture_prefill_ranges": prefill_capture_ranges,
                "routing_capture_prefill_req_ids": [],
                "routing_capture_prefill_only_for_placement": bool(
                    self._compute_placement_callback
                ),
                "routing_capture_skip_expert_load": (
                    self._uses_graph_only_callback_placement()
                ),
            })
        return profile_additional_forward_context

    def _resolve_genome_cudagraph_mode_overrides(
        self,
        cudagraph_mode: CUDAGraphMode,
    ) -> CUDAGraphMode:
        if (
            self.parallel_config.enable_expert_parallel
            and self.parallel_config.all2all_backend == "nccl_alltoall"
            and self.parallel_config.data_parallel_size > 1
            and cudagraph_mode != CUDAGraphMode.NONE
        ):
            logger.warning(
                "CUDAGraphMode.%s is not supported with nccl_alltoall "
                "expert-parallel backend; setting cudagraph_mode=NONE.",
                cudagraph_mode.name,
            )
            cudagraph_mode = self.compilation_config.cudagraph_mode = (
                CUDAGraphMode.NONE
            )
        return cudagraph_mode
