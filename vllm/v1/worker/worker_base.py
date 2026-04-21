# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec
# ///////////// Expert-based load balancing
from vllm.v1.prefix_router import TokenRadixTree
# ///////////// Expert-based load balancing

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None
        # ///////////// Expert-based load balancing
        self._prefix_router_tree = TokenRadixTree()
        self._prefix_router_tree_epoch = 0
        self._prefix_router_owner_cache: list[list[tuple[int, ...]]] | None = None
        self._prefix_router_owner_cache_epoch: int | None = None
        # ///////////// Expert-based load balancing

    # ///////////// Expert-based load balancing
    def _prefix_router_enabled(self) -> bool:
        if not self.model_config.enable_prefix_affinity_routing:
            return False

        unsupported_reasons = []
        if self.parallel_config.data_parallel_size <= 1:
            unsupported_reasons.append("data_parallel_size must be greater than 1")
        if not self.parallel_config.enable_expert_parallel:
            unsupported_reasons.append("expert parallelism must be enabled")
        if self.parallel_config.tensor_parallel_size != 1:
            unsupported_reasons.append("tensor_parallel_size must equal 1")
        if self.parallel_config.pipeline_parallel_size != 1:
            unsupported_reasons.append("pipeline_parallel_size must equal 1")
        if self.parallel_config.local_engines_only:
            unsupported_reasons.append(
                "internal DP load balancing must manage all ranks"
            )

        if unsupported_reasons:
            logger.warning_once(
                "Prefix affinity routing is enabled but unsupported for this "
                "serving topology; leaving existing routing unchanged. %s",
                "; ".join(unsupported_reasons),
            )
            return False
        return True

    def _get_prefix_router_epoch(self) -> int:
        try:
            from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

            return StaticPlacementPolicy.get_prefix_router_epoch()
        except Exception:
            return 0

    def _build_prefix_router_owner_cache(self) -> list[list[tuple[int, ...]]]:
        from vllm.model_executor.models.interfaces import is_mixture_of_experts

        model = self.get_model()
        if not is_mixture_of_experts(model):
            return []

        num_layers = model.num_moe_layers
        num_logical_experts = model.num_logical_experts
        num_ranks = self.parallel_config.data_parallel_size

        owner_sets = [
            [set() for _ in range(num_logical_experts)] for _ in range(num_layers)
        ]

        model_runner = self.model_runner
        eplb_state = getattr(model_runner, "eplb_state", None)
        if eplb_state is not None and eplb_state.model_states:
            model_state = next(iter(eplb_state.model_states.values()))
            physical_to_logical = model_state.physical_to_logical_map.detach().cpu()
            slots_per_rank = max(physical_to_logical.shape[1] // num_ranks, 1)
            for layer_idx, logical_ids in enumerate(physical_to_logical.tolist()):
                for physical_idx, logical_id in enumerate(logical_ids):
                    if logical_id < 0:
                        continue
                    owner_rank = min(physical_idx // slots_per_rank, num_ranks - 1)
                    owner_sets[layer_idx][logical_id].add(owner_rank)
        else:
            strategy = self.parallel_config.expert_placement_strategy
            if strategy == "round_robin":
                for layer_idx in range(num_layers):
                    for expert_id in range(num_logical_experts):
                        owner_sets[layer_idx][expert_id].add(expert_id % num_ranks)
            else:
                base = num_logical_experts // num_ranks
                remainder = num_logical_experts % num_ranks
                start_idx = 0
                for rank_idx in range(num_ranks):
                    count = base + (1 if rank_idx < remainder else 0)
                    for expert_id in range(start_idx, start_idx + count):
                        for layer_idx in range(num_layers):
                            owner_sets[layer_idx][expert_id].add(rank_idx)
                    start_idx += count

        return [
            [tuple(sorted(rank_ids)) for rank_ids in layer_sets]
            for layer_sets in owner_sets
        ]

    def _get_prefix_router_owner_cache(self) -> list[list[tuple[int, ...]]]:
        epoch = self._get_prefix_router_epoch()
        if (
            self._prefix_router_owner_cache is None
            or self._prefix_router_owner_cache_epoch != epoch
        ):
            self._prefix_router_owner_cache = self._build_prefix_router_owner_cache()
            self._prefix_router_owner_cache_epoch = epoch
        return self._prefix_router_owner_cache

    def prefix_router_compute_owner(
        self,
        routed_experts: list,
        prompt_token_count: int,
    ) -> dict[str, int] | None:
        if not self._prefix_router_enabled():
            return None
        if not routed_experts or prompt_token_count <= 0:
            return None

        owner_cache = self._get_prefix_router_owner_cache()
        if not owner_cache:
            return None

        num_ranks = self.parallel_config.data_parallel_size
        num_prompt_tokens = min(prompt_token_count, len(routed_experts))
        scores = [0] * num_ranks
        seen_pairs: set[tuple[int, int]] = set()

        for token_layers in routed_experts[:num_prompt_tokens]:
            for layer_idx, layer_experts in enumerate(token_layers[: len(owner_cache)]):
                for raw_expert_id in layer_experts:
                    expert_id = int(raw_expert_id)
                    if expert_id < 0 or expert_id >= len(owner_cache[layer_idx]):
                        continue
                    layer_expert = (layer_idx, expert_id)
                    if layer_expert in seen_pairs:
                        continue
                    seen_pairs.add(layer_expert)
                    for owner_rank in owner_cache[layer_idx][expert_id]:
                        scores[owner_rank] += 1

        if not seen_pairs:
            return None

        target_rank = max(range(num_ranks), key=lambda rank: (scores[rank], -rank))
        return {
            "target_rank": target_rank,
            "epoch": self._get_prefix_router_epoch(),
        }

    def prefix_router_upsert(self, prompt_token_ids: list[int], epoch: int) -> None:
        if not self._prefix_router_enabled() or not prompt_token_ids:
            return

        if epoch > self._prefix_router_tree_epoch:
            self._prefix_router_tree.clear()
            self._prefix_router_tree_epoch = epoch
        elif epoch < self._prefix_router_tree_epoch:
            return

        self._prefix_router_tree.insert(prompt_token_ids)
    # ///////////// Expert-based load balancing

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        Returns:
            The accumulated compilation time in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def get_routing_data(self) -> dict[int, list[dict]]:
        """Return MoE routing data captured during inference."""
        from vllm.model_executor.layers.fused_moe.layer import get_routing_data
        return get_routing_data()

    def clear_routing_data(self) -> None:
        """Clear captured MoE routing data."""
        from vllm.model_executor.layers.fused_moe.layer import clear_routing_data
        clear_routing_data()

    def get_routing_data_serialized(self) -> dict[int, list[dict]]:
        """
        Return MoE routing data with tensors serialized to bytes.

        RPC serialization corrupts tensor objects, so we serialize them
        to raw bytes using torch.save + BytesIO. The bytes survive RPC
        and can be deserialized back to proper PyTorch tensors.
        """
        import io
        import torch
        from vllm.model_executor.layers.fused_moe.layer import get_routing_data

        local_data = get_routing_data()
        if not local_data:
            return {}

        serialized = {}
        for layer_id, captures in local_data.items():
            serialized[layer_id] = []
            for capture in captures:
                buf = io.BytesIO()
                torch.save({
                    'topk_ids': capture['topk_ids'].cpu(),
                    'topk_weights': capture['topk_weights'].cpu(),
                }, buf)
                serialized[layer_id].append({
                    'tensor_bytes': buf.getvalue(),
                    'num_tokens': capture['num_tokens'],
                })
        return serialized

    def drain_step_snapshots_serialized(self) -> list[dict]:
        """Return and clear per-step routing snapshots with serialized tensors.

        Each element of the returned list corresponds to one model forward
        pass (step_idx=0 is the prefill, 1..N are decode steps).  Tensors are
        serialized to bytes so they survive the RPC boundary intact.

        Returns:
            List of dicts::

                {
                    'step_idx': int,
                    'routing': {
                        layer_id: [{'tensor_bytes': bytes, 'num_tokens': int}]
                    }
                }
        """
        import io
        import torch
        from vllm.model_executor.layers.fused_moe.layer import drain_step_snapshots

        raw_snaps = drain_step_snapshots()
        serialized_snaps = []
        for snap in raw_snaps:
            ser_routing: dict = {}
            for layer_id, captures in snap['routing'].items():
                ser_routing[layer_id] = []
                for capture in captures:
                    buf = io.BytesIO()
                    torch.save({
                        'topk_ids': capture['topk_ids'].cpu(),
                        'topk_weights': capture['topk_weights'].cpu(),
                    }, buf)
                    ser_routing[layer_id].append({
                        'tensor_bytes': buf.getvalue(),
                        'num_tokens': capture['num_tokens'],
                    })
            serialized_snaps.append({
                'step_idx': snap['step_idx'],
                'routing': ser_routing,
            })
        return serialized_snaps

    def reset_step_counter(self) -> None:
        """Reset the step counter and clear pending snapshots."""
        from vllm.model_executor.layers.fused_moe.layer import reset_step_counter
        reset_step_counter()

    def register_placement_callback(self, file_path: str, func_name: str) -> None:
        """Load a placement function from an absolute file path and register it.

        Called via ``llm.collective_rpc("register_placement_callback", ...)``
        after LLM initialisation so the worker subprocess loads and caches
        the function in its own address space.

        Args:
            file_path: Absolute path to the Python file containing the function.
            func_name: Name of the callable inside that file.
        """
        # ///////////// Expert-based load balancing
        import importlib.util
        spec = importlib.util.spec_from_file_location("_placement_callback", file_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        callback = getattr(mod, func_name)
        self.model_runner._compute_placement_callback = callback
        try:
            from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy
            StaticPlacementPolicy.set_compute_placement_callback(callback)
        except Exception:
            pass
        # ///////////// Expert-based load balancing

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
