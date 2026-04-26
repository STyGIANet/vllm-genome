# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import contextlib
import multiprocessing
import queue
import sys
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.queues import Queue
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
# ///////////// Expert-based load balancing
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    CoordinatorRouteRequest,
    CoordinatorRouteResponse,
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.tensor_ipc import TensorIpcSender
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.prefix_router import (
    build_query_block_extra_keys,
    build_owner_cache_from_physical_to_logical_map,
    compute_owner_from_layer_expert_pairs,
    ExactBlockPrefixIndex,
    TokenRadixTree,
    prepare_block_prefix_query,
)
# ///////////// Expert-based load balancing
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

PREFIX_ROUTER_LEARNING_TIMEOUT_S = 2.0

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                # ///////////// Expert-based load balancing
                if (
                    vllm_config.model_config.enable_prefix_affinity_routing
                    or vllm_config.model_config.enable_kv_block_prefix_routing
                    or vllm_config.model_config.enable_load_score_routing
                ):
                    logger.warning_once(
                        "Internal routing preference scoring is enabled but external DP "
                        "load balancing is active; leaving existing routing unchanged."
                    )
                # ///////////// Expert-based load balancing
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    def reset_encoder_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    async def reset_encoder_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)
        assert result is None

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    # ///////////// Expert-based load balancing
    route_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    route_reply_task: asyncio.Task | None = None
    prefix_learning_queue_task: asyncio.Task | None = None
    # ///////////// Expert-based load balancing
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.shutdown()
        if self.coordinator is not None:
            self.coordinator.shutdown()

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
                self.route_socket,
            )

            tasks = (
                self.output_queue_task,
                self.stats_update_task,
                self.route_reply_task,
                self.prefix_learning_queue_task,
            )

            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
                del self.stats_update_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


@dataclass
class ElasticScalingCache:
    existing_core_engines: list[EngineIdentity]
    num_new_core_engines: int
    pending_notifications: dict[EEPNotificationType, set[int]]


# ///////////// Expert-based load balancing
@dataclass
class InFlightRequestInfo:
    request_id: str
    engine: EngineIdentity
    prompt_token_ids: list[int] | None
    is_shadow: bool = False
    primary_request_id: str | None = None
    expert_affinity_prefill_learned: bool = False
    pending_engine_indices: tuple[int, ...] = ()
    pending_slot_released: bool = False


@dataclass
class QueuedDispatchRequest:
    request: EngineCoreRequest
    dispatched: asyncio.Future[None]
    cancelled: bool = False


@dataclass
class PrefixLearningWorkItem:
    request_id: str
    engine: EngineIdentity
    prompt_token_ids: list[int]
    layer_expert_pairs: Any
    owner: dict[str, int] | None = None


@dataclass
class PrefixLearningContext:
    engine: EngineIdentity
    prompt_token_ids: list[int]


@dataclass
class PrefixRouterUpdate:
    target_rank: int
    epoch: int
    prompt_token_ids: list[int]
# ///////////// Expert-based load balancing


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
    ):
        self.vllm_config = vllm_config

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # State used for data parallel.
            self.engines_running = False
            parallel_config = vllm_config.parallel_config
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            enable_input_socket_handover = parallel_config.enable_elastic_ep

            self.stats_update_address: str | None = None
            # ///////////// Expert-based load balancing
            self.route_query_address: str | None = None
            tensor_queue: Queue | None = None
            if client_addresses:
                # Engines are managed externally to this client.
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                self.stats_update_address = client_addresses.get("stats_update_address")
                self.route_query_address = client_addresses.get("route_query_address")
                # ///////////// Expert-based load balancing
                # Tensor queues passed via client_addresses for multi-API-server case
                tensor_queue = client_addresses.get("tensor_queue")  # type: ignore[assignment]
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )
            else:
                # Engines are managed by this client.
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as (engine_manager, coordinator, addresses, tensor_queue):
                    self.resources.coordinator = coordinator
                    self.resources.engine_manager = engine_manager

                self.stats_update_address = addresses.frontend_stats_publish_address
                # ///////////// Expert-based load balancing
                self.route_query_address = addresses.frontend_route_query_address
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )
                # ///////////// Expert-based load balancing

            # Serialization setup with tensor queues for multimodal tensor IPC.
            tensor_ipc_sender: TensorIpcSender | None = None
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is not None and model_config.multimodal_config is not None:
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)

            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_index
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ZMQ identity of each engine that this client will talk to.
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, _ = sync_input_socket.recv_multipart()
                identities.remove(identity)

            self.core_engine: EngineIdentity = self.core_engines[0]
            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # Start monitoring engine core processes for unexpected failures
            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        if self._finalizer.detach() is not None:
            if self.resources.engine_manager is not None:
                self.resources.engine_manager.shutdown(timeout=timeout)
            self.resources()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    def free_pending_messages(self):
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            # No engine processes to monitor
            return

        engine_processes = engine_manager.processes
        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # logs an error, shuts down the client and invokes the failure
        # callback to inform the engine.
        def monitor_engine_cores():
            sentinels = [proc.sentinel for proc in engine_processes]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            proc_name = next(
                proc.name for proc in engine_processes if proc.sentinel == died[0]
            )
            logger.error(
                "Engine core proc %s died unexpectedly, shutting down client.",
                proc_name,
            )
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    @instrument(span_name="SyncMPClient init")
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)
            return

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        self.add_pending_message(tracker, request)

    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        if self.is_dp:
            self.engines_running = True
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.call_utility("profile", is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.call_utility("reset_encoder_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self) if output_handler else None
        output_socket = resources.output_socket
        assert output_socket is not None

        notification_callback_handler: (
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            and notification_callback_handler is not None
                        ):
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            asyncio.create_task(
                                notification_callback_handler(_self, notification_data)
                            )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    async def resume_scheduler_async(self) -> None:
        await self.call_utility_async("resume_scheduler")

    async def is_scheduler_paused_async(self) -> bool:
        return await self.call_utility_async("is_scheduler_paused")

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def reset_encoder_cache_async(self) -> None:
        await self.call_utility_async("reset_encoder_cache")

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]

        self.eep_scaling_cache: ElasticScalingCache | None = None

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]
                            # Update engine_ranks_managed and count_slice
                            parallel_config = self.vllm_config.parallel_config
                            dp_size = parallel_config.data_parallel_size
                            dp_rank = parallel_config.data_parallel_rank
                            assert dp_rank == 0
                            assert dp_size == new_engine_count
                            assert not (
                                parallel_config.data_parallel_hybrid_lb
                                or parallel_config.data_parallel_external_lb
                            )
                            num_ranks = dp_size
                            self.engine_ranks_managed = list(
                                range(dp_rank, dp_rank + num_ranks)
                            )
                            if len(self.lb_engines) < new_engine_count:
                                self.lb_engines = self.lb_engines + [
                                    [0, 0]
                                    for _ in range(
                                        new_engine_count - len(self.lb_engines)
                                    )
                                ]
                            else:
                                self.lb_engines = self.lb_engines[:new_engine_count]
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    # ///////////// Expert-based load balancing
                    decoded = msgspec.msgpack.decode(buf)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 5
                        and decoded[0] == "PREFIX_ROUTER_UPDATE"
                    ):
                        if hasattr(self, "_handle_prefix_router_coordinator_update"):
                            self._handle_prefix_router_coordinator_update(decoded)
                        continue

                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 3
                        and decoded[0] == "PREFIX_ROUTER_UPDATE_BATCH"
                    ):
                        if hasattr(
                            self, "_handle_prefix_router_coordinator_update_batch"
                        ):
                            self._handle_prefix_router_coordinator_update_batch(
                                decoded
                            )
                        continue
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 3
                        and decoded[0] == "PREFIX_ROUTER_PLACEMENT_UPDATE"
                    ):
                        if hasattr(
                            self, "_handle_prefix_router_placement_coordinator_update"
                        ):
                            self._handle_prefix_router_placement_coordinator_update(
                                decoded
                            )
                        continue
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 3
                        and decoded[0] == "KV_PREFIX_EVENT_BATCH"
                    ):
                        if hasattr(self, "_handle_kv_prefix_coordinator_update"):
                            self._handle_kv_prefix_coordinator_update(decoded)
                        continue
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 5
                        and decoded[0] == "LB_WEIGHT_UPDATE"
                    ):
                        _, _, expert_weight, kv_weight, load_weight = decoded
                        if hasattr(self, "_apply_runtime_load_balancer_weights"):
                            self._apply_runtime_load_balancer_weights(
                                expert_affinity_routing_weight=float(expert_weight),
                                kv_block_prefix_routing_weight=float(kv_weight),
                                load_score_routing_weight=float(load_weight),
                            )
                        continue
                    # ///////////// Expert-based load balancing

                    counts, wave, running = decoded
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        ranks = self.engine_ranks_managed
                        count_slice = slice(ranks[0], ranks[-1] + 1)
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    # ///////////// Expert-based load balancing
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, InFlightRequestInfo] = {}
        self.shadow_req_to_primary: dict[str, str] = {}
        self.primary_req_to_shadows: dict[str, list[str]] = {}

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1

        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count
        self.rank_to_engine_index = {
            rank: idx for idx, rank in enumerate(self.engine_ranks_managed)
        }
        self.expert_affinity_enabled = self._is_prefix_router_supported()
        self.expert_affinity_learning_enabled = (
            self._is_prefix_router_learning_supported()
        )
        self.expert_affinity_send_lock = asyncio.Lock()
        self.kv_block_prefix_enabled = self._is_kv_block_prefix_supported()
        model_config = self.vllm_config.model_config
        self.expert_affinity_weight = (
            model_config.expert_affinity_routing_weight
            if self.expert_affinity_enabled
            else 0.0
        )
        self.prefix_affinity_only_prefill = (
            model_config.prefix_affinity_only_prefill
            if self.expert_affinity_learning_enabled
            else False
        )
        self.kv_block_prefix_weight = (
            model_config.kv_block_prefix_routing_weight
            if self.kv_block_prefix_enabled
            else 0.0
        )
        self.load_score_enabled = model_config.enable_load_score_routing
        self.load_score_weight = (
            model_config.load_score_routing_weight
            if self.load_score_enabled
            else 0.0
        )
        self.load_balancer_debug = model_config.load_balancer_debug
        self.prefix_affinity_learning_queue_size = max(
            1, int(model_config.prefix_affinity_learning_queue_size)
        )
        self.max_pending_requests_per_engine = max(
            0, int(model_config.max_pending_requests_per_engine)
        )
        self.frontend_dispatch_queue_enabled = (
            self.max_pending_requests_per_engine > 0
        )
        self.engine_pending_request_counts = [0] * len(self.core_engines)
        self.queued_dispatch_requests: deque[QueuedDispatchRequest] = deque()
        self.queued_dispatch_by_request_id: dict[str, QueuedDispatchRequest] = {}
        self.dispatch_queue_lock = asyncio.Lock()
        self.dispatch_queue_task: asyncio.Task | None = None
        self.coordinator_routing_enabled = (
            self.route_query_address is not None
            and (
                self.expert_affinity_weight > 0.0
                or self.kv_block_prefix_weight > 0.0
            )
        )
        if self.load_balancer_debug:
            logger.warning(
                "LB debug enabled expert_affinity_enabled=%s kv_block_prefix_enabled=%s "
                "expert_affinity_learning_enabled=%s "
                "load_score_enabled=%s coordinator_routing_enabled=%s "
                "prefix_affinity_learning_queue_size=%d "
                "frontend_dispatch_queue_enabled=%s max_pending_requests_per_engine=%d "
                "route_query_address=%s",
                self.expert_affinity_enabled,
                self.kv_block_prefix_enabled,
                self.expert_affinity_learning_enabled,
                self.load_score_enabled,
                self.coordinator_routing_enabled,
                self.prefix_affinity_learning_queue_size,
                self.frontend_dispatch_queue_enabled,
                self.max_pending_requests_per_engine,
                self.route_query_address,
            )
        self.expert_affinity_epoch = 0
        self.expert_affinity_trees: dict[int, TokenRadixTree] = (
            {}
            if self.coordinator_routing_enabled
            else {
                rank: TokenRadixTree() for rank in self.engine_ranks_managed
            }
        )
        self.prefix_router_owner_cache = None
        self.prefix_router_owner_cache_epoch: int | None = None
        self.kv_block_prefix_indices: dict[int, ExactBlockPrefixIndex] = (
            {}
            if self.coordinator_routing_enabled
            else {
                rank: ExactBlockPrefixIndex() for rank in self.engine_ranks_managed
            }
        )
        self.kv_block_prefix_decoder = MsgpackDecoder(KVEventBatch)
        scheduler_block_size = self.vllm_config.cache_config.block_size
        scheduler_block_size *= (
            self.vllm_config.parallel_config.decode_context_parallel_size
        )
        scheduler_block_size *= (
            self.vllm_config.parallel_config.prefill_context_parallel_size
        )
        self.kv_block_prefix_block_size = scheduler_block_size
        self.route_reply_decoder = MsgpackDecoder(CoordinatorRouteResponse)
        self.route_request_futures: dict[int, asyncio.Future[CoordinatorRouteResponse]] = {}
        self.route_call_id = 0
        self.route_send_lock = asyncio.Lock()
        self.prefix_learning_queue: deque[PrefixLearningWorkItem] = deque()
        self.prefix_learning_queue_lock = asyncio.Lock()
        self.prefix_learning_context_by_request: dict[str, PrefixLearningContext] = {}

        if self.coordinator_routing_enabled:
            assert self.route_query_address is not None
            self.resources.route_socket = make_zmq_socket(
                self.ctx,
                self.route_query_address,
                zmq.DEALER,
                bind=False,
            )
            try:
                asyncio.get_running_loop()
                self._ensure_route_reply_task()
            except RuntimeError:
                pass

    def _get_runtime_load_balancer_weights(self) -> dict[str, Any]:
        return {
            "expert_affinity_routing_weight": self.expert_affinity_weight,
            "kv_block_prefix_routing_weight": self.kv_block_prefix_weight,
            "load_score_routing_weight": self.load_score_weight,
            "expert_affinity_enabled": self.expert_affinity_enabled,
            "kv_block_prefix_enabled": self.kv_block_prefix_enabled,
            "load_score_enabled": self.load_score_enabled,
            "coordinator_routing_enabled": self.coordinator_routing_enabled,
        }

    def get_runtime_load_balancer_weights(self) -> dict[str, Any]:
        return self._get_runtime_load_balancer_weights()

    def _apply_runtime_load_balancer_weights(
        self,
        *,
        expert_affinity_routing_weight: float | None = None,
        kv_block_prefix_routing_weight: float | None = None,
        load_score_routing_weight: float | None = None,
    ) -> dict[str, Any]:
        if expert_affinity_routing_weight is not None:
            if expert_affinity_routing_weight < 0.0:
                raise ValueError(
                    "expert_affinity_routing_weight must be non-negative"
                )
            if (
                expert_affinity_routing_weight > 0.0
                and not self.expert_affinity_enabled
            ):
                raise ValueError(
                    "Prefix-affinity routing was not enabled at startup"
                )
            self.expert_affinity_weight = (
                float(expert_affinity_routing_weight)
                if self.expert_affinity_enabled
                else 0.0
            )

        if kv_block_prefix_routing_weight is not None:
            if kv_block_prefix_routing_weight < 0.0:
                raise ValueError(
                    "kv_block_prefix_routing_weight must be non-negative"
                )
            if (
                kv_block_prefix_routing_weight > 0.0
                and not self.kv_block_prefix_enabled
            ):
                raise ValueError(
                    "KV block-prefix routing was not enabled at startup"
                )
            self.kv_block_prefix_weight = (
                float(kv_block_prefix_routing_weight)
                if self.kv_block_prefix_enabled
                else 0.0
            )

        if load_score_routing_weight is not None:
            if load_score_routing_weight < 0.0:
                raise ValueError("load_score_routing_weight must be non-negative")
            if load_score_routing_weight > 0.0 and not self.load_score_enabled:
                raise ValueError(
                    "Load-score routing was not enabled at startup"
                )
            self.load_score_weight = (
                float(load_score_routing_weight)
                if self.load_score_enabled
                else 0.0
            )

        self.coordinator_routing_enabled = (
            self.route_query_address is not None
            and (
                self.expert_affinity_weight > 0.0
                or self.kv_block_prefix_weight > 0.0
            )
        )
        if self.coordinator_routing_enabled and self.resources.route_socket is None:
            assert self.route_query_address is not None
            self.resources.route_socket = make_zmq_socket(
                self.ctx,
                self.route_query_address,
                zmq.DEALER,
                bind=False,
            )
            try:
                asyncio.get_running_loop()
                self._ensure_route_reply_task()
            except RuntimeError:
                pass

        return self._get_runtime_load_balancer_weights()

    async def update_runtime_load_balancer_weights(
        self,
        *,
        expert_affinity_routing_weight: float | None = None,
        kv_block_prefix_routing_weight: float | None = None,
        load_score_routing_weight: float | None = None,
    ) -> dict[str, Any]:
        state = self._apply_runtime_load_balancer_weights(
            expert_affinity_routing_weight=expert_affinity_routing_weight,
            kv_block_prefix_routing_weight=kv_block_prefix_routing_weight,
            load_score_routing_weight=load_score_routing_weight,
        )
        self._ensure_stats_update_task()
        socket = self.resources.stats_update_socket
        if socket is None:
            await asyncio.sleep(0)
            socket = self.resources.stats_update_socket
        if socket is not None:
            await socket.send(
                msgspec.msgpack.encode(
                    (
                        "LB_WEIGHT_UPDATE",
                        self.client_index,
                        self.expert_affinity_weight,
                        self.kv_block_prefix_weight,
                        self.load_score_weight,
                    )
                )
            )
        return state

    def _should_broadcast_ep_request(self) -> bool:
        parallel_config = self.vllm_config.parallel_config
        return (
            parallel_config.enable_expert_parallel
            and parallel_config.data_parallel_size > 1
        )

    def _engine_index_for_identity(self, engine: EngineIdentity) -> int:
        for idx, candidate in enumerate(self.core_engines):
            if candidate == engine:
                return idx
        raise ValueError("Unknown engine identity")

    def _pending_engine_indices_for_engine(
        self,
        engine: EngineIdentity,
    ) -> tuple[int, ...]:
        if self._should_broadcast_ep_request():
            return tuple(range(len(self.core_engines)))
        return (self._engine_index_for_identity(engine),)

    def _can_reserve_pending_indices(self, engine_indices: Sequence[int]) -> bool:
        if not self.frontend_dispatch_queue_enabled:
            return True
        return all(
            self.engine_pending_request_counts[idx]
            < self.max_pending_requests_per_engine
            for idx in engine_indices
        )

    def _reserve_pending_indices(self, engine_indices: Sequence[int]) -> None:
        if not self.frontend_dispatch_queue_enabled:
            return
        for idx in engine_indices:
            self.engine_pending_request_counts[idx] += 1

    def _release_pending_indices(self, engine_indices: Sequence[int]) -> None:
        if not self.frontend_dispatch_queue_enabled:
            return
        for idx in engine_indices:
            if self.engine_pending_request_counts[idx] > 0:
                self.engine_pending_request_counts[idx] -= 1

    def _ensure_dispatch_queue_task(self) -> None:
        if not self.frontend_dispatch_queue_enabled:
            return
        task = self.dispatch_queue_task
        if task is None or task.done():
            self.dispatch_queue_task = asyncio.create_task(
                self._drain_dispatch_queue(),
                name="FrontendDispatchQueue",
            )

    async def _drain_dispatch_queue(self) -> None:
        while True:
            item: QueuedDispatchRequest | None = None
            async with self.dispatch_queue_lock:
                while self.queued_dispatch_requests:
                    head = self.queued_dispatch_requests[0]
                    if not head.cancelled:
                        break
                    self.queued_dispatch_requests.popleft()
                    self.queued_dispatch_by_request_id.pop(
                        head.request.request_id, None
                    )
                if not self.queued_dispatch_requests:
                    self.dispatch_queue_task = None
                    return

                item = self.queued_dispatch_requests[0]

            assert item is not None
            if item.cancelled:
                continue

            chosen_engine = await self._select_core_engine_for_request_async(
                item.request
            )
            pending_engine_indices = self._pending_engine_indices_for_engine(
                chosen_engine
            )

            async with self.dispatch_queue_lock:
                while self.queued_dispatch_requests:
                    head = self.queued_dispatch_requests[0]
                    if not head.cancelled:
                        break
                    self.queued_dispatch_requests.popleft()
                    self.queued_dispatch_by_request_id.pop(
                        head.request.request_id, None
                    )

                if (
                    not self.queued_dispatch_requests
                    or self.queued_dispatch_requests[0] is not item
                    or item.cancelled
                ):
                    continue

                if not self._can_reserve_pending_indices(pending_engine_indices):
                    self.dispatch_queue_task = None
                    return

                self.queued_dispatch_requests.popleft()
                self.queued_dispatch_by_request_id.pop(item.request.request_id, None)
                self._reserve_pending_indices(pending_engine_indices)

            try:
                await self._dispatch_request_to_engine(
                    item.request,
                    chosen_engine,
                    pending_engine_indices,
                )
            except Exception as exc:
                async with self.dispatch_queue_lock:
                    self._release_pending_indices(pending_engine_indices)
                if not item.dispatched.done():
                    item.dispatched.set_exception(exc)
            else:
                if not item.dispatched.done():
                    item.dispatched.set_result(None)

    def _ensure_prefix_learning_queue_task(self) -> None:
        if not self.expert_affinity_learning_enabled:
            return
        resources = self.resources
        task = resources.prefix_learning_queue_task
        if task is None or task.done():
            resources.prefix_learning_queue_task = asyncio.create_task(
                self._drain_prefix_learning_queue(),
                name="PrefixAffinityLearningQueue",
            )

    def _take_prefix_learning_context(
        self,
        request_id: str,
        request_info: InFlightRequestInfo | None = None,
    ) -> PrefixLearningContext | None:
        context = self.prefix_learning_context_by_request.pop(request_id, None)
        if context is not None:
            return context
        if request_info is None or request_info.prompt_token_ids is None:
            return None
        return PrefixLearningContext(
            engine=request_info.engine,
            prompt_token_ids=list(request_info.prompt_token_ids),
        )

    async def _enqueue_prefix_learning_work_item(
        self,
        item: PrefixLearningWorkItem,
    ) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        dropped_item: PrefixLearningWorkItem | None = None
        queue_size = 0
        async with self.prefix_learning_queue_lock:
            if len(self.prefix_learning_queue) >= self.prefix_affinity_learning_queue_size:
                dropped_item = self.prefix_learning_queue.pop()
            self.prefix_learning_queue.appendleft(item)
            queue_size = len(self.prefix_learning_queue)

        if self.load_balancer_debug:
            logger.warning(
                "Prefix affinity learning queue enqueue request_id=%s size=%d/%d",
                item.request_id,
                queue_size,
                self.prefix_affinity_learning_queue_size,
            )
            if dropped_item is not None:
                logger.warning(
                    "Prefix affinity learning queue dropped_oldest request_id=%s max_size=%d",
                    dropped_item.request_id,
                    self.prefix_affinity_learning_queue_size,
                )

        self._ensure_prefix_learning_queue_task()

    async def _drain_prefix_learning_queue(self) -> None:
        resources = self.resources
        while True:
            items: list[PrefixLearningWorkItem] = []
            async with self.prefix_learning_queue_lock:
                if not self.prefix_learning_queue:
                    resources.prefix_learning_queue_task = None
                    return

                batch_size = min(len(self.prefix_learning_queue), 32)
                for _ in range(batch_size):
                    items.append(self.prefix_learning_queue.popleft())

            pending_updates: list[PrefixRouterUpdate] = []
            for item in items:
                try:
                    update = await self._learn_prefix_router_update(
                        request_id=item.request_id,
                        engine=item.engine,
                        prompt_token_ids=item.prompt_token_ids,
                        layer_expert_pairs=item.layer_expert_pairs,
                        owner=item.owner,
                    )
                except Exception as exc:
                    logger.warning(
                        "Prefix affinity learning failed request_id=%s error=%s",
                        item.request_id,
                        exc,
                    )
                    continue

                if update is not None:
                    pending_updates.append(update)

            if not pending_updates:
                continue

            try:
                await self._send_prefix_router_update_batch(pending_updates)
            except asyncio.TimeoutError:
                request_ids = ",".join(
                    item.request_id for item in items[:4]
                )
                logger.warning(
                    "Prefix affinity learning timed out step=send_update_batch "
                    "size=%d sample_request_ids=%s",
                    len(pending_updates),
                    request_ids,
                )

    async def _release_pending_slots_for_request(self, request_id: str) -> None:
        if not self.frontend_dispatch_queue_enabled:
            return

        request_info = self.reqs_in_flight.get(request_id)
        if request_info is None:
            return
        if request_info.is_shadow:
            primary_request_id = request_info.primary_request_id
            if primary_request_id is None:
                return
            request_id = primary_request_id

        async with self.dispatch_queue_lock:
            current = self.reqs_in_flight.get(request_id)
            if current is None or current.pending_slot_released:
                return
            self._release_pending_indices(current.pending_engine_indices)
            current.pending_slot_released = True

        self._ensure_dispatch_queue_task()

    async def _select_core_engine_for_request_async(
        self,
        request: EngineCoreRequest,
    ) -> EngineIdentity:
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            if self.coordinator_routing_enabled:
                self._ensure_route_reply_task()
                route_socket = self.resources.route_socket
                assert route_socket is not None
                loop = asyncio.get_running_loop()
                call_id = self.route_call_id
                self.route_call_id += 1
                route_request = self._build_coordinator_route_request(call_id, request)
                future = loop.create_future()
                self.route_request_futures[call_id] = future
                try:
                    if self.load_balancer_debug:
                        logger.warning(
                            "LB route query send request_id=%s call_id=%d",
                            request.request_id,
                            call_id,
                        )
                    async with self.route_send_lock:
                        await asyncio.wait_for(
                            route_socket.send(msgspec.msgpack.encode(route_request)),
                            timeout=1.0,
                        )
                    if self.load_balancer_debug:
                        logger.warning(
                            "LB route query sent request_id=%s call_id=%d",
                            request.request_id,
                            call_id,
                        )
                    response = await asyncio.wait_for(future, timeout=5.0)
                except Exception as exc:
                    self.route_request_futures.pop(call_id, None)
                    logger.warning(
                        "Coordinator route query failed for request_id=%s; "
                        "falling back to local scoring. error=%s",
                        request.request_id,
                        exc,
                    )
                    return self._select_core_engine_for_request(request)
                eng_index = response.engine_index
            else:
                return self._select_core_engine_for_request(request)

        return self.core_engines[eng_index]

    def _select_core_engine_for_request(
        self,
        request: EngineCoreRequest,
    ) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            expert_scores = (
                self._get_expert_affinity_scores(request.prompt_token_ids)
                if self.expert_affinity_weight > 0.0
                else {}
            )
            kv_scores = (
                self._get_kv_block_prefix_scores(request)
                if self.kv_block_prefix_weight > 0.0
                else {}
            )
            load_scores = (
                self._get_load_scores()
                if self.load_score_weight > 0.0
                else {}
            )
            score_by_index: dict[int, float] = {}
            if self.expert_affinity_weight > 0.0:
                for idx, score in expert_scores.items():
                    score_by_index[idx] = score_by_index.get(idx, 0.0) + (
                        self.expert_affinity_weight * score
                    )
            if self.kv_block_prefix_weight > 0.0:
                for idx, score in kv_scores.items():
                    score_by_index[idx] = score_by_index.get(idx, 0.0) + (
                        self.kv_block_prefix_weight * score
                    )
            if self.load_score_weight > 0.0:
                for idx, score in load_scores.items():
                    score_by_index[idx] = score_by_index.get(idx, 0.0) + (
                        self.load_score_weight * score
                    )

            if score_by_index:
                best_score = max(score_by_index.values())
                matched_indices = [
                    idx for idx, score in score_by_index.items() if score == best_score
                ]
                eng_index = self._choose_engine_by_load(matched_indices)
                if self.load_balancer_debug:
                    logger.warning(
                        "LB route request_id=%s expert_scores=%s kv_scores=%s "
                        "load_scores=%s final_scores=%s chosen_engine=%d best_score=%.4f",
                        request.request_id,
                        self._format_score_map(expert_scores),
                        self._format_score_map(kv_scores),
                        self._format_score_map(load_scores),
                        self._format_score_map(score_by_index),
                        eng_index,
                        best_score,
                    )
            else:
                eng_index = self._choose_engine_by_load()
                if self.load_balancer_debug:
                    logger.warning(
                        "LB route request_id=%s expert_scores=%s kv_scores=%s "
                        "load_scores=%s final_scores={} chosen_engine=%d fallback=load_only",
                        request.request_id,
                        self._format_score_map(expert_scores),
                        self._format_score_map(kv_scores),
                        self._format_score_map(load_scores),
                        eng_index,
                    )

        return self.core_engines[eng_index]

    def _make_shadow_request_id(self, request_id: str, eng_idx: int) -> str:
        return f"{request_id}::shadow::{eng_idx}"

    def _register_shadow_request(
        self,
        primary_request_id: str,
        shadow_request_id: str,
        engine: EngineIdentity,
        prompt_token_ids: list[int] | None,
    ) -> None:
        self.shadow_req_to_primary[shadow_request_id] = primary_request_id
        self.primary_req_to_shadows.setdefault(primary_request_id, []).append(
            shadow_request_id
        )
        self.reqs_in_flight[shadow_request_id] = InFlightRequestInfo(
            request_id=shadow_request_id,
            engine=engine,
            prompt_token_ids=prompt_token_ids,
            is_shadow=True,
            primary_request_id=primary_request_id,
        )

    def _cleanup_shadow_request(self, shadow_request_id: str) -> None:
        primary_request_id = self.shadow_req_to_primary.pop(shadow_request_id, None)
        self.reqs_in_flight.pop(shadow_request_id, None)
        if primary_request_id is None:
            return
        shadow_ids = self.primary_req_to_shadows.get(primary_request_id)
        if not shadow_ids:
            return
        remaining_shadow_ids = [
            req_id for req_id in shadow_ids if req_id != shadow_request_id
        ]
        if remaining_shadow_ids:
            self.primary_req_to_shadows[primary_request_id] = remaining_shadow_ids
        else:
            self.primary_req_to_shadows.pop(primary_request_id, None)

    async def _abort_shadow_requests(self, primary_request_id: str) -> None:
        shadow_ids = self.primary_req_to_shadows.pop(primary_request_id, [])
        if not shadow_ids:
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for shadow_req_id in shadow_ids:
            if request_info := self.reqs_in_flight.get(shadow_req_id):
                by_engine[request_info.engine].append(shadow_req_id)
            self.shadow_req_to_primary.pop(shadow_req_id, None)
            self.reqs_in_flight.pop(shadow_req_id, None)

        if by_engine:
            await asyncio.gather(
                *[
                    self._abort_requests(req_ids, engine)
                    for engine, req_ids in by_engine.items()
                ]
            )

    def _is_prefix_router_supported(self) -> bool:
        if not self.vllm_config.model_config.enable_prefix_affinity_routing:
            return False
        return self._is_prefix_router_learning_supported()

    def _is_prefix_router_learning_supported(self) -> bool:
        if not self.vllm_config.model_config.enable_prefix_affinity_routing:
            return False

        parallel_config = self.vllm_config.parallel_config
        unsupported_reasons = []
        if parallel_config.data_parallel_size <= 1:
            unsupported_reasons.append("data_parallel_size must be greater than 1")
        if not parallel_config.enable_expert_parallel:
            unsupported_reasons.append("expert parallelism must be enabled")
        if parallel_config.tensor_parallel_size != 1:
            unsupported_reasons.append("tensor_parallel_size must equal 1")
        if parallel_config.pipeline_parallel_size != 1:
            unsupported_reasons.append("pipeline_parallel_size must equal 1")
        if parallel_config.local_engines_only:
            unsupported_reasons.append(
                "internal DP load balancing must manage all ranks"
            )

        if unsupported_reasons:
            logger.warning_once(
                "Prefix affinity learning is enabled but unsupported for this "
                "serving topology; leaving existing behavior unchanged. %s",
                "; ".join(unsupported_reasons),
            )
            return False
        return True

    def _is_kv_block_prefix_supported(self) -> bool:
        if not self.vllm_config.model_config.enable_kv_block_prefix_routing:
            return False

        parallel_config = self.vllm_config.parallel_config
        cache_config = self.vllm_config.cache_config
        model_config = self.vllm_config.model_config
        unsupported_reasons = []
        if parallel_config.data_parallel_size <= 1:
            unsupported_reasons.append("data_parallel_size must be greater than 1")
        if parallel_config.tensor_parallel_size != 1:
            unsupported_reasons.append("tensor_parallel_size must equal 1")
        if parallel_config.pipeline_parallel_size != 1:
            unsupported_reasons.append("pipeline_parallel_size must equal 1")
        if parallel_config.local_engines_only:
            unsupported_reasons.append(
                "internal DP load balancing must manage all ranks"
            )
        if not cache_config.enable_prefix_caching:
            unsupported_reasons.append("prefix caching must be enabled")
        if cache_config.sliding_window is not None:
            unsupported_reasons.append("sliding-window attention is not supported")
        if model_config.is_encoder_decoder:
            unsupported_reasons.append("encoder-decoder models are not supported")

        if unsupported_reasons:
            logger.warning_once(
                "KV block-prefix routing is enabled but unsupported for this "
                "serving topology; leaving existing load balancing unchanged. %s",
                "; ".join(unsupported_reasons),
            )
            return False
        return True

    def _ensure_route_reply_task(self) -> None:
        resources = self.resources
        if not self.coordinator_routing_enabled or resources.route_reply_task is not None:
            return

        route_socket = resources.route_socket
        assert route_socket is not None
        assert isinstance(route_socket, zmq.asyncio.Socket)

        async def process_route_replies():
            try:
                while True:
                    buffer = await route_socket.recv()
                    response = self.route_reply_decoder.decode(buffer)
                    future = self.route_request_futures.pop(response.call_id, None)
                    if future is not None and not future.done():
                        future.set_result(response)
            except asyncio.CancelledError:
                for future in self.route_request_futures.values():
                    if not future.done():
                        future.set_exception(EngineDeadError())
                self.route_request_futures.clear()
            except Exception as exc:
                for future in self.route_request_futures.values():
                    if not future.done():
                        future.set_exception(exc)
                self.route_request_futures.clear()

        resources.route_reply_task = asyncio.create_task(
            process_route_replies(),
            name="CoordinatorRouteReplyTask",
        )

    def _clear_prefix_router_state(self, epoch: int) -> None:
        for tree in self.expert_affinity_trees.values():
            tree.clear()
        self.expert_affinity_epoch = epoch

    def _apply_prefix_router_placement_update(
        self,
        epoch: int,
        physical_to_logical_map: Sequence[Sequence[int]],
    ) -> None:
        if not self.expert_affinity_learning_enabled:
            return
        if epoch > self.expert_affinity_epoch:
            self._clear_prefix_router_state(epoch)
        elif epoch < self.expert_affinity_epoch:
            return
        self.prefix_router_owner_cache = build_owner_cache_from_physical_to_logical_map(
            physical_to_logical_map,
            len(self.core_engines),
        )
        self.prefix_router_owner_cache_epoch = epoch

    def _apply_prefix_router_update(
        self,
        target_rank: int,
        epoch: int,
        prompt_token_ids: list[int],
    ) -> None:
        if not self.expert_affinity_learning_enabled or not prompt_token_ids:
            return
        if epoch > self.expert_affinity_epoch:
            self._clear_prefix_router_state(epoch)
        elif epoch < self.expert_affinity_epoch:
            return
        if target_rank not in self.expert_affinity_trees:
            self.expert_affinity_trees[target_rank] = TokenRadixTree()
        self.expert_affinity_trees[target_rank].insert(prompt_token_ids)

    def _handle_prefix_router_coordinator_update(self, decoded: Sequence[Any]) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, source_client_index, target_rank, epoch, prompt_token_ids = decoded
        if int(source_client_index) == self.client_index:
            return
        self._apply_prefix_router_update(
            int(target_rank),
            int(epoch),
            list(prompt_token_ids),
        )

    def _handle_prefix_router_coordinator_update_batch(
        self, decoded: Sequence[Any]
    ) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, source_client_index, updates = decoded
        if int(source_client_index) == self.client_index:
            return

        for update in updates:
            if len(update) != 3:
                continue
            target_rank, epoch, prompt_token_ids = update
            self._apply_prefix_router_update(
                int(target_rank),
                int(epoch),
                list(prompt_token_ids),
            )

    def _handle_prefix_router_placement_coordinator_update(
        self, decoded: Sequence[Any]
    ) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, epoch, physical_to_logical_map = decoded
        self._apply_prefix_router_placement_update(
            int(epoch),
            list(physical_to_logical_map),
        )

    def _apply_kv_prefix_event_batch(
        self,
        target_rank: int,
        batch: KVEventBatch,
    ) -> None:
        if not self.kv_block_prefix_enabled:
            return

        index = self.kv_block_prefix_indices.get(target_rank)
        if index is None:
            index = ExactBlockPrefixIndex()
            self.kv_block_prefix_indices[target_rank] = index

        for event in batch.events:
            if isinstance(event, BlockStored):
                if int(event.block_size) != self.kv_block_prefix_block_size:
                    logger.warning_once(
                        "KV block-prefix routing observed block_size=%d, expected %d; "
                        "disabling exact KV block-prefix routing.",
                        int(event.block_size),
                        self.kv_block_prefix_block_size,
                    )
                    self.kv_block_prefix_enabled = False
                    self.kv_block_prefix_indices.clear()
                    return
                index.store_blocks(
                    block_hashes=event.block_hashes,
                    token_ids=event.token_ids,
                    block_size=int(event.block_size),
                    parent_block_hash=event.parent_block_hash,
                    extra_keys=event.extra_keys,
                )
            elif isinstance(event, BlockRemoved):
                index.remove_blocks(event.block_hashes)
            elif isinstance(event, AllBlocksCleared):
                index.clear()

    def _handle_kv_prefix_coordinator_update(self, decoded: Sequence[Any]) -> None:
        if not self.kv_block_prefix_enabled:
            return

        _, target_rank, payload = decoded
        batch = self.kv_block_prefix_decoder.decode(payload)
        self._apply_kv_prefix_event_batch(int(target_rank), batch)

    def _choose_engine_by_load(self, candidate_indices: list[int] | None = None) -> int:
        current_counts = self.lb_engines
        min_score = sys.maxsize
        chosen_index = 0
        indices = candidate_indices or list(range(len(current_counts)))
        for i in range(len(indices)):
            idx = indices[(self.eng_start_index + i) % len(indices)]
            waiting, running = current_counts[idx]
            score = waiting * 4 + running
            if score < min_score:
                min_score = score
                chosen_index = idx
        current_counts[chosen_index][0] += self.client_count
        return chosen_index

    async def _send_prefix_router_update(
        self,
        target_rank: int,
        epoch: int,
        prompt_token_ids: list[int],
    ) -> None:
        self._ensure_stats_update_task()
        socket = self.resources.stats_update_socket
        if socket is None:
            return
        update = (
            "PREFIX_ROUTER_UPDATE",
            self.client_index,
            target_rank,
            epoch,
            prompt_token_ids,
        )
        async with self.expert_affinity_send_lock:
            await asyncio.wait_for(
                socket.send(msgspec.msgpack.encode(update)),
                timeout=PREFIX_ROUTER_LEARNING_TIMEOUT_S,
            )

    async def _send_prefix_router_update_batch(
        self,
        updates: list[PrefixRouterUpdate],
    ) -> None:
        self._ensure_stats_update_task()
        socket = self.resources.stats_update_socket
        if socket is None or not updates:
            return

        encoded_updates = [
            (update.target_rank, update.epoch, update.prompt_token_ids)
            for update in updates
            if update.prompt_token_ids
        ]
        if not encoded_updates:
            return

        payload = (
            "PREFIX_ROUTER_UPDATE_BATCH",
            self.client_index,
            encoded_updates,
        )
        async with self.expert_affinity_send_lock:
            await asyncio.wait_for(
                socket.send(msgspec.msgpack.encode(payload)),
                timeout=PREFIX_ROUTER_LEARNING_TIMEOUT_S,
            )

    async def _apply_prefix_router_owner_update_batch(
        self,
        updates: list[tuple[str, list[int], int, int]],
    ) -> None:
        if not self.expert_affinity_learning_enabled or not updates:
            return

        pending_updates: list[PrefixRouterUpdate] = []
        num_engines = len(self.core_engines)
        for request_id, prompt_token_ids, target_rank, epoch in updates:
            if not prompt_token_ids:
                continue
            if target_rank < 0 or target_rank >= num_engines or epoch < 0:
                continue
            if self.load_balancer_debug:
                logger.info(
                    "Prefix affinity update request_id=%s target_rank=%d epoch=%d "
                    "prompt_tokens=%d unique_pairs=None",
                    request_id,
                    target_rank,
                    epoch,
                    len(prompt_token_ids),
                )
            self._apply_prefix_router_update(target_rank, epoch, prompt_token_ids)
            if self.load_balancer_debug:
                logger.warning(
                    "Prefix affinity learning request_id=%s step=upsert:skipped target_rank=%d",
                    request_id,
                    target_rank,
                )
            pending_updates.append(
                PrefixRouterUpdate(
                    target_rank=target_rank,
                    epoch=epoch,
                    prompt_token_ids=prompt_token_ids,
                )
            )

        if pending_updates:
            await self._send_prefix_router_update_batch(pending_updates)

    async def _learn_prefix_router_update(
        self,
        request_id: str,
        engine: EngineIdentity,
        prompt_token_ids: list[int] | None,
        layer_expert_pairs: Any,
        owner: dict[str, int] | None = None,
    ) -> PrefixRouterUpdate | None:
        if not self.expert_affinity_learning_enabled:
            return None

        if not prompt_token_ids:
            return None

        unique_pairs_count: int | None = None

        engine_index = self._engine_index_for_identity(engine)
        if self.load_balancer_debug:
            logger.warning(
                "Prefix affinity learning start request_id=%s engine_index=%d "
                "prompt_tokens=%d",
                request_id,
                engine_index,
                len(prompt_token_ids),
            )

        if owner is None:
            if layer_expert_pairs is None:
                return None
            if self.load_balancer_debug:
                logger.warning(
                    "Prefix affinity learning request_id=%s step=compute_owner:start",
                    request_id,
                )
            owner_cache = self.prefix_router_owner_cache
            owner_epoch = self.prefix_router_owner_cache_epoch
            if owner_cache is None or owner_epoch is None:
                if self.load_balancer_debug:
                    logger.warning(
                        "Prefix affinity learning request_id=%s step=compute_owner:skipped "
                        "reason=missing_owner_cache",
                        request_id,
                    )
                return None
            layer_expert_pairs_list = (
                layer_expert_pairs.tolist()
                if hasattr(layer_expert_pairs, "tolist")
                else layer_expert_pairs
            )
            unique_pairs_count = len(layer_expert_pairs_list)
            owner = compute_owner_from_layer_expert_pairs(
                layer_expert_pairs=layer_expert_pairs_list,
                owner_cache=owner_cache,
                num_ranks=len(self.core_engines),
                epoch=owner_epoch,
            )
            if self.load_balancer_debug:
                logger.warning(
                    "Prefix affinity learning request_id=%s step=compute_owner:done owner=%s",
                    request_id,
                    owner,
                )
        if not owner:
            return None

        target_rank = int(owner["target_rank"])
        epoch = int(owner["epoch"])
        if target_rank < 0 or target_rank >= len(self.core_engines):
            return None
        if self.load_balancer_debug:
            logger.info(
                "Prefix affinity update request_id=%s target_rank=%d epoch=%d "
                "prompt_tokens=%d unique_pairs=%s",
                request_id,
                target_rank,
                epoch,
                len(prompt_token_ids),
                unique_pairs_count,
            )
        self._apply_prefix_router_update(target_rank, epoch, prompt_token_ids)
        # Routing consults the frontend/coordinator radix-tree mirrors. The
        # worker-local prefix tree is not used on the routing path, so avoid a
        # per-request utility RPC here.
        if self.load_balancer_debug:
            logger.warning(
                "Prefix affinity learning request_id=%s step=upsert:skipped target_rank=%d",
                request_id,
                target_rank,
            )
        return PrefixRouterUpdate(
            target_rank=target_rank,
            epoch=epoch,
            prompt_token_ids=list(prompt_token_ids),
        )

    def _get_expert_affinity_scores(
        self,
        prompt_token_ids: list[int] | None,
    ) -> dict[int, float]:
        if not self.expert_affinity_enabled or not prompt_token_ids:
            return {}

        total_tokens = len(prompt_token_ids)
        if total_tokens <= 0:
            return {}

        scores: dict[int, float] = {}
        for rank, tree in self.expert_affinity_trees.items():
            prefix_len = tree.longest_prefix_length(prompt_token_ids)
            if prefix_len <= 0:
                continue
            engine_index = self.rank_to_engine_index.get(rank)
            if engine_index is None:
                continue
            scores[engine_index] = prefix_len / total_tokens
        return scores

    def _get_kv_block_prefix_scores(
        self,
        request: EngineCoreRequest,
    ) -> dict[int, float]:
        if not self.kv_block_prefix_enabled:
            return {}

        query = prepare_block_prefix_query(
            token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            block_size=self.kv_block_prefix_block_size,
        )
        if query is None:
            return {}

        scores: dict[int, float] = {}
        for rank, index in self.kv_block_prefix_indices.items():
            matched_blocks = index.longest_prefix_blocks_for_query(query)
            if matched_blocks <= 0:
                continue
            engine_index = self.rank_to_engine_index.get(rank)
            if engine_index is None:
                continue
            scores[engine_index] = matched_blocks / query.total_blocks
        return scores

    def _get_load_scores(self) -> dict[int, float]:
        if not self.load_score_enabled:
            return {}

        current_counts = self.lb_engines
        if not current_counts:
            return {}

        raw_scores = [
            (waiting * 4) + running for waiting, running in current_counts
        ]
        min_raw = min(raw_scores)
        max_raw = max(raw_scores)
        if max_raw == min_raw:
            return {idx: 1.0 for idx in range(len(raw_scores))}

        span = float(max_raw - min_raw)
        return {
            idx: (max_raw - raw_score) / span
            for idx, raw_score in enumerate(raw_scores)
        }

    # ///////////// Expert-based load balancing
    @staticmethod
    def _format_score_map(scores: dict[int, float]) -> str:
        if not scores:
            return "{}"
        items = ", ".join(
            f"{idx}:{score:.4f}" for idx, score in sorted(scores.items())
        )
        return "{" + items + "}"
    # ///////////// Expert-based load balancing

    def _build_coordinator_route_request(
        self,
        call_id: int,
        request: EngineCoreRequest,
    ) -> CoordinatorRouteRequest:
        kv_total_blocks = 0
        kv_extra_keys = None
        kv_use_zero_tokens = False

        if self.kv_block_prefix_weight > 0.0:
            query = prepare_block_prefix_query(
                token_ids=request.prompt_token_ids,
                prompt_embeds=request.prompt_embeds,
                mm_features=request.mm_features,
                lora_request=request.lora_request,
                cache_salt=request.cache_salt,
                block_size=self.kv_block_prefix_block_size,
            )
            if query is not None:
                kv_total_blocks = query.total_blocks
                kv_extra_keys = build_query_block_extra_keys(query)
                kv_use_zero_tokens = query.zero_block is not None

        return CoordinatorRouteRequest(
            call_id=call_id,
            request_id=request.request_id,
            client_index=self.client_index,
            client_count=self.client_count,
            prompt_token_ids=request.prompt_token_ids,
            kv_total_blocks=kv_total_blocks,
            kv_extra_keys=kv_extra_keys,
            kv_use_zero_tokens=kv_use_zero_tokens,
        )

    async def _get_core_engine_for_request_async(
        self,
        request: EngineCoreRequest,
    ) -> EngineIdentity:
        chosen_engine = await self._select_core_engine_for_request_async(request)
        pending_engine_indices = self._pending_engine_indices_for_engine(chosen_engine)
        self.reqs_in_flight[request.request_id] = InFlightRequestInfo(
            request_id=request.request_id,
            engine=chosen_engine,
            prompt_token_ids=request.prompt_token_ids,
            pending_engine_indices=pending_engine_indices,
        )
        if (
            self.expert_affinity_learning_enabled
            and request.prompt_token_ids is not None
        ):
            self.prefix_learning_context_by_request[request.request_id] = (
                PrefixLearningContext(
                    engine=chosen_engine,
                    prompt_token_ids=list(request.prompt_token_ids),
                )
            )
        return chosen_engine

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        chosen_engine = self._select_core_engine_for_request(request)
        pending_engine_indices = self._pending_engine_indices_for_engine(chosen_engine)
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = InFlightRequestInfo(
            request_id=request.request_id,
            engine=chosen_engine,
            prompt_token_ids=request.prompt_token_ids,
            pending_engine_indices=pending_engine_indices,
        )
        if (
            self.expert_affinity_learning_enabled
            and request.prompt_token_ids is not None
        ):
            self.prefix_learning_context_by_request[request.request_id] = (
                PrefixLearningContext(
                    engine=chosen_engine,
                    prompt_token_ids=list(request.prompt_token_ids),
                )
            )
        return chosen_engine

    async def _dispatch_request_to_engine(
        self,
        request: EngineCoreRequest,
        chosen_engine: EngineIdentity,
        pending_engine_indices: Sequence[int] = (),
    ) -> None:
        request.current_wave = self.current_wave
        request.client_index = self.client_index
        self.reqs_in_flight[request.request_id] = InFlightRequestInfo(
            request_id=request.request_id,
            engine=chosen_engine,
            prompt_token_ids=request.prompt_token_ids,
            pending_engine_indices=tuple(pending_engine_indices),
        )
        if (
            self.expert_affinity_learning_enabled
            and request.prompt_token_ids is not None
        ):
            self.prefix_learning_context_by_request[request.request_id] = (
                PrefixLearningContext(
                    engine=chosen_engine,
                    prompt_token_ids=list(request.prompt_token_ids),
                )
            )
        if self._should_broadcast_ep_request():
            send_ops: list[Awaitable[Any]] = []
            created_shadow_request_ids: list[str] = []
            for eng_idx, engine in enumerate(self.core_engines):
                if engine == chosen_engine:
                    send_ops.append(
                        self._send_input(EngineCoreRequestType.ADD, request, engine)
                    )
                    continue

                shadow_request = copy.copy(request)
                shadow_request.request_id = self._make_shadow_request_id(
                    request.request_id, eng_idx
                )
                if shadow_request.sampling_params is not None:
                    shadow_request.sampling_params = (
                        shadow_request.sampling_params.clone()
                    )
                    shadow_request.sampling_params.skip_reading_prefix_cache = True
                    shadow_request.sampling_params.skip_writing_prefix_cache = True
                if shadow_request.pooling_params is not None:
                    shadow_request.pooling_params = (
                        shadow_request.pooling_params.clone()
                    )
                    shadow_request.pooling_params.skip_reading_prefix_cache = True
                    shadow_request.pooling_params.skip_writing_prefix_cache = True
                created_shadow_request_ids.append(shadow_request.request_id)
                self._register_shadow_request(
                    request.request_id,
                    shadow_request.request_id,
                    engine,
                    request.prompt_token_ids,
                )
                send_ops.append(
                    self._send_input(
                        EngineCoreRequestType.ADD, shadow_request, engine
                    )
                )
            to_await = asyncio.gather(*send_ops)
        else:
            created_shadow_request_ids = []
            to_await = self._send_input(
                EngineCoreRequestType.ADD, request, chosen_engine
            )

        if not self.engines_running:
            wake_exclude = None if self._should_broadcast_ep_request() else chosen_engine
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", wake_exclude))
            await self.first_req_send_socket.send(req_msg)

        try:
            await to_await
        except Exception:
            self.reqs_in_flight.pop(request.request_id, None)
            self.prefix_learning_context_by_request.pop(request.request_id, None)
            self.primary_req_to_shadows.pop(request.request_id, None)
            for shadow_request_id in created_shadow_request_ids:
                self.shadow_req_to_primary.pop(shadow_request_id, None)
                self.reqs_in_flight.pop(shadow_request_id, None)
            raise

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()
        if self.coordinator_routing_enabled:
            self._ensure_route_reply_task()

        if not self.frontend_dispatch_queue_enabled:
            chosen_engine = await self._select_core_engine_for_request_async(request)
            pending_engine_indices = self._pending_engine_indices_for_engine(
                chosen_engine
            )
            await self._dispatch_request_to_engine(
                request,
                chosen_engine,
                pending_engine_indices,
            )
            self._ensure_output_queue_task()
            return

        loop = asyncio.get_running_loop()
        queued_request = QueuedDispatchRequest(
            request=request,
            dispatched=loop.create_future(),
        )
        async with self.dispatch_queue_lock:
            self.queued_dispatch_requests.append(queued_request)
            self.queued_dispatch_by_request_id[request.request_id] = queued_request
        self._ensure_dispatch_queue_task()
        await queued_request.dispatched
        if queued_request.cancelled:
            return

        self._ensure_output_queue_task()

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        primary_ids_to_abort: set[str] = set()
        direct_prefix_owner_updates: list[tuple[str, list[int], int, int]] = []

        if outputs.prefix_router_placement_update is not None:
            update = outputs.prefix_router_placement_update
            epoch = update.get("epoch")
            physical_to_logical_map = update.get("physical_to_logical_map")
            if isinstance(epoch, int) and physical_to_logical_map is not None:
                self._apply_prefix_router_placement_update(
                    epoch,
                    list(physical_to_logical_map),
                )

        if (
            self.expert_affinity_learning_enabled
            and outputs.prefix_learning_owner_updates
        ):
            for request_id, target_rank, epoch in outputs.prefix_learning_owner_updates:
                request_info = self.reqs_in_flight.get(request_id)
                context = self._take_prefix_learning_context(request_id, request_info)
                if context is None or not context.prompt_token_ids:
                    continue
                if request_info is not None:
                    request_info.expert_affinity_prefill_learned = True
                direct_prefix_owner_updates.append(
                    (
                        request_id,
                        list(context.prompt_token_ids),
                        int(target_rank),
                        int(epoch),
                    )
                )

        if outputs.outputs:
            filtered_outputs = []
            for output in outputs.outputs:
                request_info = self.reqs_in_flight.get(output.request_id)
                if request_info is not None and request_info.is_shadow:
                    if request_info.primary_request_id is not None:
                        primary_ids_to_abort.add(request_info.primary_request_id)
                    continue
                filtered_outputs.append(output)
            outputs.outputs = filtered_outputs

        if outputs.outputs:
            for output in outputs.outputs:
                await self._release_pending_slots_for_request(output.request_id)

        if self.expert_affinity_learning_enabled and outputs.outputs:
            for output in outputs.outputs:
                request_info = self.reqs_in_flight.get(output.request_id)
                if request_info is None:
                    continue
                if self.prefix_affinity_only_prefill:
                    if request_info.expert_affinity_prefill_learned:
                        continue
                    if (
                        output.prefix_learning_owner is None
                        and output.prefix_learning_pairs is None
                    ):
                        continue
                    if self.load_balancer_debug and output.prefix_learning_pairs is not None:
                        logger.warning(
                            "Prefix learning pairs captured request_id=%s phase=prefill "
                            "unique_pairs=%d",
                            output.request_id,
                            len(output.prefix_learning_pairs),
                        )
                    request_info.expert_affinity_prefill_learned = True
                else:
                    if output.finish_reason is None:
                        continue
                    if (
                        output.prefix_learning_owner is None
                        and output.prefix_learning_pairs is None
                    ):
                        continue
                    if self.load_balancer_debug and output.prefix_learning_pairs is not None:
                        logger.warning(
                            "Prefix learning pairs captured request_id=%s phase=finish "
                            "unique_pairs=%d",
                            output.request_id,
                            len(output.prefix_learning_pairs),
                        )
                context = self._take_prefix_learning_context(
                    output.request_id,
                    request_info,
                )
                if context is None or not context.prompt_token_ids:
                    continue
                if output.prefix_learning_owner is not None:
                    direct_prefix_owner_updates.append(
                        (
                            output.request_id,
                            list(context.prompt_token_ids),
                            int(output.prefix_learning_owner["target_rank"]),
                            int(output.prefix_learning_owner["epoch"]),
                        )
                    )
                    continue
                await self._enqueue_prefix_learning_work_item(
                    PrefixLearningWorkItem(
                        request_id=output.request_id,
                        engine=context.engine,
                        prompt_token_ids=list(context.prompt_token_ids),
                        layer_expert_pairs=output.prefix_learning_pairs,
                        owner=None,
                    )
                )

        if direct_prefix_owner_updates:
            await self._apply_prefix_router_owner_update_batch(
                direct_prefix_owner_updates
            )
    # ///////////// Expert-based load balancing

        if outputs.finished_requests and self.reqs_in_flight:
            filtered_finished_requests = set[str]()
            for req_id in outputs.finished_requests:
                request_info = self.reqs_in_flight.get(req_id)
                if request_info is not None and request_info.is_shadow:
                    if request_info.primary_request_id is not None:
                        primary_ids_to_abort.add(request_info.primary_request_id)
                    self._cleanup_shadow_request(req_id)
                    continue
                await self._release_pending_slots_for_request(req_id)
                filtered_finished_requests.add(req_id)
                self.reqs_in_flight.pop(req_id, None)
            outputs.finished_requests = filtered_finished_requests or None

        if outputs.outputs:
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    primary_ids_to_abort.add(output.request_id)

        for primary_request_id in primary_ids_to_abort:
            await self._abort_shadow_requests(primary_request_id)

    @staticmethod
    async def eep_process_engine_core_notification(
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        cache = self.eep_scaling_cache
        notification_type_str, dp_rank = notification_data
        try:
            notification_type = EEPNotificationType(notification_type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            from vllm.v1.engine import UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            _process_utility_output(dummy_output, self.utility_results)
            return
        assert cache is not None
        if notification_type not in cache.pending_notifications:
            cache.pending_notifications[notification_type] = set()
        if dp_rank in cache.pending_notifications[notification_type]:
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
        cache.pending_notifications[notification_type].add(dp_rank)
        if len(cache.pending_notifications[notification_type]) >= abs(
            cache.num_new_core_engines
        ):
            if notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:
                assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
                assert cache.num_new_core_engines < 0
                old_dp_size = len(cache.existing_core_engines)
                new_dp_size = old_dp_size + cache.num_new_core_engines
                self.resources.engine_manager.scale_down_elastic_ep(
                    old_dp_size, new_dp_size
                )
            else:
                await asyncio.gather(
                    *[
                        self._call_utility_async(
                            "eep_handle_engine_core_notification",
                            notification_type,
                            engine=engine,
                        )
                        for engine in cache.existing_core_engines
                    ]
                )
            cache.pending_notifications[notification_type] = set()
            if notification_type in [
                EEPNotificationType.SHUTDOWN_COMPLETE,
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
            ]:
                self.eep_scaling_cache = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return

        remaining_request_ids = request_ids
        if self.frontend_dispatch_queue_enabled:
            cancelled_request_ids = set[str]()
            async with self.dispatch_queue_lock:
                for req_id in request_ids:
                    queued_request = self.queued_dispatch_by_request_id.pop(req_id, None)
                    if queued_request is None:
                        continue
                    queued_request.cancelled = True
                    cancelled_request_ids.add(req_id)
                    if not queued_request.dispatched.done():
                        queued_request.dispatched.set_result(None)

            if cancelled_request_ids:
                self._ensure_dispatch_queue_task()
                remaining_request_ids = [
                    req_id
                    for req_id in request_ids
                    if req_id not in cancelled_request_ids
                ]
                if not remaining_request_ids:
                    return

        if len(remaining_request_ids) == 1:
            # Fast-path common case.
            request_id = remaining_request_ids[0]
            if request_info := self.reqs_in_flight.get(request_id):
                primary_request_id = (
                    request_info.primary_request_id
                    if request_info.is_shadow
                    else request_id
                )
                self.prefix_learning_context_by_request.pop(primary_request_id, None)
                await self._abort_requests([request_id], request_info.engine)
                await self._abort_shadow_requests(primary_request_id)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        primary_request_ids = set[str]()
        for req_id in remaining_request_ids:
            if request_info := self.reqs_in_flight.get(req_id):
                by_engine[request_info.engine].append(req_id)
                primary_request_id = (
                    request_info.primary_request_id
                    if request_info.is_shadow and request_info.primary_request_id
                    else req_id
                )
                primary_request_ids.add(primary_request_id)
                self.prefix_learning_context_by_request.pop(primary_request_id, None)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)
        for primary_request_id in primary_request_ids:
            await self._abort_shadow_requests(primary_request_id)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale elastic EP data parallel size"""
        cur_data_parallel_size = len(self.core_engines)

        assert new_data_parallel_size != cur_data_parallel_size, (
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )

        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size

        if scale_up:
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
        else:
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )

    async def _eep_wait_for_setup_switch_complete(self) -> None:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        future = asyncio.get_running_loop().create_future()
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future
        self._ensure_output_queue_task()
        await future

    def _setup_elastic_ep_reconfig_bootstrap(self) -> tuple[str, int]:
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self.vllm_config.parallel_config
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )

        ip = parallel_config.data_parallel_master_ip
        store = create_tcp_store(
            ip,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = store.port
        self._coord_store = store
        return ip, store.port

    async def _scale_up_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.vllm_config.parallel_config
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()

        # Phase 1: Send reconfig messages to existing engines
        reconfig_futures = []
        for engine in self.core_engines:
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                coord_store_port=coord_store_port,
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # Phase 2: Create new engines
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        parallel_config.eplb_config.num_redundant_experts = 0
        start_new_worker_future = asyncio.to_thread(
            self.resources.engine_manager.scale_up_elastic_ep,
            self.vllm_config,
            new_data_parallel_size,
        )
        wait_future = self._eep_wait_for_setup_switch_complete()

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        await asyncio.gather(start_new_worker_future, *reconfig_futures)
        logger.info("[Elastic EP] Successfully started new engines")

        # Create new CoreEngine objects for the new engines
        new_engine_identities = set()
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            new_engine = i.to_bytes(2, "little")
            self.core_engines.append(new_engine)
            # NOTE(yongji): we don't update lb_engines here,
            # we let run_engine_stats_update_task to update it.
            new_engine_identities.add(new_engine)

        # Wait for ready messages from new engines on the input socket
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(
                timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
            ):
                raise TimeoutError(
                    f"Timed out waiting for new engine core processes to "
                    f"start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            identity, _ = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)

        # NOTE(yongji): Before we schedule any requests on the new workers,
        # we should wait for them to switch to the new setup.
        await wait_future
        # Update the parallel config
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )

    async def _scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.vllm_config.parallel_config
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()

        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(self.core_engines):
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                coord_store_port=coord_store_port,
            )
            if cur_dp_rank >= new_data_parallel_size:
                reconfig_request.new_data_parallel_rank = (
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        self.core_engines = self.core_engines[:new_data_parallel_size]
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        wait_future = self._eep_wait_for_setup_switch_complete()

        await asyncio.gather(*reconfig_futures)

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        self._ensure_stats_update_task()
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_down_marker)

        # NOTE(yongji): Unlike scaling up,
        # here we don't actually need to wait for the setup switch to complete.
        # We may want to remove it in the future.
        await wait_future
        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
