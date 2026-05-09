# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import sys
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.genome.engine.load_balancing_types import (
    PREFIX_ROUTER_LEARNING_TIMEOUT_S,
    InFlightRequestInfo,
    PrefixLearningContext,
    PrefixLearningWorkItem,
    PrefixRouterUpdate,
    QueuedDispatchRequest,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import (
    CoordinatorRouteRequest,
    CoordinatorRouteResponse,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
from vllm.v1.prefix_router import (
    ExactBlockPrefixIndex,
    ExpertAffinityIndex,
    build_owner_cache_from_physical_to_logical_map,
    build_query_block_extra_keys,
    compute_owner_from_layer_expert_pairs,
    make_expert_affinity_index,
    prepare_block_prefix_query,
)
from vllm.v1.serial_utils import MsgpackDecoder

logger = init_logger(__name__)


class GenomeDPLBRoutingMixin:

    def _init_genome_routing_state(self, *, client_count: int) -> None:
        assert len(self.core_engines) > 1

        self.eng_start_index = (len(self.core_engines) * self.client_index) // client_count
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
            model_config.load_score_routing_weight if self.load_score_enabled else 0.0
        )
        self.prefix_learning_algorithm = model_config.prefix_learning_algorithm
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
                "prefix_learning_algorithm=%s "
                "load_score_enabled=%s coordinator_routing_enabled=%s "
                "prefix_affinity_learning_queue_size=%d "
                "frontend_dispatch_queue_enabled=%s max_pending_requests_per_engine=%d "
                "route_query_address=%s",
                self.expert_affinity_enabled,
                self.kv_block_prefix_enabled,
                self.expert_affinity_learning_enabled,
                self.prefix_learning_algorithm,
                self.load_score_enabled,
                self.coordinator_routing_enabled,
                self.prefix_affinity_learning_queue_size,
                self.frontend_dispatch_queue_enabled,
                self.max_pending_requests_per_engine,
                self.route_query_address,
            )
        self.expert_affinity_epoch = 0
        self.expert_affinity_indices: dict[int, ExpertAffinityIndex] = (
            {}
            if self.coordinator_routing_enabled
            else {
                rank: self._make_expert_affinity_index()
                for rank in self.engine_ranks_managed
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

    async def _dispatch_request_to_engine(
        self,
        request: EngineCoreRequest,
        chosen_engine: Any,
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
        to_await = self._send_input(
            EngineCoreRequestType.ADD, request, chosen_engine
        )

        if not self.engines_running:
            wake_exclude = self._engine_index_for_identity(chosen_engine)
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", wake_exclude))
            await self.first_req_send_socket.send(req_msg)

        try:
            await to_await
        except Exception:
            self.reqs_in_flight.pop(request.request_id, None)
            self.prefix_learning_context_by_request.pop(request.request_id, None)
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

    @staticmethod
    async def process_engine_outputs(
        self: Any, outputs: EngineCoreOutputs
    ) -> None:
        direct_prefix_owner_updates: list[PrefixRouterUpdate] = []

        if outputs.kv_cache_event_batch is not None:
            self._apply_kv_prefix_event_batch(
                outputs.engine_index, outputs.kv_cache_event_batch
            )
            await self._send_kv_prefix_event_batch(
                outputs.engine_index, outputs.kv_cache_event_batch
            )

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
                    PrefixRouterUpdate(
                        target_rank=int(target_rank),
                        epoch=int(epoch),
                        prompt_token_ids=list(context.prompt_token_ids),
                    )
                )

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
                        PrefixRouterUpdate(
                            target_rank=int(output.prefix_learning_owner["target_rank"]),
                            epoch=int(output.prefix_learning_owner["epoch"]),
                            prompt_token_ids=list(context.prompt_token_ids),
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

        if outputs.finished_requests and self.reqs_in_flight:
            filtered_finished_requests = set[str]()
            for req_id in outputs.finished_requests:
                await self._release_pending_slots_for_request(req_id)
                filtered_finished_requests.add(req_id)
                self.reqs_in_flight.pop(req_id, None)
            outputs.finished_requests = filtered_finished_requests or None

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
            request_id = remaining_request_ids[0]
            if request_info := self.reqs_in_flight.get(request_id):
                self.prefix_learning_context_by_request.pop(request_id, None)
                await self._abort_requests([request_id], request_info.engine)
            return

        by_engine = defaultdict[Any, list[str]](list)
        for req_id in remaining_request_ids:
            if request_info := self.reqs_in_flight.get(req_id):
                by_engine[request_info.engine].append(req_id)
                self.prefix_learning_context_by_request.pop(req_id, None)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: Any
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    def _make_expert_affinity_index(self) -> ExpertAffinityIndex:
        return make_expert_affinity_index(self.prefix_learning_algorithm)

    def _get_runtime_load_balancer_weights(self) -> dict[str, Any]:
        return {
            "expert_affinity_routing_weight": self.expert_affinity_weight,
            "kv_block_prefix_routing_weight": self.kv_block_prefix_weight,
            "load_score_routing_weight": self.load_score_weight,
            "expert_affinity_enabled": self.expert_affinity_enabled,
            "kv_block_prefix_enabled": self.kv_block_prefix_enabled,
            "load_score_enabled": self.load_score_enabled,
            "coordinator_routing_enabled": self.coordinator_routing_enabled,
            "prefix_learning_algorithm": self.prefix_learning_algorithm,
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
                raise ValueError("expert_affinity_routing_weight must be non-negative")
            if (
                expert_affinity_routing_weight > 0.0
                and not self.expert_affinity_enabled
            ):
                raise ValueError("Prefix-affinity routing was not enabled at startup")
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
            if kv_block_prefix_routing_weight > 0.0 and not self.kv_block_prefix_enabled:
                raise ValueError("KV block-prefix routing was not enabled at startup")
            self.kv_block_prefix_weight = (
                float(kv_block_prefix_routing_weight)
                if self.kv_block_prefix_enabled
                else 0.0
            )

        if load_score_routing_weight is not None:
            if load_score_routing_weight < 0.0:
                raise ValueError("load_score_routing_weight must be non-negative")
            if load_score_routing_weight > 0.0 and not self.load_score_enabled:
                raise ValueError("Load-score routing was not enabled at startup")
            self.load_score_weight = (
                float(load_score_routing_weight) if self.load_score_enabled else 0.0
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

    def _engine_index_for_identity(self, engine: bytes) -> int:
        for idx, candidate in enumerate(self.core_engines):
            if candidate == engine:
                return idx
        raise ValueError("Unknown engine identity")

    def _pending_engine_indices_for_engine(self, engine: bytes) -> tuple[int, ...]:
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
            if (
                len(self.prefix_learning_queue)
                >= self.prefix_affinity_learning_queue_size
            ):
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
                await self._apply_prefix_router_owner_update_batch(pending_updates)
            except asyncio.TimeoutError:
                request_ids = ",".join(item.request_id for item in items[:4])
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
    ) -> bytes:
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
    ) -> bytes:
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
            matched_indices, selection_trace = (
                self._select_candidates_by_routing_precedence(
                    expert_scores,
                    kv_scores,
                    load_scores,
                )
            )

            if matched_indices is not None:
                eng_index = self._choose_engine_by_load(matched_indices)
                if self.load_balancer_debug:
                    logger.warning(
                        "LB route request_id=%s expert_scores=%s kv_scores=%s "
                        "load_scores=%s precedence=%s chosen_engine=%d",
                        request.request_id,
                        self._format_score_map(expert_scores),
                        self._format_score_map(kv_scores),
                        self._format_score_map(load_scores),
                        selection_trace,
                        eng_index,
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
            unsupported_reasons.append("internal DP load balancing must manage all ranks")

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
            unsupported_reasons.append("internal DP load balancing must manage all ranks")
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
        for index in self.expert_affinity_indices.values():
            index.clear()
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
        if target_rank not in self.expert_affinity_indices:
            self.expert_affinity_indices[target_rank] = (
                self._make_expert_affinity_index()
            )
        self.expert_affinity_indices[target_rank].insert(prompt_token_ids)

    def _handle_prefix_router_coordinator_update(self, decoded: Sequence[Any]) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, source_client_index, target_rank, epoch, prompt_token_ids = decoded
        if int(source_client_index) == self.client_index:
            return
        self._apply_prefix_router_update(int(target_rank), int(epoch), list(prompt_token_ids))

    def _handle_prefix_router_coordinator_update_batch(self, decoded: Sequence[Any]) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, source_client_index, updates = decoded
        if int(source_client_index) == self.client_index:
            return

        for update in updates:
            if len(update) != 3:
                continue
            target_rank, epoch, prompt_token_ids = update
            self._apply_prefix_router_update(int(target_rank), int(epoch), list(prompt_token_ids))

    def _handle_prefix_router_placement_coordinator_update(self, decoded: Sequence[Any]) -> None:
        if not self.expert_affinity_learning_enabled:
            return

        _, epoch, physical_to_logical_map = decoded
        self._apply_prefix_router_placement_update(int(epoch), list(physical_to_logical_map))

    def _apply_kv_prefix_event_batch(self, target_rank: int, batch: KVEventBatch) -> None:
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
        update = ("PREFIX_ROUTER_UPDATE", self.client_index, target_rank, epoch, prompt_token_ids)
        async with self.expert_affinity_send_lock:
            await asyncio.wait_for(
                socket.send(msgspec.msgpack.encode(update)),
                timeout=PREFIX_ROUTER_LEARNING_TIMEOUT_S,
            )

    async def _send_prefix_router_update_batch(self, updates: list[PrefixRouterUpdate]) -> None:
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

        payload = ("PREFIX_ROUTER_UPDATE_BATCH", self.client_index, encoded_updates)
        async with self.expert_affinity_send_lock:
            await asyncio.wait_for(
                socket.send(msgspec.msgpack.encode(payload)),
                timeout=PREFIX_ROUTER_LEARNING_TIMEOUT_S,
            )

    async def _send_kv_prefix_event_batch(self, target_rank: int, batch: KVEventBatch) -> None:
        self._ensure_stats_update_task()
        socket = self.resources.stats_update_socket
        if socket is None or not batch.events:
            return

        payload = ("KV_PREFIX_EVENT_BATCH", target_rank, msgspec.msgpack.encode(batch))
        async with self.expert_affinity_send_lock:
            await asyncio.wait_for(
                socket.send(msgspec.msgpack.encode(payload)),
                timeout=PREFIX_ROUTER_LEARNING_TIMEOUT_S,
            )

    async def _apply_prefix_router_owner_update_batch(
        self,
        updates: list[PrefixRouterUpdate],
    ) -> None:
        if not self.expert_affinity_learning_enabled or not updates:
            return

        pending_updates: list[PrefixRouterUpdate] = []
        num_engines = len(self.core_engines)
        for update in updates:
            prompt_token_ids = update.prompt_token_ids
            target_rank = update.target_rank
            epoch = update.epoch
            if not prompt_token_ids:
                continue
            if target_rank < 0 or target_rank >= num_engines or epoch < 0:
                continue
            if self.load_balancer_debug:
                logger.info(
                    "Prefix affinity update request_id=%s target_rank=%d epoch=%d "
                    "prompt_tokens=%d unique_pairs=None",
                    "",
                    target_rank,
                    epoch,
                    len(prompt_token_ids),
                )
            self._apply_prefix_router_update(target_rank, epoch, prompt_token_ids)
            if self.load_balancer_debug:
                logger.warning(
                    "Prefix affinity learning request_id=%s step=upsert:skipped target_rank=%d",
                    "",
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
        engine: bytes,
        prompt_token_ids: list[int] | None,
        layer_expert_pairs: Any,
        owner: dict[str, int] | None = None,
    ) -> PrefixRouterUpdate | None:
        if not self.expert_affinity_learning_enabled or not prompt_token_ids:
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
        return PrefixRouterUpdate(
            target_rank=target_rank,
            epoch=epoch,
            prompt_token_ids=list(prompt_token_ids),
        )

    def _get_expert_affinity_scores(self, prompt_token_ids: list[int] | None) -> dict[int, float]:
        if not self.expert_affinity_enabled or not prompt_token_ids:
            return {}

        scores: dict[int, float] = {}
        for rank, index in self.expert_affinity_indices.items():
            score = index.score(prompt_token_ids)
            if score <= 0.0:
                continue
            engine_index = self.rank_to_engine_index.get(rank)
            if engine_index is None:
                continue
            scores[engine_index] = score
        return self._normalize_score_map(scores)

    def _get_kv_block_prefix_scores(self, request: EngineCoreRequest) -> dict[int, float]:
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
        return self._normalize_score_map(scores)

    def _get_load_scores(self) -> dict[int, float]:
        if not self.load_score_enabled:
            return {}

        current_counts = self.lb_engines
        if not current_counts:
            return {}

        raw_scores = [(waiting * 4) + running for waiting, running in current_counts]
        min_raw = min(raw_scores)
        max_raw = max(raw_scores)
        if max_raw == min_raw:
            return self._normalize_score_map(
                {idx: 1.0 for idx in range(len(raw_scores))}
            )

        span = float(max_raw - min_raw)
        return self._normalize_score_map({
            idx: (max_raw - raw_score) / span
            for idx, raw_score in enumerate(raw_scores)
        })

    @staticmethod
    def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}

        total = sum(score for score in scores.values() if score > 0.0)
        if total <= 0.0:
            return {}

        return {
            idx: score / total
            for idx, score in scores.items()
            if score > 0.0
        }

    def _get_routing_score_precedence(
        self,
        expert_scores: dict[int, float],
        kv_scores: dict[int, float],
        load_scores: dict[int, float],
    ) -> list[tuple[str, float, dict[int, float]]]:
        precedence: list[tuple[str, float, dict[int, float]]] = []
        if self.expert_affinity_weight > 0.0:
            precedence.append(("expert_affinity", self.expert_affinity_weight, expert_scores))
        if self.kv_block_prefix_weight > 0.0:
            precedence.append(("kv_block_prefix", self.kv_block_prefix_weight, kv_scores))
        if self.load_score_weight > 0.0:
            precedence.append(("load_score", self.load_score_weight, load_scores))

        tie_break_order = {
            "expert_affinity": 0,
            "kv_block_prefix": 1,
            "load_score": 2,
        }
        precedence.sort(key=lambda item: (-item[1], tie_break_order[item[0]]))
        return precedence

    def _select_candidates_by_routing_precedence(
        self,
        expert_scores: dict[int, float],
        kv_scores: dict[int, float],
        load_scores: dict[int, float],
    ) -> tuple[list[int] | None, list[str]]:
        precedence = self._get_routing_score_precedence(
            expert_scores,
            kv_scores,
            load_scores,
        )
        if not precedence:
            return None, []

        candidate_indices = list(range(len(self.core_engines)))
        selection_trace: list[str] = []
        for name, _, scores in precedence:
            candidate_scores = {idx: scores.get(idx, 0.0) for idx in candidate_indices}
            best_score = max(candidate_scores.values(), default=0.0)
            candidate_indices = [
                idx for idx, score in candidate_scores.items() if score == best_score
            ]
            selection_trace.append(f"{name}:{best_score:.4f}->{candidate_indices}")
            if len(candidate_indices) <= 1:
                break

        return candidate_indices, selection_trace

    @staticmethod
    def _format_score_map(scores: dict[int, float]) -> str:
        if not scores:
            return "{}"
        items = ", ".join(f"{idx}:{score:.4f}" for idx, score in sorted(scores.items()))
        return "{" + items + "}"

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

    async def _get_core_engine_for_request_async(self, request: EngineCoreRequest) -> bytes:
        chosen_engine = await self._select_core_engine_for_request_async(request)
        pending_engine_indices = self._pending_engine_indices_for_engine(chosen_engine)
        self.reqs_in_flight[request.request_id] = InFlightRequestInfo(
            request_id=request.request_id,
            engine=chosen_engine,
            prompt_token_ids=request.prompt_token_ids,
            pending_engine_indices=pending_engine_indices,
        )
        if self.expert_affinity_learning_enabled and request.prompt_token_ids is not None:
            self.prefix_learning_context_by_request[request.request_id] = (
                PrefixLearningContext(
                    engine=chosen_engine,
                    prompt_token_ids=list(request.prompt_token_ids),
                )
            )
        return chosen_engine

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> bytes:
        chosen_engine = self._select_core_engine_for_request(request)
        pending_engine_indices = self._pending_engine_indices_for_engine(chosen_engine)
        self.reqs_in_flight[request.request_id] = InFlightRequestInfo(
            request_id=request.request_id,
            engine=chosen_engine,
            prompt_token_ids=request.prompt_token_ids,
            pending_engine_indices=pending_engine_indices,
        )
        if self.expert_affinity_learning_enabled and request.prompt_token_ids is not None:
            self.prefix_learning_context_by_request[request.request_id] = (
                PrefixLearningContext(
                    engine=chosen_engine,
                    prompt_token_ids=list(request.prompt_token_ids),
                )
            )
        return chosen_engine
