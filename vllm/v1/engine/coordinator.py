# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
# ///////////// Expert-based load balancing
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch
from vllm.logger import init_logger
from vllm.utils.network_utils import get_tcp_uri, make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import (
    CoordinatorRouteRequest,
    CoordinatorRouteResponse,
    EngineCoreOutputs,
    EngineCoreRequestType,
)
from vllm.v1.prefix_router import ExactBlockPrefixIndex, TokenRadixTree
# ///////////// Expert-based load balancing
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str, str]:
        try:
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=30
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    "during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self,
        parallel_config: ParallelConfig,
        enable_wave_coordination: bool = True,
        # ///////////// Expert-based load balancing
        enable_prefix_affinity_routing: bool = False,
        enable_kv_block_prefix_routing: bool = False,
        enable_load_score_routing: bool = False,
        expert_affinity_routing_weight: float = 1.0,
        kv_block_prefix_routing_weight: float = 1.0,
        load_score_routing_weight: float = 1.0,
        load_balancer_debug: bool = False,
        kv_block_prefix_block_size: int = 0,
        # ///////////// Expert-based load balancing
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # NOTE(yongji): handling scaling from intra-node to inter-node
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        def bind_address(local_only: bool) -> str:
            return (
                get_engine_client_zmq_addr(local_only=True, host=host)
                if local_only
                else get_tcp_uri(host, 0)
            )

        front_publish_address = bind_address(local_only)
        # ///////////// Expert-based load balancing
        front_route_address = bind_address(local_only)
        back_publish_address = bind_address(local_only_eng)
        back_output_address = bind_address(local_only_eng)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "front_route_address": front_route_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
                # ///////////// Expert-based load balancing
                "enable_prefix_affinity_routing": enable_prefix_affinity_routing,
                "enable_kv_block_prefix_routing": enable_kv_block_prefix_routing,
                "enable_load_score_routing": enable_load_score_routing,
                "expert_affinity_routing_weight": expert_affinity_routing_weight,
                "kv_block_prefix_routing_weight": kv_block_prefix_routing_weight,
                "load_score_routing_weight": load_score_routing_weight,
                "load_balancer_debug": load_balancer_debug,
                "kv_block_prefix_block_size": kv_block_prefix_block_size,
                # ///////////// Expert-based load balancing
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            front_route_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        # ///////////// Expert-based load balancing
        self.route_query_address = front_route_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    # ///////////// Expert-based load balancing
    def get_route_query_address(self) -> str:
        return self.route_query_address
    # ///////////// Expert-based load balancing

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    def __init__(self):
        self.request_counts = [0, 0]  # [waiting, running]


class DPCoordinatorProc:
    # ///////////// Expert-based load balancing
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        enable_prefix_affinity_routing: bool = False,
        enable_kv_block_prefix_routing: bool = False,
        enable_load_score_routing: bool = False,
        expert_affinity_routing_weight: float = 1.0,
        kv_block_prefix_routing_weight: float = 1.0,
        load_score_routing_weight: float = 1.0,
        load_balancer_debug: bool = False,
        kv_block_prefix_block_size: int = 0,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination
        self.expert_affinity_enabled = enable_prefix_affinity_routing
        self.kv_block_prefix_enabled = (
            enable_kv_block_prefix_routing and kv_block_prefix_block_size > 0
        )
        self.load_score_enabled = enable_load_score_routing
        self.expert_affinity_weight = (
            expert_affinity_routing_weight if self.expert_affinity_enabled else 0.0
        )
        self.kv_block_prefix_weight = (
            kv_block_prefix_routing_weight if self.kv_block_prefix_enabled else 0.0
        )
        self.load_score_weight = (
            load_score_routing_weight if self.load_score_enabled else 0.0
        )
        self.load_balancer_debug = load_balancer_debug
        self.kv_block_prefix_block_size = kv_block_prefix_block_size
        self.expert_affinity_epoch = 0
        self.prefix_router_placement_epoch: int | None = None
        self.expert_affinity_trees = {
            rank: TokenRadixTree() for rank in range(engine_count)
        }
        self.kv_block_prefix_indices = {
            rank: ExactBlockPrefixIndex() for rank in range(engine_count)
        }

    def _apply_runtime_routing_weights(
        self,
        expert_affinity_routing_weight: float,
        kv_block_prefix_routing_weight: float,
        load_score_routing_weight: float,
    ) -> None:
        self.expert_affinity_weight = (
            float(expert_affinity_routing_weight)
            if self.expert_affinity_enabled
            else 0.0
        )
        self.kv_block_prefix_weight = (
            float(kv_block_prefix_routing_weight)
            if self.kv_block_prefix_enabled
            else 0.0
        )
        self.load_score_weight = (
            float(load_score_routing_weight)
            if self.load_score_enabled
            else 0.0
        )

    def _clear_prefix_router_state(self, epoch: int) -> None:
        for tree in self.expert_affinity_trees.values():
            tree.clear()
        self.expert_affinity_epoch = epoch

    def _apply_prefix_router_update(
        self,
        target_rank: int,
        epoch: int,
        prompt_token_ids: list[int],
    ) -> None:
        if not self.expert_affinity_enabled or not prompt_token_ids:
            return
        if epoch > self.expert_affinity_epoch:
            self._clear_prefix_router_state(epoch)
        elif epoch < self.expert_affinity_epoch:
            return

        tree = self.expert_affinity_trees.get(target_rank)
        if tree is None:
            tree = self.expert_affinity_trees[target_rank] = TokenRadixTree()
        tree.insert(prompt_token_ids)

    def _apply_prefix_router_placement_update(self, epoch: int) -> bool:
        if not self.expert_affinity_enabled:
            return False
        if self.prefix_router_placement_epoch == epoch:
            return False
        self.prefix_router_placement_epoch = epoch
        if epoch > self.expert_affinity_epoch:
            self._clear_prefix_router_state(epoch)
        return True

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
            if prefix_len > 0:
                scores[rank] = prefix_len / total_tokens
        return scores

    def _get_kv_block_prefix_scores(
        self,
        query: CoordinatorRouteRequest,
    ) -> dict[int, float]:
        if not self.kv_block_prefix_enabled or query.kv_total_blocks <= 0:
            return {}

        zero_block = (
            (0,) * self.kv_block_prefix_block_size if query.kv_use_zero_tokens else None
        )
        scores: dict[int, float] = {}
        for rank, index in self.kv_block_prefix_indices.items():
            matched_blocks = index.longest_prefix_blocks_from_parts(
                token_ids=query.prompt_token_ids,
                total_blocks=query.kv_total_blocks,
                block_size=self.kv_block_prefix_block_size,
                extra_keys=query.kv_extra_keys,
                zero_block=zero_block,
            )
            if matched_blocks > 0:
                scores[rank] = matched_blocks / query.kv_total_blocks
        return scores

    def _get_load_scores(self) -> dict[int, float]:
        if not self.load_score_enabled or not self.engines:
            return {}

        raw_scores = [
            (state.request_counts[0] * 4) + state.request_counts[1]
            for state in self.engines
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

    def _choose_engine_by_load(
        self,
        client_index: int,
        client_count: int,
        candidate_indices: list[int] | None = None,
    ) -> int:
        indices = candidate_indices or list(range(len(self.engines)))
        if not indices:
            return 0

        min_score = float("inf")
        chosen_index = 0
        eng_start_index = 0
        if client_count > 0:
            eng_start_index = (len(self.engines) * client_index) // client_count

        for i in range(len(indices)):
            idx = indices[(eng_start_index + i) % len(indices)]
            waiting, running = self.engines[idx].request_counts
            score = (waiting * 4) + running
            if score < min_score:
                min_score = score
                chosen_index = idx

        self.engines[chosen_index].request_counts[0] += 1
        return chosen_index

    def _route_request(
        self,
        query: CoordinatorRouteRequest,
    ) -> CoordinatorRouteResponse:
        expert_scores = (
            self._get_expert_affinity_scores(query.prompt_token_ids)
            if self.expert_affinity_weight > 0.0
            else {}
        )
        kv_scores = (
            self._get_kv_block_prefix_scores(query)
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
            chosen_index = self._choose_engine_by_load(
                query.client_index,
                query.client_count,
                matched_indices,
            )
            if self.load_balancer_debug:
                logger.warning(
                    "LB route request_id=%s expert_scores=%s kv_scores=%s "
                    "load_scores=%s final_scores=%s chosen_engine=%d best_score=%.4f",
                    query.request_id,
                    self._format_score_map(expert_scores),
                    self._format_score_map(kv_scores),
                    self._format_score_map(load_scores),
                    self._format_score_map(score_by_index),
                    chosen_index,
                    best_score,
                )
            return CoordinatorRouteResponse(
                call_id=query.call_id,
                engine_index=chosen_index,
                score=best_score,
            )

        chosen_index = self._choose_engine_by_load(
            query.client_index,
            query.client_count,
        )
        if self.load_balancer_debug:
            logger.warning(
                "LB route request_id=%s expert_scores=%s kv_scores=%s "
                "load_scores=%s final_scores={} chosen_engine=%d fallback=load_only",
                query.request_id,
                self._format_score_map(expert_scores),
                self._format_score_map(kv_scores),
                self._format_score_map(load_scores),
                chosen_index,
            )
        return CoordinatorRouteResponse(
            call_id=query.call_id,
            engine_index=chosen_index,
        )
    # ///////////// Expert-based load balancing

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        # ///////////// Expert-based load balancing
        front_route_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        enable_prefix_affinity_routing: bool = False,
        enable_kv_block_prefix_routing: bool = False,
        enable_load_score_routing: bool = False,
        expert_affinity_routing_weight: float = 1.0,
        kv_block_prefix_routing_weight: float = 1.0,
        load_score_routing_weight: float = 1.0,
        load_balancer_debug: bool = False,
        kv_block_prefix_block_size: int = 0,
        # ///////////// Expert-based load balancing
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
            enable_prefix_affinity_routing=enable_prefix_affinity_routing,
            enable_kv_block_prefix_routing=enable_kv_block_prefix_routing,
            enable_load_score_routing=enable_load_score_routing,
            expert_affinity_routing_weight=expert_affinity_routing_weight,
            kv_block_prefix_routing_weight=kv_block_prefix_routing_weight,
            load_score_routing_weight=load_score_routing_weight,
            load_balancer_debug=load_balancer_debug,
            kv_block_prefix_block_size=kv_block_prefix_block_size,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                front_route_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
            # ///////////// Expert-based load balancing
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        # ///////////// Expert-based load balancing
        front_route_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)
        route_decoder = MsgpackDecoder(CoordinatorRouteRequest)

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=front_route_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.ROUTER,
                bind=True,
            ) as route_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            route_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # Wait until all engines subscribe.
            num_expected = len(self.engines)
            for subscribed in range(num_expected):
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
                logger.info(
                    "DP Coordinator received engine subscription %d/%d",
                    subscribed + 1,
                    num_expected,
                )
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(route_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_back in events:
                    buffer = publish_back.recv()
                    if buffer == b"\x01":
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            for rank in range(current_count, new_engine_count):
                                self.expert_affinity_trees[rank] = TokenRadixTree()
                                self.kv_block_prefix_indices[rank] = (
                                    ExactBlockPrefixIndex()
                                )
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            self.expert_affinity_trees = {
                                rank: tree
                                for rank, tree in self.expert_affinity_trees.items()
                                if rank < new_engine_count
                            }
                            self.kv_block_prefix_indices = {
                                rank: index
                                for rank, index in self.kv_block_prefix_indices.items()
                                if rank < new_engine_count
                            }
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue  # Skip normal engine notification processing

                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 5
                        and decoded[0] == "PREFIX_ROUTER_UPDATE"
                    ):
                        _, _, target_rank, epoch, prompt_token_ids = decoded
                        self._apply_prefix_router_update(
                            int(target_rank),
                            int(epoch),
                            list(prompt_token_ids),
                        )
                        continue

                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 3
                        and decoded[0] == "PREFIX_ROUTER_UPDATE_BATCH"
                    ):
                        _, _, updates = decoded
                        for update in updates:
                            if len(update) != 3:
                                continue
                            target_rank, epoch, prompt_token_ids = update
                            self._apply_prefix_router_update(
                                int(target_rank),
                                int(epoch),
                                list(prompt_token_ids),
                            )
                        continue

                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 5
                        and decoded[0] == "LB_WEIGHT_UPDATE"
                    ):
                        _, source_client_index, expert_weight, kv_weight, load_weight = (
                            decoded
                        )
                        self._apply_runtime_routing_weights(
                            float(expert_weight),
                            float(kv_weight),
                            float(load_weight),
                        )
                        publish_front.send(
                            msgspec.msgpack.encode(
                                (
                                    "LB_WEIGHT_UPDATE",
                                    int(source_client_index),
                                    self.expert_affinity_weight,
                                    self.kv_block_prefix_weight,
                                    self.load_score_weight,
                                )
                            )
                        )
                        continue

                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        # We received a message on the front-end XPUB socket,
                        # from an API server sending a new request while the
                        # engines are paused, so that we can wake the other
                        # engines.
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            if wave < current_wave:
                                # If the wave number is stale, ensure the message
                                # is handled by all the engines.
                                engine_to_exclude = None

                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if route_front in events:
                    frames = route_front.recv_multipart()
                    if len(frames) >= 2:
                        identity = frames[0]
                        route_request = route_decoder.decode(frames[-1])
                        route_response = self._route_request(route_request)
                        route_front.send_multipart(
                            (identity, msgspec.msgpack.encode(route_response))
                        )

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    if outputs.prefix_router_placement_update is not None:
                        update = outputs.prefix_router_placement_update
                        epoch = int(update["epoch"])
                        if self._apply_prefix_router_placement_update(epoch):
                            publish_front.send(
                                msgspec.msgpack.encode(
                                    (
                                        "PREFIX_ROUTER_PLACEMENT_UPDATE",
                                        epoch,
                                        update["physical_to_logical_map"],
                                    )
                                )
                            )
                    if outputs.kv_cache_event_batch is not None:
                        self._apply_kv_prefix_event_batch(
                            eng_index,
                            outputs.kv_cache_event_batch,
                        )
                    # ///////////// Expert-based load balancing

                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
