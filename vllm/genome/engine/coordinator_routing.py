# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec.msgpack
import zmq

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.logger import init_logger
from vllm.v1.engine import (
    CoordinatorRouteRequest,
    CoordinatorRouteResponse,
    EngineCoreOutputs,
)
from vllm.v1.prefix_router import (
    ExactBlockPrefixIndex,
    ExpertAffinityIndex,
    make_expert_affinity_index,
)
from vllm.v1.serial_utils import MsgpackDecoder

logger = init_logger(__name__)


class GenomeDPCoordinatorRoutingMixin:

    def _init_genome_routing_state(
        self,
        *,
        engine_count: int,
        enable_prefix_affinity_routing: bool,
        enable_kv_block_prefix_routing: bool,
        enable_load_score_routing: bool,
        expert_affinity_routing_weight: float,
        kv_block_prefix_routing_weight: float,
        load_score_routing_weight: float,
        prefix_learning_algorithm: str,
        load_balancer_debug: bool,
        kv_block_prefix_block_size: int,
    ) -> None:
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
        self.prefix_learning_algorithm = prefix_learning_algorithm
        self.load_balancer_debug = load_balancer_debug
        self.kv_block_prefix_block_size = kv_block_prefix_block_size
        self.kv_block_prefix_decoder = MsgpackDecoder(KVEventBatch)
        self.expert_affinity_epoch = 0
        self.prefix_router_placement_epoch: int | None = None
        self.expert_affinity_indices: dict[int, ExpertAffinityIndex] = {
            rank: self._make_expert_affinity_index() for rank in range(engine_count)
        }
        self.kv_block_prefix_indices = {
            rank: ExactBlockPrefixIndex() for rank in range(engine_count)
        }

    def _make_expert_affinity_index(self) -> ExpertAffinityIndex:
        return make_expert_affinity_index(self.prefix_learning_algorithm)

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
        for index in self.expert_affinity_indices.values():
            index.clear()
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

        index = self.expert_affinity_indices.get(target_rank)
        if index is None:
            index = self.expert_affinity_indices[target_rank] = (
                self._make_expert_affinity_index()
            )
        index.insert(prompt_token_ids)

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

        scores: dict[int, float] = {}
        for rank, index in self.expert_affinity_indices.items():
            score = index.score(prompt_token_ids)
            if score > 0.0:
                scores[rank] = score
        return self._normalize_score_map(scores)

    def _get_kv_block_prefix_scores(
        self,
        query: CoordinatorRouteRequest,
    ) -> dict[int, float]:
        if not self.kv_block_prefix_enabled or query.kv_total_blocks <= 0:
            return {}

        zero_block = (
            (0,) * self.kv_block_prefix_block_size
            if query.kv_use_zero_tokens
            else None
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
        return self._normalize_score_map(scores)

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
            precedence.append(
                ("expert_affinity", self.expert_affinity_weight, expert_scores)
            )
        if self.kv_block_prefix_weight > 0.0:
            precedence.append(
                ("kv_block_prefix", self.kv_block_prefix_weight, kv_scores)
            )
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

        candidate_indices = list(range(len(self.engines)))
        selection_trace: list[str] = []
        for name, _, scores in precedence:
            candidate_scores = {
                idx: scores.get(idx, 0.0) for idx in candidate_indices
            }
            best_score = max(candidate_scores.values(), default=0.0)
            candidate_indices = [
                idx for idx, score in candidate_scores.items()
                if score == best_score
            ]
            selection_trace.append(f"{name}:{best_score:.4f}->{candidate_indices}")
            if len(candidate_indices) <= 1:
                break

        return candidate_indices, selection_trace

    @staticmethod
    def _format_score_map(scores: dict[int, float]) -> str:
        if not scores:
            return "{}"
        items = ", ".join(
            f"{idx}:{score:.4f}" for idx, score in sorted(scores.items())
        )
        return "{" + items + "}"

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
        matched_indices, selection_trace = (
            self._select_candidates_by_routing_precedence(
                expert_scores,
                kv_scores,
                load_scores,
            )
        )

        if matched_indices is not None:
            chosen_index = self._choose_engine_by_load(
                query.client_index,
                query.client_count,
                matched_indices,
            )
            if self.load_balancer_debug:
                logger.warning(
                    "LB route request_id=%s expert_scores=%s kv_scores=%s "
                    "load_scores=%s precedence=%s chosen_engine=%d",
                    query.request_id,
                    self._format_score_map(expert_scores),
                    self._format_score_map(kv_scores),
                    self._format_score_map(load_scores),
                    selection_trace,
                    chosen_index,
                )
            return CoordinatorRouteResponse(
                call_id=query.call_id,
                engine_index=chosen_index,
                score=1.0 if len(matched_indices) == 1 else 0.0,
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

    def _handle_scale_elastic_ep_message(self, decoded: object) -> bool:
        if not (
            isinstance(decoded, (list, tuple))
            and len(decoded) == 2
            and decoded[0] == "SCALE_ELASTIC_EP"
        ):
            return False

        new_engine_count = decoded[1]
        current_count = len(self.engines)
        if new_engine_count > current_count:
            for _ in range(new_engine_count - current_count):
                self.engines.append(type(self.engines[0])())
            for rank in range(current_count, new_engine_count):
                self.expert_affinity_indices[rank] = self._make_expert_affinity_index()
                self.kv_block_prefix_indices[rank] = ExactBlockPrefixIndex()
            logger.info(
                "DPCoordinator scaled up from %s to %s engines",
                current_count,
                new_engine_count,
            )
        else:
            self.engines = self.engines[:new_engine_count]
            self.expert_affinity_indices = {
                rank: index
                for rank, index in self.expert_affinity_indices.items()
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
        return True

    def _handle_frontend_control_message(
        self,
        decoded: object,
        publish_front: zmq.Socket,
    ) -> bool:
        if self._handle_scale_elastic_ep_message(decoded):
            return True

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
            return True

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
            return True

        if (
            isinstance(decoded, (list, tuple))
            and len(decoded) == 3
            and decoded[0] == "KV_PREFIX_EVENT_BATCH"
        ):
            _, target_rank, payload = decoded
            batch = self.kv_block_prefix_decoder.decode(payload)
            self._apply_kv_prefix_event_batch(int(target_rank), batch)
            return True

        if (
            isinstance(decoded, (list, tuple))
            and len(decoded) == 5
            and decoded[0] == "LB_WEIGHT_UPDATE"
        ):
            _, source_client_index, expert_weight, kv_weight, load_weight = decoded
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
            return True

        return False

    def _handle_route_query_event(
        self,
        route_front: zmq.Socket,
        route_decoder: MsgpackDecoder,
    ) -> None:
        frames = route_front.recv_multipart()
        if len(frames) >= 2:
            identity = frames[0]
            route_request = route_decoder.decode(frames[-1])
            route_response = self._route_request(route_request)
            route_front.send_multipart(
                (identity, msgspec.msgpack.encode(route_response))
            )

    def _handle_engine_output_custom(
        self,
        outputs: EngineCoreOutputs,
        publish_front: zmq.Socket,
    ) -> int:
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
        return eng_index
