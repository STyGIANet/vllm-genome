# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

PREFIX_ROUTER_LEARNING_TIMEOUT_S = 2.0


@dataclass
class InFlightRequestInfo:
    request_id: str
    engine: bytes
    prompt_token_ids: list[int] | None
    expert_affinity_prefill_learned: bool = False
    pending_engine_indices: tuple[int, ...] = ()
    pending_slot_released: bool = False


@dataclass
class QueuedDispatchRequest:
    request: Any
    dispatched: Any
    cancelled: bool = False


@dataclass
class PrefixLearningWorkItem:
    request_id: str
    engine: bytes
    prompt_token_ids: list[int]
    layer_expert_pairs: Any
    owner: dict[str, int] | None = None


@dataclass
class PrefixLearningContext:
    engine: bytes
    prompt_token_ids: list[int]


@dataclass
class PrefixRouterUpdate:
    target_rank: int
    epoch: int
    prompt_token_ids: list[int]
