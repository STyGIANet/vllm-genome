#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, TypeAlias

from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.core.kv_cache_utils import ExternalBlockHash, generate_block_hash_extra_keys
from vllm.v1.request import Request


# ///////////// Expert-based load balancing
@dataclass
class _RadixNode:
    children: dict[int, "_RadixNode"] = field(default_factory=dict)


class TokenRadixTree:
    """A token-keyed radix tree used for prompt prefix matching."""

    def __init__(self) -> None:
        self._root = _RadixNode()

    def clear(self) -> None:
        self._root = _RadixNode()

    def insert(self, token_ids: list[int]) -> None:
        if not token_ids:
            return

        node = self._root
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                child = _RadixNode()
                node.children[token_id] = child
            node = child

    def longest_prefix_length(self, token_ids: list[int]) -> int:
        node = self._root
        depth = 0

        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            depth += 1
        return depth


PrefixRouterOwnerCache: TypeAlias = list[list[tuple[int, ...]]]


def build_owner_cache_from_physical_to_logical_map(
    physical_to_logical_map: Sequence[Sequence[int]],
    num_ranks: int,
) -> PrefixRouterOwnerCache:
    if num_ranks <= 0 or not physical_to_logical_map:
        return []

    num_layers = len(physical_to_logical_map)
    num_physical_experts = len(physical_to_logical_map[0])
    if num_physical_experts <= 0:
        return [[] for _ in range(num_layers)]

    max_logical_expert = -1
    for logical_ids in physical_to_logical_map:
        for logical_id in logical_ids:
            if logical_id > max_logical_expert:
                max_logical_expert = logical_id

    if max_logical_expert < 0:
        return [[] for _ in range(num_layers)]

    owner_sets = [
        [set() for _ in range(max_logical_expert + 1)] for _ in range(num_layers)
    ]
    slots_per_rank = max(num_physical_experts // num_ranks, 1)
    for layer_idx, logical_ids in enumerate(physical_to_logical_map):
        for physical_idx, logical_id in enumerate(logical_ids):
            if logical_id < 0:
                continue
            owner_rank = min(physical_idx // slots_per_rank, num_ranks - 1)
            owner_sets[layer_idx][logical_id].add(owner_rank)

    return [
        [tuple(sorted(rank_ids)) for rank_ids in layer_sets]
        for layer_sets in owner_sets
    ]


def compute_owner_from_routed_experts(
    routed_experts: Sequence[Sequence[Sequence[int]]],
    prompt_token_count: int,
    owner_cache: PrefixRouterOwnerCache,
    num_ranks: int,
    epoch: int,
) -> dict[str, int] | None:
    if (
        not routed_experts
        or prompt_token_count <= 0
        or not owner_cache
        or num_ranks <= 0
    ):
        return None

    num_prompt_tokens = min(prompt_token_count, len(routed_experts))
    scores = [0] * num_ranks
    seen_pairs: set[tuple[int, int]] = set()

    for token_layers in routed_experts[:num_prompt_tokens]:
        for layer_idx, layer_experts in enumerate(token_layers[: len(owner_cache)]):
            layer_owner_cache = owner_cache[layer_idx]
            for raw_expert_id in layer_experts:
                expert_id = int(raw_expert_id)
                if expert_id < 0 or expert_id >= len(layer_owner_cache):
                    continue
                layer_expert = (layer_idx, expert_id)
                if layer_expert in seen_pairs:
                    continue
                seen_pairs.add(layer_expert)
                for owner_rank in layer_owner_cache[expert_id]:
                    scores[owner_rank] += 1

    if not seen_pairs:
        return None

    target_rank = max(range(num_ranks), key=lambda rank: (scores[rank], -rank))
    return {
        "target_rank": target_rank,
        "epoch": epoch,
    }


def compute_owner_from_layer_expert_pairs(
    layer_expert_pairs: Sequence[Sequence[int]],
    owner_cache: PrefixRouterOwnerCache,
    num_ranks: int,
    epoch: int,
) -> dict[str, int] | None:
    if not layer_expert_pairs or not owner_cache or num_ranks <= 0:
        return None

    scores = [0] * num_ranks
    seen_pairs: set[tuple[int, int]] = set()

    for raw_pair in layer_expert_pairs:
        if len(raw_pair) < 2:
            continue
        layer_idx = int(raw_pair[0])
        expert_id = int(raw_pair[1])
        if layer_idx < 0 or layer_idx >= len(owner_cache):
            continue
        layer_owner_cache = owner_cache[layer_idx]
        if expert_id < 0 or expert_id >= len(layer_owner_cache):
            continue
        layer_expert = (layer_idx, expert_id)
        if layer_expert in seen_pairs:
            continue
        seen_pairs.add(layer_expert)
        for owner_rank in layer_owner_cache[expert_id]:
            scores[owner_rank] += 1

    if not seen_pairs:
        return None

    target_rank = max(range(num_ranks), key=lambda rank: (scores[rank], -rank))
    return {
        "target_rank": target_rank,
        "epoch": epoch,
    }


BlockPrefixKey: TypeAlias = tuple[tuple[int, ...], tuple[Any, ...] | None]


@dataclass
class _BlockPrefixRequestView:
    mm_features: list[Any] | None
    lora_request: Any
    cache_salt: str | None
    prompt_embeds: Any
    _prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = field(
        default_factory=dict
    )


@dataclass
class PreparedBlockPrefixQuery:
    token_ids: Sequence[int] | None
    total_blocks: int
    block_size: int
    request_view: _BlockPrefixRequestView
    zero_block: tuple[int, ...] | None = None

    @classmethod
    def from_inputs(
        cls,
        token_ids: Sequence[int] | None,
        prompt_embeds: Any,
        mm_features: list[Any] | None,
        lora_request: Any,
        cache_salt: str | None,
        block_size: int,
    ) -> "PreparedBlockPrefixQuery | None":
        if block_size <= 0:
            return None

        total_tokens = length_from_prompt_token_ids_or_embeds(token_ids, prompt_embeds)
        total_blocks = total_tokens // block_size
        if total_blocks <= 0:
            return None

        return cls(
            token_ids=token_ids,
            total_blocks=total_blocks,
            block_size=block_size,
            request_view=_BlockPrefixRequestView(
                mm_features=mm_features,
                lora_request=lora_request,
                cache_salt=cache_salt,
                prompt_embeds=prompt_embeds,
            ),
            zero_block=(0,) * block_size if token_ids is None else None,
        )


@dataclass
class _BlockPrefixNode:
    children: dict[BlockPrefixKey, "_BlockPrefixNode"] = field(default_factory=dict)
    cached_count: int = 0
    subtree_count: int = 0
    parent: "_BlockPrefixNode | None" = None
    edge_key: BlockPrefixKey | None = None


class ExactBlockPrefixIndex:
    """Tracks exact live cached block prefixes for one worker.

    The trie is keyed by block-token slices plus the per-block extra keys used
    in prefix-cache hashing. This preserves exact prefix-cache semantics while
    avoiding frontend-side block hash recomputation.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._root = _BlockPrefixNode()
        self._hash_to_node: dict[ExternalBlockHash, _BlockPrefixNode] = {}
        self._hash_counts: dict[ExternalBlockHash, int] = {}

    def store_blocks(
        self,
        block_hashes: list[ExternalBlockHash],
        token_ids: list[int],
        block_size: int,
        parent_block_hash: ExternalBlockHash | None,
        extra_keys: list[tuple[Any, ...] | None] | None = None,
    ) -> None:
        if not block_hashes or block_size <= 0:
            return

        expected_tokens = len(block_hashes) * block_size
        if len(token_ids) != expected_tokens:
            return

        if extra_keys is None:
            extra_keys = [None] * len(block_hashes)
        elif len(extra_keys) != len(block_hashes):
            return

        parent = (
            self._root
            if parent_block_hash is None
            else self._hash_to_node.get(parent_block_hash)
        )
        if parent is None:
            return

        for block_idx, block_hash in enumerate(block_hashes):
            start = block_idx * block_size
            end = start + block_size
            key: BlockPrefixKey = (
                tuple(token_ids[start:end]),
                extra_keys[block_idx],
            )
            node = parent.children.get(key)
            if node is None:
                node = _BlockPrefixNode(parent=parent, edge_key=key)
                parent.children[key] = node

            node.cached_count += 1
            curr = node
            while curr is not None:
                curr.subtree_count += 1
                curr = curr.parent

            self._hash_to_node[block_hash] = node
            self._hash_counts[block_hash] = self._hash_counts.get(block_hash, 0) + 1
            parent = node

    def remove_blocks(self, block_hashes: list[ExternalBlockHash]) -> None:
        for block_hash in block_hashes:
            count = self._hash_counts.get(block_hash)
            node = self._hash_to_node.get(block_hash)
            if not count or node is None:
                continue

            node.cached_count = max(node.cached_count - 1, 0)
            curr = node
            while curr is not None:
                curr.subtree_count = max(curr.subtree_count - 1, 0)
                parent = curr.parent
                if (
                    curr.subtree_count == 0
                    and parent is not None
                    and curr.edge_key is not None
                ):
                    parent.children.pop(curr.edge_key, None)
                curr = parent

            if count <= 1:
                self._hash_counts.pop(block_hash, None)
                self._hash_to_node.pop(block_hash, None)
            else:
                self._hash_counts[block_hash] = count - 1

    def longest_prefix_blocks(self, block_keys: list[BlockPrefixKey]) -> int:
        node = self._root
        matched = 0

        for key in block_keys:
            child = node.children.get(key)
            if child is None or child.cached_count <= 0:
                break
            node = child
            matched += 1
        return matched

    def longest_prefix_blocks_for_query(
        self,
        query: PreparedBlockPrefixQuery,
    ) -> int:
        return self.longest_prefix_blocks_from_parts(
            token_ids=query.token_ids,
            total_blocks=query.total_blocks,
            block_size=query.block_size,
            extra_keys=build_query_block_extra_keys(query),
            zero_block=query.zero_block,
        )

    def longest_prefix_blocks_from_parts(
        self,
        token_ids: Sequence[int] | None,
        total_blocks: int,
        block_size: int,
        extra_keys: list[tuple[Any, ...] | None] | None = None,
        zero_block: tuple[int, ...] | None = None,
    ) -> int:
        if total_blocks <= 0 or block_size <= 0:
            return 0

        node = self._root
        matched = 0

        for block_idx in range(total_blocks):
            key_extra = None if extra_keys is None else extra_keys[block_idx]
            start = block_idx * block_size
            end = start + block_size
            token_key = (
                zero_block
                if zero_block is not None
                else tuple(token_ids[start:end])
            )
            child = node.children.get((token_key, key_extra))
            if child is None or child.cached_count <= 0:
                break
            node = child
            matched += 1
        return matched


def prepare_block_prefix_query(
    token_ids: Sequence[int] | None,
    prompt_embeds: Any,
    mm_features: list[Any] | None,
    lora_request: Any,
    cache_salt: str | None,
    block_size: int,
) -> PreparedBlockPrefixQuery | None:
    return PreparedBlockPrefixQuery.from_inputs(
        token_ids=token_ids,
        prompt_embeds=prompt_embeds,
        mm_features=mm_features,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_size=block_size,
    )


def query_needs_block_extra_keys(query: PreparedBlockPrefixQuery) -> bool:
    request_view = query.request_view
    return bool(
        request_view.mm_features
        or request_view.lora_request is not None
        or request_view.cache_salt is not None
        or request_view.prompt_embeds is not None
    )


def build_query_block_extra_keys(
    query: PreparedBlockPrefixQuery,
) -> list[tuple[Any, ...] | None] | None:
    if not query_needs_block_extra_keys(query):
        return None

    extra_keys: list[tuple[Any, ...] | None] = []
    curr_mm_idx = 0
    for block_idx in range(query.total_blocks):
        start = block_idx * query.block_size
        end = start + query.block_size
        block_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
            query.request_view,
            start,
            end,
            curr_mm_idx,
        )
        extra_keys.append(block_extra_keys)
    return extra_keys


def build_request_block_keys(
    request: Request,
    block_size: int,
) -> list[BlockPrefixKey]:
    query = prepare_block_prefix_query(
        token_ids=request.all_token_ids,
        prompt_embeds=request.prompt_embeds,
        mm_features=request.mm_features,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        block_size=block_size,
    )
    if query is None:
        return []

    keys: list[BlockPrefixKey] = []
    extra_keys_by_block = build_query_block_extra_keys(query)

    for block_idx in range(query.total_blocks):
        start_token_idx = block_idx * query.block_size
        end_token_idx = start_token_idx + query.block_size
        keys.append(
            (
                (
                    query.zero_block
                    if query.zero_block is not None
                    else tuple(query.token_ids[start_token_idx:end_token_idx])
                ),
                None if extra_keys_by_block is None else extra_keys_by_block[block_idx],
            )
        )

    return keys
# ///////////// Expert-based load balancing
