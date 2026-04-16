"""
placement_fns.py — Expert placement callback for combined_launch.py.

Registered via:
    llm.collective_rpc("register_placement_callback",
                       args=(path_to_this_file, "compute_placement"))

The callback runs after every forward pass once routing load has been
all-reduced across every EP rank. This implementation keeps the callback fully
in-memory and deterministic across ranks:

- placement decisions only use ``expert_load`` and ``num_gpus``
- no JSON files are read
- local-only fields like ``topk_ids`` are ignored

The algorithm is METIS-style rather than file-driven METIS:

1. For each MoE layer, treat experts as graph nodes.
2. Assign node weights from the global per-expert token load.
3. Connect experts with weighted edges derived from shared load intensity.
4. Partition the graph into ``num_gpus`` parts using ``pymetis`` when
   available, otherwise use a deterministic greedy graph-growing fallback.
5. Return ``layer_configs`` so each layer can use its own expert->GPU map.
"""

from __future__ import annotations

import heapq
import os
from collections import defaultdict

import torch

try:
    import pymetis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymetis = None


# Skip rebalancing if every layer is already close to perfectly balanced.
IMBALANCE_THRESHOLD = 0.05

# Scale floating/token loads to integer node / edge weights for graph
# partitioning. The exact value is not important as long as it is stable.
WEIGHT_SCALE = 1024


def _scale_weight(value: float) -> int:
    """Convert a non-negative float weight to a positive METIS integer weight."""
    if value <= 0:
        return 1
    return max(1, int(round(value * WEIGHT_SCALE)))


def _greedy_pack(loads: list[float], num_gpus: int) -> dict[str, int]:
    """Assign experts to GPUs by greedily minimizing the max GPU load."""
    gpu_heap = [(0.0, gpu_id) for gpu_id in range(num_gpus)]
    heapq.heapify(gpu_heap)
    assignment: dict[str, int] = {}
    for expert_id in sorted(range(len(loads)), key=lambda idx: (-loads[idx], idx)):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        assignment[str(expert_id)] = gpu_id
        heapq.heappush(gpu_heap, (gpu_load + loads[expert_id], gpu_id))
    return assignment


def _is_balanced(loads: list[float], num_gpus: int) -> bool:
    """Return True if the best greedy packing is already near-balanced."""
    if not loads or sum(loads) == 0:
        return True

    gpu_heap = [(0.0, gpu_id) for gpu_id in range(num_gpus)]
    heapq.heapify(gpu_heap)
    for expert_load in sorted(loads, reverse=True):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        heapq.heappush(gpu_heap, (gpu_load + expert_load, gpu_id))

    gpu_loads = [gpu_load for gpu_load, _ in gpu_heap]
    avg = sum(gpu_loads) / num_gpus
    if avg == 0:
        return True
    spread = (max(gpu_loads) - min(gpu_loads)) / avg
    return spread < IMBALANCE_THRESHOLD


def _build_expert_graph(loads: list[float]) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Build a dense, weighted expert graph from per-expert load.

    Nodes are experts in one layer. Node weights are the expert loads. Edge
    weights use ``min(load_i, load_j)`` so experts that are both heavily used
    are more likely to stay together during partitioning.
    """
    num_experts = len(loads)
    node_weights = [_scale_weight(load) for load in loads]
    adjacency: list[list[int]] = [[] for _ in range(num_experts)]
    edge_weights: list[list[int]] = [[] for _ in range(num_experts)]

    for left in range(num_experts):
        for right in range(left + 1, num_experts):
            shared_weight = min(loads[left], loads[right])
            if shared_weight <= 0:
                continue
            scaled = _scale_weight(shared_weight)
            adjacency[left].append(right)
            edge_weights[left].append(scaled)
            adjacency[right].append(left)
            edge_weights[right].append(scaled)

    return node_weights, adjacency, edge_weights


def _pymetis_partition(
    loads: list[float],
    num_parts: int,
) -> list[int] | None:
    """Partition experts with pymetis when it is installed."""
    if pymetis is None:
        return None

    node_weights, adjacency, edge_weights = _build_expert_graph(loads)
    if not any(adjacency):
        return None

    try:
        _, parts = pymetis.part_graph(
            num_parts,
            adjacency=adjacency,
            vweights=node_weights,
            eweights=edge_weights,
        )
    except TypeError:
        # Older pymetis versions may not support keyword arguments uniformly.
        return None
    except Exception:
        return None

    return [int(part) for part in parts]


def _fallback_partition(loads: list[float], num_parts: int) -> list[int]:
    """Deterministic graph-growing fallback when pymetis is unavailable.

    Seeds the heaviest experts first, then assigns remaining experts to the
    partition where they have the strongest edge affinity, with load balancing
    as a secondary tie-breaker.
    """
    num_experts = len(loads)
    parts = [-1] * num_experts
    part_loads = [0.0] * num_parts
    node_weights, adjacency, edge_weights = _build_expert_graph(loads)

    order = sorted(range(num_experts), key=lambda idx: (-loads[idx], idx))
    seeds = order[:num_parts]
    for part_id, expert_id in enumerate(seeds):
        parts[expert_id] = part_id
        part_loads[part_id] += loads[expert_id]

    for expert_id in order[num_parts:]:
        affinity_by_part = [0] * num_parts
        for neighbor, weight in zip(adjacency[expert_id], edge_weights[expert_id]):
            neighbor_part = parts[neighbor]
            if neighbor_part != -1:
                affinity_by_part[neighbor_part] += weight

        best_part = min(
            range(num_parts),
            key=lambda part_id: (
                -affinity_by_part[part_id],
                part_loads[part_id],
                part_id,
            ),
        )
        parts[expert_id] = best_part
        part_loads[best_part] += loads[expert_id]

    return parts


def _partition_experts(loads: list[float], num_gpus: int) -> dict[str, int]:
    """Return an expert->GPU map using METIS-style graph partitioning."""
    if not loads:
        return {}

    num_experts = len(loads)
    num_parts = max(1, min(num_gpus, num_experts))
    if num_parts == num_experts:
        return {str(expert_id): expert_id for expert_id in range(num_experts)}

    parts = _pymetis_partition(loads, num_parts)
    if parts is None:
        parts = _fallback_partition(loads, num_parts)

    experts_by_part: dict[int, list[int]] = defaultdict(list)
    for expert_id, part_id in enumerate(parts):
        experts_by_part[int(part_id)].append(expert_id)

    # Map partitions to GPUs by descending partition load. This keeps the
    # heaviest partitions on the emptiest GPUs deterministically.
    part_loads = {
        part_id: sum(loads[expert_id] for expert_id in experts)
        for part_id, experts in experts_by_part.items()
    }
    gpu_heap = [(0.0, gpu_id) for gpu_id in range(num_gpus)]
    heapq.heapify(gpu_heap)
    part_to_gpu: dict[int, int] = {}
    for part_id in sorted(part_loads, key=lambda pid: (-part_loads[pid], pid)):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        part_to_gpu[part_id] = gpu_id
        heapq.heappush(gpu_heap, (gpu_load + part_loads[part_id], gpu_id))

    assignment: dict[str, int] = {}
    for expert_id in range(num_experts):
        assignment[str(expert_id)] = part_to_gpu[parts[expert_id]]
    return assignment


def compute_placement(routing: dict) -> dict:
    """Compute deterministic, in-memory expert placement from global load.

    Args:
        routing: ``{layer_id: [capture, ...]}``
            capture keys used here:
              ``expert_load``  Tensor[num_experts]  global token counts
              ``num_gpus``     int                  EP group size

    Returns:
        ``{"expert_to_gpu": {...}, "layer_configs": {"<layer_id>": {...}}}``
        or ``{}`` to skip the update.
    """
    if not routing:
        return {}

    first_caps = next(iter(routing.values()), [])
    if not first_caps:
        return {}
    first_cap = first_caps[0]
    if "expert_load" not in first_cap or "num_gpus" not in first_cap:
        return {}

    num_gpus = int(first_cap["num_gpus"])
    layer_loads: dict[int, list[float]] = {}
    for layer_id, captures in routing.items():
        if not captures:
            continue
        layer_tensor = torch.stack(
            [capture["expert_load"].detach().cpu() for capture in captures]
        ).sum(dim=0)
        layer_loads[int(layer_id)] = layer_tensor.tolist()

    if not layer_loads:
        return {}

    if all(_is_balanced(loads, num_gpus) for loads in layer_loads.values()):
        return {}

    layer_configs = {
        str(layer_id): _partition_experts(loads, num_gpus)
        for layer_id, loads in sorted(layer_loads.items())
    }

    global_loads = torch.tensor(
        [loads for _, loads in sorted(layer_loads.items())], dtype=torch.float32
    ).sum(dim=0).tolist()

    return {
        "expert_to_gpu": _partition_experts(global_loads, num_gpus),
        "layer_configs": layer_configs,
    }
