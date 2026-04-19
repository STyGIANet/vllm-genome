"""
placement_fns.py — Expert placement callback for combined_launch.py.

Registered via:
    llm.collective_rpc("register_placement_callback",
                       args=(path_to_this_file, "compute_placement"))

The callback runs after every forward pass once routing load has been
all-reduced across every EP rank. This implementation keeps the callback fully
in-memory and deterministic across ranks:

- placement decisions only use globally aggregated routing fields
- no JSON files are read
- local-only fields like ``topk_ids`` are only used as a local fallback in
  single-process tests; distributed runs rely on the global
  ``expert_pair_load`` tensor injected by the worker

The algorithm is METIS-style rather than file-driven METIS:

1. For each MoE layer, treat experts as graph nodes.
2. Assign node weights from the global per-expert token load.
3. Connect experts with weighted edges equal to the number of tokens that
   activated both experts.
4. Partition the graph into ``num_gpus`` parts using ``pymetis`` when
   available, otherwise use a deterministic greedy graph-growing fallback.
5. Return ``layer_configs`` so each layer can use its own expert->GPU map.
"""

from __future__ import annotations

import heapq
import os
from logging import getLogger
from collections import defaultdict

import torch

try:
    import pymetis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymetis = None

logger = getLogger(__name__)

# Skip rebalancing if every layer is already close to perfectly balanced.
IMBALANCE_THRESHOLD = 0.05

# Scale floating/token loads to integer node / edge weights for graph
# partitioning. The exact value is not important as long as it is stable.
WEIGHT_SCALE = 1024
COACT_LOG_TOPK = 8


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


def _coactivation_from_topk_ids(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> list[list[float]]:
    """Derive pairwise co-activation counts from token routing ids."""
    pair_counts = [[0.0] * num_experts for _ in range(num_experts)]
    if topk_ids.numel() == 0 or topk_ids.shape[1] < 2:
        return pair_counts

    for token_experts in topk_ids.detach().cpu().tolist():
        unique_experts = sorted({int(expert_id) for expert_id in token_experts})
        for left_idx, left in enumerate(unique_experts[:-1]):
            for right in unique_experts[left_idx + 1:]:
                pair_counts[left][right] += 1.0
                pair_counts[right][left] += 1.0
    return pair_counts


def _build_expert_graph(
    loads: list[float],
    pair_counts: list[list[float]] | None = None,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Build a weighted expert graph from per-expert load and co-activation.

    Nodes are experts in one layer. Node weights are the expert loads. Edge
    weights count how many tokens activated both experts.
    """
    num_experts = len(loads)
    node_weights = [_scale_weight(load) for load in loads]
    adjacency: list[list[int]] = [[] for _ in range(num_experts)]
    edge_weights: list[list[int]] = [[] for _ in range(num_experts)]
    pair_counts = pair_counts or [[0.0] * num_experts for _ in range(num_experts)]

    for left in range(num_experts):
        for right in range(left + 1, num_experts):
            shared_weight = pair_counts[left][right]
            if shared_weight <= 0:
                continue
            scaled = _scale_weight(shared_weight)
            adjacency[left].append(right)
            edge_weights[left].append(scaled)
            adjacency[right].append(left)
            edge_weights[right].append(scaled)

    return node_weights, adjacency, edge_weights


def _log_coactivation_summary(
    layer_id: int,
    loads: list[float],
    pair_counts: list[list[float]] | None,
) -> None:
    """Log the hottest co-activating expert pairs for one layer."""
    if os.environ.get("VLLM_DP_RANK", "0") != "0":
        return
    if not pair_counts:
        return

    num_experts = len(loads)
    hot_pairs: list[tuple[float, int, int]] = []
    for left in range(num_experts):
        for right in range(left + 1, num_experts):
            pair_weight = float(pair_counts[left][right])
            if pair_weight <= 0:
                continue
            hot_pairs.append((pair_weight, left, right))

    if not hot_pairs:
        logger.info("[co-act] layer=%s no co-activated expert pairs", layer_id)
        return

    hot_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    pair_summary = "  ".join(
        f"E{left}-E{right}:{int(weight)}"
        for weight, left, right in hot_pairs[:COACT_LOG_TOPK]
    )
    logger.info(
        "[co-act] layer=%s top_pairs=%s total_pair_weight=%s",
        layer_id,
        pair_summary,
        int(sum(weight for weight, _, _ in hot_pairs)),
    )


def _log_partition_coactivation_summary(
    layer_id: int,
    assignment: dict[str, int],
    pair_counts: list[list[float]] | None,
) -> None:
    """Log how much co-activation is kept within partitions vs cut across them."""
    if os.environ.get("VLLM_DP_RANK", "0") != "0":
        return
    if not pair_counts:
        return

    within_weight = 0.0
    cut_weight = 0.0
    num_experts = len(pair_counts)
    for left in range(num_experts):
        for right in range(left + 1, num_experts):
            pair_weight = float(pair_counts[left][right])
            if pair_weight <= 0:
                continue
            left_gpu = assignment.get(str(left))
            right_gpu = assignment.get(str(right))
            if left_gpu is None or right_gpu is None:
                continue
            if left_gpu == right_gpu:
                within_weight += pair_weight
            else:
                cut_weight += pair_weight

    total_weight = within_weight + cut_weight
    kept_ratio = 0.0 if total_weight == 0 else within_weight / total_weight
    logger.info(
        "[co-act] layer=%s kept=%s cut=%s kept_ratio=%.4f",
        layer_id,
        int(within_weight),
        int(cut_weight),
        kept_ratio,
    )


def _pymetis_partition(
    loads: list[float],
    pair_counts: list[list[float]] | None,
    num_parts: int,
) -> list[int] | None:
    """Partition experts with pymetis when it is installed."""
    if pymetis is None:
        return None

    node_weights, adjacency, edge_weights = _build_expert_graph(loads, pair_counts)
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


def _fallback_partition(
    loads: list[float],
    pair_counts: list[list[float]] | None,
    num_parts: int,
) -> list[int]:
    """Deterministic graph-growing fallback when pymetis is unavailable.

    Seeds the heaviest experts first, then assigns remaining experts to the
    partition where they have the strongest edge affinity, with load balancing
    as a secondary tie-breaker.
    """
    num_experts = len(loads)
    parts = [-1] * num_experts
    part_loads = [0.0] * num_parts
    node_weights, adjacency, edge_weights = _build_expert_graph(loads, pair_counts)

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


def _partition_experts(
    loads: list[float],
    num_gpus: int,
    pair_counts: list[list[float]] | None = None,
) -> dict[str, int]:
    """Return an expert->GPU map using METIS-style graph partitioning."""
    if not loads:
        return {}

    num_experts = len(loads)
    num_parts = max(1, min(num_gpus, num_experts))
    if num_parts == num_experts:
        return {str(expert_id): expert_id for expert_id in range(num_experts)}

    parts = _pymetis_partition(loads, pair_counts, num_parts)
    if parts is None:
        parts = _fallback_partition(loads, pair_counts, num_parts)

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
              ``expert_load``       Tensor[num_experts] global token counts
              ``expert_pair_load``  Tensor[num_experts, num_experts]
                                    global pairwise co-activation counts
              ``num_gpus``          int EP group size

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
    layer_pair_loads: dict[int, list[list[float]]] = {}
    for layer_id, captures in routing.items():
        if not captures:
            continue
        layer_tensor = torch.stack(
            [capture["expert_load"].detach().cpu() for capture in captures]
        ).sum(dim=0)
        layer_loads[int(layer_id)] = layer_tensor.tolist()
        if "expert_pair_load" in captures[0]:
            layer_pair_loads[int(layer_id)] = (
                captures[0]["expert_pair_load"].detach().cpu().tolist()
            )
        elif "topk_ids" in captures[0]:
            layer_pair_loads[int(layer_id)] = _coactivation_from_topk_ids(
                captures[0]["topk_ids"],
                len(layer_loads[int(layer_id)]),
            )

    if not layer_loads:
        return {}

    if all(_is_balanced(loads, num_gpus) for loads in layer_loads.values()):
        return {}

    for layer_id, loads in sorted(layer_loads.items()):
        _log_coactivation_summary(
            layer_id,
            loads,
            layer_pair_loads.get(layer_id),
        )

    layer_configs: dict[str, dict[str, int]] = {}
    for layer_id, loads in sorted(layer_loads.items()):
        assignment = _partition_experts(
            loads,
            num_gpus,
            layer_pair_loads.get(layer_id),
        )
        layer_configs[str(layer_id)] = assignment
        _log_partition_coactivation_summary(
            layer_id,
            assignment,
            layer_pair_loads.get(layer_id),
        )

    global_loads = torch.tensor(
        [loads for _, loads in sorted(layer_loads.items())], dtype=torch.float32
    ).sum(dim=0).tolist()

    return {
        "expert_to_gpu": _partition_experts(global_loads, num_gpus),
        "layer_configs": layer_configs,
    }
