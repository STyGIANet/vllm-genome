"""
placement_fns.py - Expert placement callback for combined_launch.py.

Registered via:
    llm.collective_rpc("register_placement_callback",
                       args=(path_to_this_file, "compute_placement"))

The callback runs on the normal EPLB rearrangement cadence.
"""

import logging

import torch
from vllm.logger import init_logger

try:
    import pymetis
except ImportError:
    pymetis = None


logger = init_logger("vllm.genome.placement")

def _greedy_pack(loads: list[float], num_gpus: int) -> dict[str, int]:
    gpu_loads = [0.0] * num_gpus
    gpu_counts = [0] * num_gpus
    slots_per_gpu = max(1, len(loads) // num_gpus)
    assignment: dict[str, int] = {}

    for expert_id in sorted(range(len(loads)), key=lambda e: -loads[e]):
        candidates = [
            gpu_id for gpu_id in range(num_gpus)
            if gpu_counts[gpu_id] < slots_per_gpu
        ] or list(range(num_gpus))
        gpu_id = min(candidates, key=lambda g: (gpu_loads[g], gpu_counts[g], g))
        assignment[str(expert_id)] = gpu_id
        gpu_counts[gpu_id] += 1
        gpu_loads[gpu_id] += loads[expert_id]
    return assignment

def _build_layer_loads_from_node_activation_counts(
    node_activation_counts: torch.Tensor,
    num_layers: int,
    num_experts: int,
) -> dict[int, list[int]]:
    counts = node_activation_counts.to(torch.int64).cpu().reshape(
        num_layers, num_experts
    )
    return {
        int(layer_id): [int(v) for v in counts[layer_id].tolist()]
        for layer_id in range(num_layers)
    }


def _build_metis_inputs(
    coactivation_edges: torch.Tensor,
    num_nodes: int,
) -> tuple[list[int], list[int], list[int]]:
    pair_rows, pair_cols = torch.triu_indices(num_nodes, num_nodes, offset=1)
    weights = coactivation_edges.cpu()
    nonzero_mask = weights > 0

    adjacency_lists: list[list[int]] = [[] for _ in range(num_nodes)]
    edge_weight_lists: list[list[int]] = [[] for _ in range(num_nodes)]

    nz_rows = pair_rows[nonzero_mask].tolist()
    nz_cols = pair_cols[nonzero_mask].tolist()
    nz_weights = weights[nonzero_mask].tolist()

    for src, dst, weight in zip(nz_rows, nz_cols, nz_weights):
        w = int(weight)
        adjacency_lists[src].append(dst)
        edge_weight_lists[src].append(w)
        adjacency_lists[dst].append(src)
        edge_weight_lists[dst].append(w)

    xadj = [0]
    adjncy: list[int] = []
    eweights: list[int] = []
    for neighbors, neighbor_weights in zip(adjacency_lists, edge_weight_lists):
        adjncy.extend(neighbors)
        eweights.extend(neighbor_weights)
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights


def _compact_graph_to_active_nodes(
    coactivation_edges: torch.Tensor,
    node_activation_counts: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, list[int]]:
    active_node_ids = torch.nonzero(
        node_activation_counts.to(torch.int64).cpu() > 0,
        as_tuple=False,
    ).flatten()
    if active_node_ids.numel() == 0:
        return coactivation_edges.new_zeros(0, dtype=torch.int64), []

    active_node_ids = active_node_ids.to(torch.int64)
    num_active_nodes = int(active_node_ids.numel())
    if num_active_nodes < 2:
        return coactivation_edges.new_zeros(0, dtype=torch.int64), (
            active_node_ids.tolist()
        )

    compact_rows, compact_cols = torch.triu_indices(
        num_active_nodes, num_active_nodes, offset=1
    )
    full_src = active_node_ids[compact_rows]
    full_dst = active_node_ids[compact_cols]
    lo = torch.minimum(full_src, full_dst)
    hi = torch.maximum(full_src, full_dst)
    full_edge_ids = lo * (2 * num_nodes - lo - 1) // 2 + (hi - lo - 1)
    compact_edges = coactivation_edges.to(torch.int64).cpu()[full_edge_ids]
    return compact_edges, active_node_ids.tolist()


def _partition_with_metis(
    coactivation_edges: torch.Tensor,
    node_activation_counts: torch.Tensor,
    layer_ids: list[int],
    num_experts: int,
    num_gpus: int,
) -> list[int] | None:
    if pymetis is None:
        logger.warning("pymetis is not installed; falling back to greedy placement.")
        return None

    num_nodes = len(layer_ids) * num_experts
    if num_nodes == 0:
        return None

    compact_edges, active_node_ids = _compact_graph_to_active_nodes(
        coactivation_edges=coactivation_edges,
        node_activation_counts=node_activation_counts,
        num_nodes=num_nodes,
    )
    num_active_nodes = len(active_node_ids)
    if num_active_nodes < 2:
        return None

    xadj, adjncy, eweights = _build_metis_inputs(compact_edges, num_active_nodes)
    if not adjncy:
        return None

    _, compact_membership = pymetis.part_graph(
        num_gpus,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
    )

    membership = [-1] * num_nodes
    for compact_node_id, full_node_id in enumerate(active_node_ids):
        membership[full_node_id] = int(compact_membership[compact_node_id])

    logger.info(
        "METIS node compaction: %d total nodes -> %d active nodes",
        num_nodes,
        num_active_nodes,
    )
    return membership


def _repair_partition_to_layer_configs(
    membership: list[int],
    layer_ids: list[int],
    layer_loads: dict[int, list[int]],
    num_experts: int,
    num_gpus: int,
) -> dict[str, dict[str, int]]:
    slots_per_gpu = max(1, num_experts // num_gpus)
    layer_configs: dict[str, dict[str, int]] = {}

    for layer_pos, layer_id in enumerate(layer_ids):
        gpu_counts = [0] * num_gpus
        gpu_loads = [0] * num_gpus
        layer_mapping: dict[str, int] = {}
        loads = layer_loads[layer_id]

        for expert_id in sorted(range(num_experts), key=lambda e: -loads[e]):
            node_id = layer_pos * num_experts + expert_id
            preferred_gpu = membership[node_id]

            if (
                preferred_gpu >= 0
                and gpu_counts[preferred_gpu] < slots_per_gpu
            ):
                chosen_gpu = preferred_gpu
            else:
                candidates = [
                    gpu_id for gpu_id in range(num_gpus)
                    if gpu_counts[gpu_id] < slots_per_gpu
                ] or list(range(num_gpus))
                chosen_gpu = min(
                    candidates,
                    key=lambda g: (gpu_loads[g], gpu_counts[g], g),
                )

            layer_mapping[str(expert_id)] = chosen_gpu
            gpu_counts[chosen_gpu] += 1
            gpu_loads[chosen_gpu] += loads[expert_id]

        layer_configs[str(layer_id)] = layer_mapping

    return layer_configs


def compute_placement(routing: dict) -> dict:
    """Compute per-layer expert placement from a global co-activation graph."""
    if not routing:
        return {}

    graph_meta = routing.get("__graph__", {})
    coactivation_edges = graph_meta.get("coactivation_edges")
    node_activation_counts = graph_meta.get("node_activation_counts")
    if coactivation_edges is None or node_activation_counts is None:
        return {}

    num_layers = int(graph_meta["num_layers"])
    num_experts = int(graph_meta["num_experts"])
    num_gpus = int(graph_meta["num_gpus"])
    layer_ids = list(range(num_layers))
    num_nodes = len(layer_ids) * num_experts

    if logger.isEnabledFor(logging.INFO):
        total = coactivation_edges.nbytes + node_activation_counts.nbytes
        logger.info("Total routing payload size: %s bytes", total)

    layer_loads = _build_layer_loads_from_node_activation_counts(
        node_activation_counts=node_activation_counts,
        num_layers=num_layers,
        num_experts=num_experts,
    )

    membership = _partition_with_metis(
        coactivation_edges=coactivation_edges,
        node_activation_counts=node_activation_counts,
        layer_ids=layer_ids,
        num_experts=num_experts,
        num_gpus=num_gpus,
    )

    if membership is None:
        layer_configs = {
            str(layer_id): _greedy_pack(loads, num_gpus)
            for layer_id, loads in layer_loads.items()
        }
    else:
        expected_nodes = num_nodes
        if len(membership) != expected_nodes:
            logger.warning(
                "METIS returned %d nodes, expected %d; falling back to greedy.",
                len(membership),
                expected_nodes,
            )
            layer_configs = {
                str(layer_id): _greedy_pack(loads, num_gpus)
                for layer_id, loads in layer_loads.items()
            }
        else:
            layer_configs = _repair_partition_to_layer_configs(
                membership=membership,
                layer_ids=layer_ids,
                layer_loads=layer_loads,
                num_experts=num_experts,
                num_gpus=num_gpus,
            )

    global_loads = [0] * num_experts
    for loads in layer_loads.values():
        for expert_id, load in enumerate(loads):
            global_loads[expert_id] += load

    return {
        "expert_to_gpu": _greedy_pack(global_loads, num_gpus),
        "layer_configs": layer_configs,
    }
