"""
placement_fns.py — Expert placement callback for combined_launch.py.

Registered via:
    llm.collective_rpc("register_placement_callback",
                       args=(path_to_this_file, "compute_placement"))

The callback runs after every forward pass (when routing data is available).
Edit `compute_placement()` to implement your own placement algorithm.
See OVERVIEW.md for the full API contract.
"""

import heapq
import torch


# Skip rebalancing if the optimal per-GPU load spread is below this fraction of
# the mean GPU load.  At 5%, a cluster where GPUs carry 95–105% of the average
# load is left alone — unnecessary movement avoided.
IMBALANCE_THRESHOLD = 0.05


def _greedy_pack(loads: list[float], num_gpus: int) -> dict[str, int]:
    """Assign experts to GPUs minimising max GPU load (greedy bin-packing).

    Args:
        loads:    Per-expert token counts (index = expert_id).
        num_gpus: Number of GPU slots to pack into.

    Returns:
        {"<expert_id>": <gpu_id>, ...} for every expert.
    """
    gpu_heap = [(0.0, g) for g in range(num_gpus)]
    heapq.heapify(gpu_heap)
    assignment: dict[str, int] = {}
    for expert_id in sorted(range(len(loads)), key=lambda e: -loads[e]):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        assignment[str(expert_id)] = gpu_id
        heapq.heappush(gpu_heap, (gpu_load + loads[expert_id], gpu_id))
    return assignment


def _is_balanced(loads: list[float], num_gpus: int) -> bool:
    """Return True if the optimal GPU load spread is within IMBALANCE_THRESHOLD.

    Runs the same greedy pack and checks whether moving experts would produce
    a measurably unbalanced result.  If even the best packing is nearly flat,
    the current placement is probably fine — no need to move anything.
    """
    if sum(loads) == 0:
        return True

    # Simulate the greedy result to get per-GPU loads.
    gpu_heap = [(0.0, g) for g in range(num_gpus)]
    heapq.heapify(gpu_heap)
    for expert_load in sorted(loads, reverse=True):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        heapq.heappush(gpu_heap, (gpu_load + expert_load, gpu_id))

    gpu_loads = [load for load, _ in gpu_heap]
    avg = sum(gpu_loads) / num_gpus
    if avg == 0:
        return True
    spread = (max(gpu_loads) - min(gpu_loads)) / avg
    return spread < IMBALANCE_THRESHOLD


def compute_placement(routing: dict) -> dict:
    """Per-layer load-balancing via greedy bin-packing with imbalance threshold.

    Computes an independent expert→GPU assignment for each MoE layer using
    that layer's globally aggregated token counts.  Returns per-layer configs
    so that, for example, layer 5 can put expert 4 on GPU 2 while layer 6
    puts expert 4 on GPU 7 — whatever minimises that layer's load imbalance.

    Skips the update entirely (returns {}) if the optimal placement for every
    layer is already within IMBALANCE_THRESHOLD of balanced, avoiding
    unnecessary NCCL P2P expert transfers.

    Args:
        routing: {layer_id: [capture, ...]}
            capture keys:
              'expert_load'  Tensor[num_experts]  global token counts per expert
              'num_gpus'     int                  EP group size (dp_size * tp_size)

    Returns:
        {"expert_to_gpu": {...}, "layer_configs": {"<layer_id>": {...}, ...}}
        or {} to skip (keep current placement unchanged).
    """
    if not routing:
        return {}

    first_cap = next(iter(routing.values()))[0]
    if 'expert_load' not in first_cap:
        return {}

    num_gpus: int = first_cap['num_gpus']

    # Build per-layer load vectors (CPU list, used only at rebalance time).
    layer_loads: dict[int, list[float]] = {}
    for layer_id, captures in routing.items():
        layer_tensor = torch.stack([cap['expert_load'] for cap in captures]).sum(dim=0)
        layer_loads[layer_id] = layer_tensor.tolist()

    # If every layer is already well-balanced, skip the rebalance entirely.
    if all(_is_balanced(loads, num_gpus) for loads in layer_loads.values()):
        return {}

    # Per-layer greedy bin-packing.
    layer_configs: dict[str, dict[str, int]] = {
        str(layer_id): _greedy_pack(loads, num_gpus)
        for layer_id, loads in layer_loads.items()
    }

    # Global fallback for any layer not present in routing this step (e.g. dense
    # layers that were skipped).  Use the sum across all observed layers.
    all_loads_tensor = torch.tensor(list(layer_loads.values()))  # [L, E]
    global_loads = all_loads_tensor.sum(dim=0).tolist()
    global_assignment = _greedy_pack(global_loads, num_gpus)

    return {
        "expert_to_gpu": global_assignment,
        "layer_configs": layer_configs,
    }
