"""
placement_fns.py — User-supplied expert placement function for combined_launch.py.

This module must be importable by the vLLM worker subprocesses (run from the
merged/vllm root directory), so placement logic lives here rather than in
combined_launch.py.

Register a function via:
    llm.collective_rpc("register_placement_callback",
                       args=("token_tracing.placement_fns", "compute_placement"))

How the aggregation works
-------------------------
Before this callback is called, ``GPUModelRunner._on_routing_step()`` runs
``_aggregate_routing_load()``, which builds a per-expert token-count matrix of
shape [num_layers, num_experts] from the local routing snapshot and all-reduces
it across every EP rank.  The result is attached to the snapshot as:

    routing[layer_id][*]['expert_load']  —  Tensor[num_logical_experts]

Because every rank performs the same all-reduce, each rank receives the
**identical** global load tensor and therefore ``compute_placement()`` produces
the same ``{"expert_to_gpu": {...}}`` on every rank.  This is required: if ranks
produce different placements the NCCL P2P sends/recvs inside ``eplb_step()``
will mismatch and deadlock.
"""

import heapq


def compute_placement(routing: dict) -> dict:
    """Greedy bin-packing load balancer: assign experts to GPUs by token load.

    Accumulates globally aggregated token counts per expert across all layers,
    then assigns experts to GPUs such that the total token load per GPU is
    approximately equal (heaviest experts placed on the least-loaded GPU first).

    Args:
        routing: dict[layer_id, list[capture]]
            Each capture has keys:
              'topk_ids'     — Tensor[T, K] expert indices (local rank's tokens)
              'topk_weights' — Tensor[T, K] corresponding router weights
              'num_tokens'   — int, number of tokens on this rank
              'expert_load'  — Tensor[num_experts], global token counts (all-reduced)

    Returns:
        {"expert_to_gpu": {"<expert_id>": <gpu_id>, ...}}
        Returns {} to skip the update (keeps current placement) if routing is
        empty or if 'expert_load' is missing (e.g. tracking disabled).
    """
    if not routing:
        return {}

    # Sum expert_load across layers to get total tokens per expert globally.
    expert_totals: dict[int, int] = {}
    num_gpus = 0

    for layer_captures in routing.values():
        for cap in layer_captures:
            if 'expert_load' not in cap:
                # Aggregation did not run (tracking may be disabled); skip.
                return {}
            load_tensor = cap['expert_load']
            for expert_id, count in enumerate(load_tensor.tolist()):
                expert_totals[expert_id] = (
                    expert_totals.get(expert_id, 0) + int(count)
                )
            # num_gpus is the EP group size (dp_size × tp_size), provided by
            # _aggregate_routing_load so we don't conflate experts with GPUs.
            if cap.get('num_gpus', 0) > num_gpus:
                num_gpus = cap['num_gpus']

    # Fallback: if num_gpus not supplied (e.g. hand-crafted test data),
    # treat one GPU per expert (correct for Mixtral-8x7B, 8 experts / 8 GPUs).
    if num_gpus == 0:
        num_gpus = len(expert_totals)

    if not expert_totals or num_gpus == 0:
        return {}

    # Greedy bin-packing: assign heaviest experts to least-loaded GPUs.
    # min-heap: (current_load_on_gpu, gpu_id)
    gpu_heap = [(0, g) for g in range(num_gpus)]
    heapq.heapify(gpu_heap)

    assignment: dict[str, int] = {}
    for expert_id, load in sorted(expert_totals.items(), key=lambda x: -x[1]):
        current_load, gpu_id = heapq.heappop(gpu_heap)
        assignment[str(expert_id)] = gpu_id
        heapq.heappush(gpu_heap, (current_load + load, gpu_id))

    return {"expert_to_gpu": assignment}
