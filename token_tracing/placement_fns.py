"""
placement_fns.py — User-supplied expert placement functions for combined_launch.py.

This module must be importable by the vLLM worker subprocesses (run from the
merged/vllm root directory), so placement logic lives here rather than in
combined_launch.py.

Register a function via:
    llm.collective_rpc("register_placement_callback",
                       args=("token_tracing.placement_fns", "compute_placement"))
"""

import heapq
import os


def compute_placement(routing: dict) -> dict:
    """Greedy bin-packing load balancer: assign experts to GPUs by token load.

    Accumulates token counts per expert across all layers in the current step,
    then assigns experts to GPUs such that the total token load per GPU is
    approximately equal (heaviest experts placed on the least-loaded GPU first).

    Args:
        routing: dict[layer_id, list[capture]]
            Each capture has keys:
              'topk_ids'     — Tensor[T, K] expert indices for T tokens, top-K each
              'topk_weights' — Tensor[T, K] corresponding router weights
              'num_tokens'   — int, number of tokens in this capture

    Returns:
        {"expert_to_gpu": {"<expert_id>": <gpu_id>, ...}}
        Returns {} to skip the update (keeps current placement) if routing is empty.
    """
    # Return {} to let the EPLB JSON config (VLLM_EXPERT_CONFIG_PATH) drive
    # placement.  This callback runs independently in every DP-rank worker
    # process with only that rank's local routing data, so each rank would
    # compute a *different* placement from the same step — causing NCCL P2P
    # send/recv mismatches and deadlocks.
    #
    # To use routing data for placement, aggregate it with an NCCL all-reduce
    # across EP ranks first, then compute a single consistent mapping and
    # broadcast it to all workers before returning a non-empty dict here.
    return {}
