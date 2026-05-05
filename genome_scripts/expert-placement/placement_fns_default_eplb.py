"""
placement_fns_default_eplb.py - Custom callback that reproduces default EPLB.

Registered via:
    llm.collective_rpc("register_placement_callback",
                       args=(path_to_this_file, "compute_placement"))

This uses the same rebalance inputs as the built-in default EPLB policy and
returns the exact physical_to_logical_map to the custom policy path.
"""

import torch
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
from vllm.logger import init_logger


logger = init_logger("vllm.genome.placement_default_eplb")


def compute_placement(routing: dict) -> dict:
    """Return the exact default-EPLB physical_to_logical_map."""
    if not routing:
        return {}

    eplb_meta = routing.get("__eplb__", {})
    global_expert_load = eplb_meta.get("global_expert_load")
    current_map = eplb_meta.get("current_physical_to_logical_map")
    num_layers = eplb_meta.get("num_layers")
    num_physical_experts = eplb_meta.get("num_physical_experts")
    num_gpus = eplb_meta.get("num_gpus")
    num_groups = eplb_meta.get("num_groups")
    num_nodes = eplb_meta.get("num_nodes")

    if (
        global_expert_load is None
        or current_map is None
        or num_layers is None
        or num_physical_experts is None
        or num_gpus is None
        or num_groups is None
        or num_nodes is None
    ):
        logger.warning("Missing __eplb__ metadata; cannot mirror default EPLB.")
        return {}

    weight = global_expert_load.to(torch.float32).cpu()
    current_map_cpu = current_map.to(torch.int32).cpu()
    new_map = DefaultEplbPolicy.rebalance_experts(
        weight=weight,
        num_replicas=int(num_physical_experts),
        num_groups=int(num_groups),
        num_nodes=int(num_nodes),
        num_ranks=int(num_gpus),
        old_global_expert_indices=current_map_cpu,
    )

    expected_shape = (int(num_layers), int(num_physical_experts))
    if tuple(new_map.shape) != expected_shape:
        raise ValueError(
            "Default EPLB callback produced shape "
            f"{tuple(new_map.shape)}, expected {expected_shape}."
        )

    return {"physical_to_logical_map": new_map.to(torch.int32).cpu()}
