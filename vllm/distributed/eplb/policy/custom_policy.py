from .abstract import AbstractEplbPolicy
import torch
import os
import json
from vllm.logger import init_logger

logger = init_logger(__name__)

class StaticPlacementPolicy(AbstractEplbPolicy):
    """
    A policy that ignores expert load and uses a static mapping provided via
    a configuration file.

    Single-step format (same placement every rebalance call)::

        {"expert_to_gpu": {"0": 0, "1": 1, ...}}

    Multi-step / dynamic format (cycles through the list on successive calls)::

        {
            "steps": [
                {"expert_to_gpu": {"0": 0, "1": 1}},
                {"expert_to_gpu": {"0": 1, "1": 0}}
            ]
        }

    Set VLLM_EXPERT_CONFIG_PATH to the JSON file path before launching.
    Expert movement between steps is logged at INFO level.
    """

    _step: int = 0  # incremented each time rebalance_experts is called

    # When set via set_dynamic_config(), this dict is used instead of reading
    # VLLM_EXPERT_CONFIG_PATH.  Format: {"expert_to_gpu": {"0": 0, "1": 1, ...}}
    # Reset to None after it is consumed (one-shot per step).
    _dynamic_config: dict | None = None

    @classmethod
    def set_dynamic_config(cls, config: dict) -> None:
        """Set the placement config for the NEXT rebalance_experts() call.

        ``config`` must follow the single-step JSON format::

            {"expert_to_gpu": {"0": 0, "1": 1, ...}}

        This overrides ``VLLM_EXPERT_CONFIG_PATH`` for one rebalance cycle.
        After the cycle the slot is cleared so subsequent calls fall back to
        the JSON file (or the identity mapping if no file is configured).
        """
        cls._dynamic_config = config

    @classmethod
    def _build_map_from_config(cls, config_path, num_layers, num_physical_experts, num_gpus, step=0):
        # 0. Dynamic override from compute_placement() callback (takes priority).
        if cls._dynamic_config is not None:
            user_config = cls._dynamic_config
            cls._dynamic_config = None  # consume: one-shot per rebalance cycle
            logger.info("[EPLB] Using dynamic config from compute_placement() callback.")
        # 1. Fallback: If no config, return standard sequential mapping
        elif not config_path or not os.path.exists(config_path):
            logger.warning(f"Config path {config_path} not found. Falling back to default.")
            new_map = torch.arange(num_physical_experts, dtype=torch.int32)
            return new_map.unsqueeze(0).expand(num_layers, -1).contiguous()
        else:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

        # Multi-step support: cycle through a list of configs, one per rebalance call.
        if "steps" in user_config:
            steps_list = user_config["steps"]
            effective_step = step % len(steps_list)
            step_config = steps_list[effective_step]
            layer_configs = step_config.get("layer_configs", {})
            global_config = step_config.get("expert_to_gpu", {})
            logger.info(
                f"[EPLB step {step}] Using step-config index {effective_step} "
                f"of {len(steps_list)}: {len(global_config)} global mapping(s)"
            )
        else:
            # Pull configurations
            layer_configs = user_config.get("layer_configs", {})          # Per-layer overrides
            global_config = user_config.get("expert_to_gpu", {})         # Global overrides
        
        slots_per_gpu = num_physical_experts // num_gpus
        full_2d_map = torch.full((num_layers, num_physical_experts), -1, dtype=torch.int32)

        for layer_idx in range(num_layers):
            # Determine mapping for this specific layer: 
            # Priority: Layer-specific > Global > Empty (to be filled by safety net)
            current_layer_data = layer_configs.get(str(layer_idx), global_config)
            
            gpu_slot_counters = {i: 0 for i in range(num_gpus)}
            assigned_experts = set()

            # A. Apply Configured Placements
            for expert_id_str, gpu_id in current_layer_data.items():
                expert_id, gpu_id = int(expert_id_str), int(gpu_id)
                
                if gpu_slot_counters[gpu_id] < slots_per_gpu:
                    physical_idx = (gpu_id * slots_per_gpu) + gpu_slot_counters[gpu_id]
                    full_2d_map[layer_idx, physical_idx] = expert_id
                    gpu_slot_counters[gpu_id] += 1
                    assigned_experts.add(expert_id)
                else:
                    logger.warning(f"Layer {layer_idx}, GPU {gpu_id} full! Skipping Expert {expert_id}.")

            # B. Safety Net: Fill remaining holes with unassigned experts
            # This prevents the engine from crashing due to "missing" experts
            all_logical_experts = set(range(num_physical_experts))
            unassigned = sorted(list(all_logical_experts - assigned_experts))
            
            for p_idx in range(num_physical_experts):
                if full_2d_map[layer_idx, p_idx] == -1 and unassigned:
                    full_2d_map[layer_idx, p_idx] = unassigned.pop(0)

        return full_2d_map.contiguous()
    
    @classmethod
    def _derive_inverse_maps(cls, physical_to_logical_map, num_logical_experts):
        num_layers, num_physical_experts = physical_to_logical_map.shape
        
        # DeepSeek-style models can have many experts, but 10 replicas is plenty for a "max"
        max_replicas = 10 
        
        # These are usually stored on CPU during policy calculation
        logical_to_physical = torch.full(
            (num_layers, num_logical_experts, max_replicas), -1, dtype=torch.int32)
        replica_count = torch.zeros(
            (num_layers, num_logical_experts), dtype=torch.long) # vLLM uses long for counts

        for layer in range(num_layers):
            for physical_idx in range(num_physical_experts):
                logical_id = physical_to_logical_map[layer, physical_idx].item()
                if logical_id != -1:
                    count = replica_count[layer, logical_id].item()
                    if count < max_replicas:
                        logical_to_physical[layer, logical_id, count] = physical_idx
                        replica_count[layer, logical_id] += 1
                    
        return logical_to_physical, replica_count
    
    @classmethod
    def _log_expert_movement(
        cls,
        old_map: torch.Tensor,
        new_map: torch.Tensor,
        num_gpus: int,
    ) -> None:
        """Log which experts changed GPU rank between old and new physical-to-logical maps."""
        num_layers, num_physical_experts = new_map.shape
        slots_per_gpu = num_physical_experts // num_gpus
        total_moves = 0

        for layer_idx in range(num_layers):
            old_layer = old_map[layer_idx].tolist()
            new_layer = new_map[layer_idx].tolist()
            if old_layer == new_layer:
                continue

            old_expert_to_gpu = {
                expert: slot // slots_per_gpu
                for slot, expert in enumerate(old_layer)
                if expert != -1
            }
            new_expert_to_gpu = {
                expert: slot // slots_per_gpu
                for slot, expert in enumerate(new_layer)
                if expert != -1
            }

            moves = [
                (expert, old_expert_to_gpu[expert], new_expert_to_gpu[expert])
                for expert in old_expert_to_gpu
                if expert in new_expert_to_gpu
                and old_expert_to_gpu[expert] != new_expert_to_gpu[expert]
            ]
            if moves:
                total_moves += len(moves)
                logger.info(f"  Layer {layer_idx}: {len(moves)} expert(s) moved:")
                for expert, old_gpu, new_gpu in moves[:8]:
                    logger.info(f"    Expert {expert}: GPU {old_gpu} -> GPU {new_gpu}")
                if len(moves) > 8:
                    logger.info(f"    ... and {len(moves) - 8} more")

        if total_moves == 0:
            logger.info("Expert movement check: no experts moved (placement unchanged).")
        else:
            logger.info(
                f"Expert movement check: {total_moves} total move(s) across all layers."
            )

    @classmethod
    def rebalance_experts(
        cls,
        global_expert_load: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_gpus: int,
        current_physical_to_logical_map: torch.Tensor = None,
    ):
        logger.info(f"--- STATIC CUSTOM PLACEMENT POLICY TRIGGERED (step {cls._step}) ---")
        num_layers, num_logical_experts = global_expert_load.shape

        config_path = os.getenv("VLLM_EXPERT_CONFIG_PATH")

        # Build the forward map (Physical -> Logical)
        new_physical_to_logical_map = cls._build_map_from_config(
            config_path, num_layers, num_replicas, num_gpus, step=cls._step
        )

        # Log which experts moved relative to the previous placement
        if current_physical_to_logical_map is not None:
            cls._log_expert_movement(
                current_physical_to_logical_map.cpu(),
                new_physical_to_logical_map,
                num_gpus,
            )

        cls._step += 1

        # Build the reverse maps (Logical -> Physical)
        new_logical_to_physical_map, new_logical_replica_count = \
            cls._derive_inverse_maps(new_physical_to_logical_map, num_logical_experts)

        logger.info("Expert Mapping per GPU (Sample for Layer 0):")
        for gpu_id in range(num_gpus):
            slots_per_gpu = num_replicas // num_gpus
            start, end = gpu_id * slots_per_gpu, (gpu_id + 1) * slots_per_gpu
            # Change [0, ...] to [layer_idx, ...] to inspect specific layers
            gpu_experts = new_physical_to_logical_map[0, start:end].tolist()
            logger.info(f"  Rank {gpu_id}: Experts {gpu_experts}")

        return (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count
        )