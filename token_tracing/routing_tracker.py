"""
Token Routing Tracker for vLLM MoE Models

This module provides a non-invasive way to track expert routing decisions
during inference by using PyTorch forward hooks.
"""

import torch
from typing import Dict, List, Any
from collections import defaultdict
import threading


class RoutingTracker:
    """Tracks expert routing decisions during MoE inference."""
    
    def __init__(self):
        self.routing_data = defaultdict(list)
        self.lock = threading.Lock()
        self.hooks = []
        
    def clear(self):
        """Clear all collected routing data."""
        with self.lock:
            self.routing_data.clear()
    
    def register_hooks(self, model):
        """
        Register hooks by monkey-patching the select_experts method.
        
        This is necessary because vLLM uses custom operations that bypass
        the standard PyTorch forward hooks.
        
        Args:
            model: The vLLM model instance
        """
        print("=" * 80)
        print("DEBUG: Starting hook registration")
        print("=" * 80)
        
        # Access the actual PyTorch model from vLLM's LLM wrapper
        if hasattr(model, 'llm_engine'):
            print("DEBUG: Found llm_engine")
            llm_engine = model.llm_engine
            print(f"DEBUG: llm_engine type: {type(llm_engine)}")
            
            # Try different paths to access the model
            pytorch_model = None
            
            # V1 API path 1: Try the compatibility layer first (model_executor attribute)
            if hasattr(llm_engine, 'model_executor'):
                print("DEBUG: Found model_executor (V0 compatibility layer in V1)")
                model_executor = llm_engine.model_executor
                print(f"DEBUG: model_executor type: {type(model_executor)}")
                
                if hasattr(model_executor, 'driver_worker'):
                    print("DEBUG: Found driver_worker")
                    if hasattr(model_executor.driver_worker, 'model_runner'):
                        print("DEBUG: Found model_runner")
                        if hasattr(model_executor.driver_worker.model_runner, 'model'):
                            pytorch_model = model_executor.driver_worker.model_runner.model
                            print(f"DEBUG: Found model via driver_worker: {type(pytorch_model)}")
                elif hasattr(model_executor, 'workers'):
                    print("DEBUG: Found workers")
                    if len(model_executor.workers) > 0:
                        pytorch_model = model_executor.workers[0].model_runner.model
                        print(f"DEBUG: Found model via workers: {type(pytorch_model)}")
            
            # V1 API path 2: engine_core -> engine_core -> model_executor
            if pytorch_model is None and hasattr(llm_engine, 'engine_core'):
                print("DEBUG: Found engine_core (V1 API)")
                engine_core = llm_engine.engine_core
                print(f"DEBUG: engine_core type: {type(engine_core)}")
                print(f"DEBUG: engine_core attributes: {[attr for attr in dir(engine_core) if not attr.startswith('_')]}")
                
                # Check if it's a client (multiprocess mode) or direct engine
                engine_core_type_name = type(engine_core).__name__
                if 'Client' in engine_core_type_name or 'MP' in engine_core_type_name:
                    print(f"DEBUG: engine_core is a client ({engine_core_type_name}), trying to access resources")
                    # In multiprocess mode, we need a different approach
                    # Try to get the resources which might have process info
                    if hasattr(engine_core, 'resources'):
                        print("DEBUG: Found resources")
                        resources = engine_core.resources
                        print(f"DEBUG: resources type: {type(resources)}")
                        print(f"DEBUG: resources attributes: {[attr for attr in dir(resources) if not attr.startswith('_')]}")
                        
                        # Try to get the process where the engine is running
                        if hasattr(resources, 'proc'):
                            print("DEBUG: Found proc in resources")
                            proc = resources.proc
                            print(f"DEBUG: proc type: {type(proc)}")
                    
                    # Try core_engine attribute
                    if hasattr(engine_core, 'core_engine'):
                        print("DEBUG: Found core_engine in client")
                        core_engine = engine_core.core_engine
                        print(f"DEBUG: core_engine: {core_engine}")
                
                #The EngineCoreClient wraps the actual EngineCore
                if hasattr(engine_core, 'engine_core'):
                    print("DEBUG: Found nested engine_core")
                    actual_engine_core = engine_core.engine_core
                    print(f"DEBUG: actual_engine_core type: {type(actual_engine_core)}")
                    
                    if hasattr(actual_engine_core, 'model_executor'):
                        print("DEBUG: Found model_executor in actual_engine_core")
                        model_executor = actual_engine_core.model_executor
                        print(f"DEBUG: model_executor type: {type(model_executor)}")
                        
                        if hasattr(model_executor, 'driver_worker'):
                            print("DEBUG: Found driver_worker in model_executor")
                            driver_worker = model_executor.driver_worker
                            print(f"DEBUG: driver_worker type: {type(driver_worker)}")
                            
                            if hasattr(model_runner, 'model'):
                                pytorch_model = model_runner.model
                                print(f"DEBUG: Found model via engine_core path: {type(pytorch_model)}")
                        elif hasattr(model_executor, 'workers') and len(model_executor.workers) > 0:
                            print("DEBUG: Found workers in model_executor")
                            pytorch_model = model_executor.workers[0].model_runner.model
                            print(f"DEBUG: Found model via workers: {type(pytorch_model)}")
                elif hasattr(engine_core, 'model_executor'):
                    # Direct path without nested engine_core
                    print("DEBUG: Found model_executor directly in engine_core")
                    model_executor = engine_core.model_executor
                    if hasattr(model_executor, 'driver_worker'):
                        if hasattr(model_executor.driver_worker, 'model_runner'):
                            if hasattr(model_executor.driver_worker.model_runner, 'model'):
                                pytorch_model = model_executor.driver_worker.model_runner.model
                                print(f"DEBUG: Found model via direct engine_core path: {type(pytorch_model)}")
            
            if pytorch_model is None:
                raise RuntimeError("Cannot access model from LLM engine. Please check vLLM version compatibility.")
            
            # Print all module names to debug
            print("\nDEBUG: Searching for MoE layers...")
            all_module_types = set()
            for name, module in pytorch_model.named_modules():
                module_type = module.__class__.__name__
                all_module_types.add(module_type)
                if 'moe' in module_type.lower() or 'expert' in module_type.lower():
                    print(f"  Found potential MoE module: {name} ({module_type})")
            
            print(f"\nDEBUG: Unique module types in model ({len(all_module_types)} total)")
            
            # Monkey-patch select_experts method on all FusedMoE layers
            layer_idx = 0
            for name, module in pytorch_model.named_modules():
                # Target FusedMoE layers
                if module.__class__.__name__ == 'FusedMoE':
                    print(f"DEBUG: Monkey-patching select_experts on layer {layer_idx}: {name}")
                    # Store original method
                    original_select_experts = module.select_experts
                    # Create wrapper that captures routing data
                    module.select_experts = self._create_select_experts_wrapper(
                        original_select_experts, layer_idx, name, module
                    )
                    self.hooks.append((module, original_select_experts))
                    layer_idx += 1
                    
            print(f"\nRegistered routing hooks on {layer_idx} MoE layers")
            print("=" * 80)
            return layer_idx
        else:
            raise RuntimeError("Model does not have llm_engine attribute")
    
    
    def _create_select_experts_wrapper(self, original_method, layer_idx: int, layer_name: str, module):
        """Create a wrapper around select_experts that captures routing data."""
        print(f"DEBUG: Creating select_experts wrapper for layer {layer_idx} ({layer_name})")
        
        def wrapped_select_experts(hidden_states, router_logits):
            """
            Wrapper around select_experts that captures routing information.
            """
            # Call original method
            result = original_method(hidden_states, router_logits)
            
            # Capture the routing data
            try:
                topk_weights, topk_ids = result[0], result[1]
                
                print(f"DEBUG: Captured routing for layer {layer_idx} - tokens: {hidden_states.shape[0]}, topk_ids shape: {topk_ids.shape}")
                
                routing_info = {
                    'layer_idx': layer_idx,
                    'layer_name': layer_name,
                    'num_tokens': hidden_states.shape[0],
                    'topk_ids': topk_ids.cpu().numpy(),  # [num_tokens, top_k]
                    'topk_weights': topk_weights.cpu().numpy(),  # [num_tokens, top_k]
                    'router_logits': router_logits.cpu().numpy(),  # [num_tokens, num_experts]
                }
                
                with self.lock:
                    self.routing_data[layer_idx].append(routing_info)
                    print(f"DEBUG: Stored routing info for layer {layer_idx}, total captures: {len(self.routing_data[layer_idx])}")
                    
            except Exception as e:
                print(f"ERROR capturing routing data for layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            return result
        
        return wrapped_select_experts
    
    def _create_hook_fn(self, layer_idx: int, layer_name: str):
        """Create a hook function for a specific layer (DEPRECATED - using monkey-patching instead)."""
        print(f"DEBUG: Creating hook function for layer {layer_idx} ({layer_name})")
        
        def hook_fn(module, input_tuple, output):
            """
            Forward hook to capture routing information.
            
            The FusedMoE forward method processes:
            - input: (hidden_states, router_logits)
            - output: final_hidden_states or (final_hidden_states, shared_expert_output)
            
            We want to capture the routing decisions (topk_ids, topk_weights)
            which are computed in select_experts method.
            """
            print(f"DEBUG: Hook triggered for layer {layer_idx}")
            try:
                # Extract inputs
                print(f"DEBUG: Input tuple length: {len(input_tuple)}")
                hidden_states = input_tuple[0]
                router_logits = input_tuple[1]
                print(f"DEBUG: hidden_states shape: {hidden_states.shape}")
                print(f"DEBUG: router_logits shape: {router_logits.shape}")
                
                # Capture routing info by calling select_experts
                # (This is what the forward method does internally)
                with torch.no_grad():
                    print(f"DEBUG: Calling select_experts...")
                    topk_weights, topk_ids, _ = module.select_experts(
                        hidden_states, router_logits
                    )
                    print(f"DEBUG: topk_ids shape: {topk_ids.shape}")
                    print(f"DEBUG: topk_weights shape: {topk_weights.shape}")
                    
                    routing_info = {
                        'layer_idx': layer_idx,
                        'layer_name': layer_name,
                        'num_tokens': hidden_states.shape[0],
                        'topk_ids': topk_ids.cpu().numpy(),  # [num_tokens, top_k]
                        'topk_weights': topk_weights.cpu().numpy(),  # [num_tokens, top_k]
                        'router_logits': router_logits.cpu().numpy(),  # [num_tokens, num_experts]
                    }
                    
                    with self.lock:
                        self.routing_data[layer_idx].append(routing_info)
                        print(f"DEBUG: Stored routing info for layer {layer_idx}, now have {len(self.routing_data[layer_idx])} captures")
                        
            except Exception as e:
                print(f"ERROR in routing hook for layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        return hook_fn
    
    def remove_hooks(self):
        """Remove all registered hooks and restore original methods."""
        # Restore original select_experts methods
        for module, original_method in self.hooks:
            module.select_experts = original_method
        self.hooks.clear()
        print("Removed all routing hooks and restored original methods")
    
    def get_routing_data(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all collected routing data.
        
        Returns:
            Dictionary mapping layer_idx to list of routing information dicts
        """
        with self.lock:
            return dict(self.routing_data)
    
    def to_dataframe(self):
        """
        Convert routing data to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame with columns:
            - layer_idx
            - batch_idx (index within layer's captures)
            - token_idx
            - expert_id (selected expert)
            - expert_weight
            - expert_rank (0 for top-1, 1 for top-2, etc.)
        """
        import pandas as pd
        import numpy as np
        
        records = []
        
        for layer_idx, captures in self.routing_data.items():
            for batch_idx, capture in enumerate(captures):
                num_tokens = capture['num_tokens']
                topk_ids = capture['topk_ids']  # [num_tokens, top_k]
                topk_weights = capture['topk_weights']  # [num_tokens, top_k]
                
                # Flatten the data
                for token_idx in range(num_tokens):
                    for expert_rank in range(topk_ids.shape[1]):
                        records.append({
                            'layer_idx': layer_idx,
                            'batch_idx': batch_idx,
                            'token_idx': token_idx,
                            'expert_id': topk_ids[token_idx, expert_rank],
                            'expert_weight': topk_weights[token_idx, expert_rank],
                            'expert_rank': expert_rank,
                        })
        
        return pd.DataFrame(records)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about expert routing.
        
        Returns:
            Dictionary with routing statistics
        """
        import numpy as np
        
        stats = {
            'num_layers': len(self.routing_data),
            'total_captures': sum(len(captures) for captures in self.routing_data.values()),
            'per_layer_stats': {}
        }
        
        for layer_idx, captures in self.routing_data.items():
            all_expert_ids = []
            all_weights = []
            
            for capture in captures:
                all_expert_ids.extend(capture['topk_ids'].flatten())
                all_weights.extend(capture['topk_weights'].flatten())
            
            all_expert_ids = np.array(all_expert_ids)
            all_weights = np.array(all_weights)
            
            # Compute expert load balance
            unique_experts, expert_counts = np.unique(all_expert_ids, return_counts=True)
            
            stats['per_layer_stats'][layer_idx] = {
                'total_token_expert_pairs': len(all_expert_ids),
                'unique_experts_used': len(unique_experts),
                'expert_usage_counts': dict(zip(unique_experts.tolist(), expert_counts.tolist())),
                'mean_expert_weight': float(np.mean(all_weights)),
                'std_expert_weight': float(np.std(all_weights)),
                'expert_load_balance_coefficient': float(np.std(expert_counts) / np.mean(expert_counts)) if len(expert_counts) > 0 else 0.0,
            }
        
        return stats
    
    def save_to_csv(self, filepath: str):
        """Save routing data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved routing data to {filepath}")


class MinimalRoutingTracker:
    """
    Minimal tracker that only stores aggregated statistics to reduce memory usage.
    Useful for long inference runs.
    """
    
    def __init__(self):
        self.expert_counts = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks on all MoE layers."""
        if hasattr(model, 'llm_engine'):
            model_executor = model.llm_engine.model_executor
            
            if hasattr(model_executor, 'driver_worker'):
                pytorch_model = model_executor.driver_worker.model_runner.model
            elif hasattr(model_executor, 'workers'):
                pytorch_model = model_executor.workers[0].model_runner.model
            else:
                raise RuntimeError("Cannot access model from LLM engine")
            
            layer_idx = 0
            for name, module in pytorch_model.named_modules():
                if module.__class__.__name__ == 'FusedMoE':
                    hook = module.register_forward_hook(
                        self._create_hook_fn(layer_idx, name)
                    )
                    self.hooks.append(hook)
                    layer_idx += 1
                    
            print(f"Registered minimal routing hooks on {layer_idx} MoE layers")
            return layer_idx
        else:
            raise RuntimeError("Model does not have llm_engine attribute")
    
    def _create_hook_fn(self, layer_idx: int, layer_name: str):
        """Create a hook function that only tracks expert usage counts."""
        def hook_fn(module, input_tuple, output):
            try:
                hidden_states = input_tuple[0]
                router_logits = input_tuple[1]
                
                with torch.no_grad():
                    topk_weights, topk_ids, _ = module.select_experts(
                        hidden_states, router_logits
                    )
                    
                    # Only track expert usage counts
                    expert_ids = topk_ids.cpu().numpy().flatten()
                    unique, counts = torch.unique(topk_ids, return_counts=True)
                    
                    with self.lock:
                        for expert_id, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                            self.expert_counts[layer_idx][int(expert_id)] += int(count)
                            
            except Exception as e:
                print(f"Error in minimal routing hook for layer {layer_idx}: {e}")
        
        return hook_fn
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_expert_counts(self) -> Dict[int, Dict[int, int]]:
        """Get expert usage counts per layer."""
        with self.lock:
            return dict(self.expert_counts)
    
    def print_summary(self):
        """Print a summary of expert usage."""
        counts = self.get_expert_counts()
        for layer_idx in sorted(counts.keys()):
            print(f"\nLayer {layer_idx}:")
            expert_counts = counts[layer_idx]
            for expert_id in sorted(expert_counts.keys()):
                print(f"  Expert {expert_id}: {expert_counts[expert_id]} tokens")
