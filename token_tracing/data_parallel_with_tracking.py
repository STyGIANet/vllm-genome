# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Data parallel inference with token routing tracking.

Usage:
Single node:
    python new_tracing/data_parallel_with_tracking.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Stygian:
    NCCL_IB_DISABLE=1 VLLM_TRACK_ROUTING=1 \
    python vllm/token_tracing/data_parallel_with_tracking.py \
    --trust-remote-code --enforce-eager --save-routing-csv="routing_data"
"""

import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import numpy as np
import torch


def log_msg(msg: str, dp_rank: int) -> str:
    """Format log message with DP rank and PID like vLLM."""
    pid = os.getpid()
    return f"[DP Rank {dp_rank} PID:{pid}] {msg}"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=8, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help=("Maximum number of sequences to be processed in a single iteration."),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help=("Maximum number of tokens to be processed in a single iteration."),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=("Number of seconds before unresponsive process is killed."),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=("Fraction of GPU memory vLLM is allowed to allocate (0.0, 1.0]."),
    )
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help=("Enable microbatched execution"),
    )
    parser.add_argument(
        "--compilation-config",
        type=int,
        help=("Compilation optimization (O) mode 0-3."),
    )
    parser.add_argument(
        "--quantization",
        type=str,
    )
    parser.add_argument(
        "--disable-expert-parallel",
        dest="enable_expert_parallel",
        action="store_false",
        help="Disable expert parallel (default: enabled).",
    )
    parser.add_argument(
        "--minimal-tracking",
        action="store_true",
        help="Use minimal tracking (only expert counts, not full routing data)",
    )
    parser.add_argument(
        "--save-routing-csv",
        type=str,
        default="",
        help="Path to save routing data CSV (empty to skip saving)",
    )
    parser.add_argument(
        "--num-prompts-per-rank",
        type=int,
        default=3,
        help="Number of prompts to process per DP rank (for testing)",
    )
    parser.set_defaults(enable_expert_parallel=True)
    return parser.parse_args()


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    enable_expert_parallel,
    trust_remote_code,
    max_num_seqs,
    max_model_len,
    compilation_config,
    gpu_memory_utilization,
    enable_dbo,
    quantization,
    minimal_tracking,
    save_routing_csv,
    num_prompts_per_rank,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Load dataset - only load the subset we need
    total_prompts_needed = dp_size * num_prompts_per_rank
    dataset = load_dataset("cais/mmlu", "high_school_us_history", split=f"test[:{total_prompts_needed}]")

    def format_mmlu(example):
        context = "Answer factually without further justification. "
        question = example["question"]
        choices = example["choices"]
        choice_letters = ["A", "B", "C", "D", "E"][:len(choices)]
        choice_text = "\n".join(f"{c}) {t}" for c, t in zip(choice_letters, choices))
        return context + f"question:\n{question}\n\n{choice_text}\n\nAnswer: "

    print(log_msg(f"Dataset size: {len(dataset)}", global_dp_rank))
    prompts = [format_mmlu(example) for example in dataset]

    # Distribute prompts across DP ranks based on num_prompts_per_rank
    start_idx = global_dp_rank * num_prompts_per_rank
    end_idx = start_idx + num_prompts_per_rank
    prompts = prompts[start_idx:end_idx]

    if len(prompts) == 0:
        prompts = ["Placeholder"]
    
    print(log_msg(f"Processing {len(prompts)} prompts (indices {start_idx}-{end_idx-1})", global_dp_rank))

    # Create sampling params - use VERY small max_tokens when tracking to reduce captures
    # Each generated token = 1 forward pass = 1 capture per layer
    # So max_tokens=10 with 3 prompts = ~30 captures instead of ~750
    max_tokens_for_tracking = 10 if os.environ.get("VLLM_TRACK_ROUTING") == "1" else 256
    print(log_msg(f"Using max_tokens={max_tokens_for_tracking} (tracking mode)", global_dp_rank))
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens_for_tracking)

    # Create LLM
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_dbo=enable_dbo,
        quantization=quantization,
        compilation_config=compilation_config,
        seed=0,
    )

    # Clear any existing routing data from previous runs
    print(log_msg("Clearing previous routing data...", global_dp_rank))
    llm.clear_routing_data()
    
    # Generate outputs
    print(log_msg("Starting generation...", global_dp_rank))
    outputs = llm.generate(prompts, sampling_params)
    
    # Print first few outputs
    for i, output in enumerate(outputs):
        if i >= 3:
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(log_msg(f"Prompt: {prompt!r}, Generated: {generated_text!r}", global_dp_rank))

    # ===== RETRIEVE ROUTING DATA =====
    print("\n" + "=" * 80)
    print(log_msg("ROUTING ANALYSIS", global_dp_rank))
    print("=" * 80)
    
    # Get routing data (serialized as bytes to avoid RPC corruption, deserialized to CPU tensors)
    print(log_msg("Retrieving routing data...", global_dp_rank))
    import time
    start_time = time.time()
    routing_data = llm.get_routing_data_distributed()
    elapsed = time.time() - start_time
    print(log_msg(f"Routing data retrieved successfully! (took {elapsed:.2f}s)", global_dp_rank))
    
    if routing_data:
        print(log_msg("Checking routing data structure...", global_dp_rank))
        first_layer = list(routing_data.keys())[0]
        first_capture = routing_data[first_layer][0]
        
        print(log_msg(f"  topk_ids type: {type(first_capture['topk_ids'])}", global_dp_rank))
        print(log_msg(f"  topk_ids shape: {first_capture['topk_ids'].shape}", global_dp_rank))
        print(log_msg(f"  topk_ids device: {first_capture['topk_ids'].device}", global_dp_rank))
        print(log_msg(f"  topk_weights shape: {first_capture['topk_weights'].shape}", global_dp_rank))
        print(log_msg(f"  SUCCESS: Got CPU tensors!", global_dp_rank))
    
    # Inspect what we got
    if routing_data:
        print(log_msg(f"Captured routing data for {len(routing_data)} layers", global_dp_rank))
        
        # First, determine total number of tokens (should be same across all layers)
        num_layers = len(routing_data)
        first_layer_id = sorted(routing_data.keys())[0]
        
        print(log_msg(f"First layer ID: {first_layer_id}", global_dp_rank))
        print(log_msg(f"Number of captures in first layer: {len(routing_data[first_layer_id])}", global_dp_rank))
        
        # Check capture sizes - THIS IS THE PROBLEM!
        print(log_msg(f"Have {len(routing_data[first_layer_id])} captures (= {len(routing_data[first_layer_id])} forward passes)", global_dp_rank))
        print(log_msg(f"Each forward pass goes through all 32 layers, creating 1 capture per layer", global_dp_rank))
        
        print(log_msg(f"Inspecting first capture structure...", global_dp_rank))
        try:
            first_cap = routing_data[first_layer_id][0]
            print(log_msg(f"  First capture type: {type(first_cap)}, keys: {list(first_cap.keys())}", global_dp_rank))
            print(log_msg(f"  First capture num_tokens: {first_cap['num_tokens']}", global_dp_rank))
            print(log_msg(f"  First capture topk_ids type: {type(first_cap['topk_ids'])}", global_dp_rank))
            
            # Handle both tensor and other types
            if isinstance(first_cap['topk_ids'], torch.Tensor):
                print(log_msg(f"  First capture topk_ids dtype: {first_cap['topk_ids'].dtype}", global_dp_rank))
                print(log_msg(f"  First capture topk_ids shape: {first_cap['topk_ids'].shape}", global_dp_rank))
                print(log_msg(f"  First capture topk_ids device: {first_cap['topk_ids'].device}", global_dp_rank))
                nbytes = first_cap['topk_ids'].numel() * first_cap['topk_ids'].element_size()
                print(log_msg(f"  First capture topk_ids nbytes: {nbytes}", global_dp_rank))
            else:
                print(log_msg(f"  WARNING: topk_ids is not a tensor!", global_dp_rank))
        except Exception as e:
            print(log_msg(f"  ERROR inspecting first capture: {e}", global_dp_rank))
            import traceback
            traceback.print_exc()
        
        print(log_msg(f"Checking multiple captures...", global_dp_rank))
        for i in range(min(3, len(routing_data[first_layer_id]))):
            try:
                cap = routing_data[first_layer_id][i]
                if isinstance(cap['topk_ids'], torch.Tensor):
                    nbytes = cap['topk_ids'].numel() * cap['topk_ids'].element_size()
                    print(log_msg(f"  Capture {i}: num_tokens={cap['num_tokens']}, shape={cap['topk_ids'].shape}, nbytes={nbytes}", global_dp_rank))
                else:
                    print(log_msg(f"  Capture {i}: num_tokens={cap['num_tokens']}, type={type(cap['topk_ids'])}", global_dp_rank))
            except Exception as e:
                print(log_msg(f"  ERROR with capture {i}: {e}", global_dp_rank))
        
        # Estimate total data size
        try:
            total_captures = len(routing_data[first_layer_id]) * num_layers
            if isinstance(routing_data[first_layer_id][0]['topk_ids'], torch.Tensor):
                sample_nbytes = (routing_data[first_layer_id][0]['topk_ids'].numel() * routing_data[first_layer_id][0]['topk_ids'].element_size() +
                                routing_data[first_layer_id][0]['topk_weights'].numel() * routing_data[first_layer_id][0]['topk_weights'].element_size())
                estimated_mb = (total_captures * sample_nbytes) / (1024 * 1024)
                print(log_msg(f"Estimated total data size: ~{estimated_mb:.2f} MB ({total_captures} total captures across all layers)", global_dp_rank))
        except Exception as e:
            print(log_msg(f"Could not estimate data size: {e}", global_dp_rank))
        
        # Calculate total tokens by summing across all captures in first layer
        print(log_msg("Calculating total tokens...", global_dp_rank))
        total_tokens = sum(capture['num_tokens'] for capture in routing_data[first_layer_id])
        print(log_msg(f"Total tokens processed: {total_tokens}", global_dp_rank))
        print(log_msg(f"Number of layers: {num_layers}", global_dp_rank))
        
        # Get top_k from first capture
        print(log_msg("Accessing first capture to determine top_k...", global_dp_rank))
        first_capture = routing_data[first_layer_id][0]
        print(log_msg(f"First capture keys: {list(first_capture.keys())}", global_dp_rank))
        print(log_msg(f"First capture topk_ids type: {type(first_capture['topk_ids'])}", global_dp_rank))
        print(log_msg(f"First capture topk_ids shape: {first_capture['topk_ids'].shape}", global_dp_rank))
        
        top_k = first_capture['topk_ids'].shape[1]
        print(log_msg(f"Top-k routing: {top_k} experts per token", global_dp_rank))
        
        # Create tensor to store expert IDs: [num_tokens, num_layers, top_k]
        print(log_msg(f"Allocating tensors for {total_tokens} tokens × {num_layers} layers × {top_k} experts...", global_dp_rank))
        print(log_msg(f"Memory required: ~{total_tokens * num_layers * top_k * 4 / 1024 / 1024:.2f} MB per tensor", global_dp_rank))
        expert_ids = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.int32)
        expert_weights = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.float32)
        print(log_msg("Tensors allocated successfully!", global_dp_rank))
        
        # Fill in data for each layer
        print(log_msg("Consolidating data across layers...", global_dp_rank))
        for layer_idx, layer_id in enumerate(sorted(routing_data.keys())):
            if layer_idx == 0 or layer_idx % 10 == 0:
                print(log_msg(f"  Processing layer {layer_idx}/{num_layers} (ID: {layer_id})...", global_dp_rank))
            
            captures = routing_data[layer_id]
            
            # Debug: Check first capture structure
            if layer_idx == 0:
                print(log_msg(f"    Layer {layer_idx} has {len(captures)} captures", global_dp_rank))
                print(log_msg(f"    First capture num_tokens: {captures[0]['num_tokens']}", global_dp_rank))
                print(log_msg(f"    First capture topk_ids.shape: {captures[0]['topk_ids'].shape}", global_dp_rank))
                print(log_msg(f"    First capture topk_weights.shape: {captures[0]['topk_weights'].shape}", global_dp_rank))
            
            # OPTIMIZED: Use torch.cat instead of numpy concatenate
            if layer_idx == 0:
                print(log_msg(f"    Concatenating {len(captures)} tensors (optimized)...", global_dp_rank))
            
            # Fast concatenation - single allocation, works with torch tensors
            layer_topk_ids = torch.cat([c['topk_ids'] for c in captures], dim=0)
            layer_topk_weights = torch.cat([c['topk_weights'] for c in captures], dim=0)
            
            if layer_idx == 0:
                print(log_msg(f"    Concatenation complete, shape: {layer_topk_ids.shape}", global_dp_rank))
            
            # Direct tensor assignment (already tensors)
            expert_ids[:, layer_idx, :] = layer_topk_ids.to(dtype=torch.int32)
            expert_weights[:, layer_idx, :] = layer_topk_weights.to(dtype=torch.float32)
            
            if layer_idx == 0:
                print(log_msg(f"    Layer {layer_idx} complete!", global_dp_rank))
        
        print(log_msg("All layers consolidated!", global_dp_rank))
        
        # Create metadata tensors
        print(log_msg("Creating metadata tensors...", global_dp_rank))
        token_ids = torch.arange(total_tokens, dtype=torch.int64)
        origin_gpu = torch.full((total_tokens,), global_dp_rank, dtype=torch.int32)
        print(log_msg("Metadata tensors created!", global_dp_rank))
        
        print(log_msg("Data structure:", global_dp_rank))
        print(log_msg(f"  token_ids shape: {token_ids.shape}", global_dp_rank))
        print(log_msg(f"  origin_gpu shape: {origin_gpu.shape}", global_dp_rank))
        print(log_msg(f"  expert_ids shape: {expert_ids.shape} [num_tokens, num_layers, top_k]", global_dp_rank))
        print(log_msg(f"  expert_weights shape: {expert_weights.shape} [num_tokens, num_layers, top_k]", global_dp_rank))
        
        # Print first few tokens
        print(log_msg("First 10 tokens routing (showing expert IDs):", global_dp_rank))
        print(log_msg("Format: Token [origin_gpu] -> Layer_0: (expert1, expert2), Layer_1: (expert1, expert2), ...", global_dp_rank))
        for i in range(min(10, total_tokens)):
            experts_str = ", ".join([
                f"L{layer}: ({expert_ids[i, layer, 0].item()}, {expert_ids[i, layer, 1].item()})"
                for layer in range(min(3, num_layers))  # Show first 3 layers
            ])
            print(log_msg(f"  Token {i} [GPU {origin_gpu[i].item()}] -> {experts_str}...", global_dp_rank))
        
        # Calculate statistics
        print("\n" + "=" * 80)
        print(log_msg("STATISTICS", global_dp_rank))
        print("=" * 80)
        
        for layer_id in range(num_layers):
            layer_experts = expert_ids[:, layer_id, :]  # [num_tokens, top_k]
            unique_experts, counts = torch.unique(layer_experts, return_counts=True)
            
            print(log_msg(f"Layer {layer_id}:", global_dp_rank))
            print(log_msg(f"  Total selections: {layer_experts.numel()}", global_dp_rank))
            print(log_msg(f"  Unique experts used: {len(unique_experts)}", global_dp_rank))
            print(log_msg(f"  Mean expert weight: {expert_weights[:, layer_id, :].mean():.4f}", global_dp_rank))
            print(log_msg(f"  Expert usage counts:", global_dp_rank))
            for expert, count in zip(unique_experts.tolist(), counts.tolist()):
                print(log_msg(f"    Expert {expert}: {count} selections", global_dp_rank))
        
        # Save to file if requested
        if save_routing_csv:
            save_path = f"{save_routing_csv}_rank{global_dp_rank}.pt"
            
            torch.save({
                'token_ids': token_ids,
                'origin_gpu': origin_gpu,
                'expert_ids': expert_ids,
                'expert_weights': expert_weights,
                'num_tokens': total_tokens,
                'num_layers': num_layers,
                'top_k': top_k,
                'dp_rank': global_dp_rank,
            }, save_path)
            
            print(log_msg(f"Saved routing data to {save_path}", global_dp_rank))
            print(log_msg(f"Load with: data = torch.load('{save_path}')", global_dp_rank))
            print(log_msg(f"Access expert IDs with: data['expert_ids'][token_idx, layer_idx, :]", global_dp_rank))
            print(log_msg(f"Shape: {expert_ids.shape} [num_tokens={total_tokens}, num_layers={num_layers}, top_k={top_k}]", global_dp_rank))
        # ===== PRINT FIRST FEW ROWS TO PROVE IN-MEMORY ACCESS =====
        print("\n" + "=" * 80)
        print(log_msg("IN-MEMORY ROUTING DATA (first 5 tokens, layers 0-2)", global_dp_rank))
        print("=" * 80)
        n_show = min(5, total_tokens)
        for t in range(n_show):
            gpu = origin_gpu[t].item()
            row_parts = []
            for l in range(min(3, num_layers)):
                eids = expert_ids[t, l, :].tolist()
                ewts = expert_weights[t, l, :].tolist()
                row_parts.append(f"L{l}:{eids}({', '.join(f'{w:.3f}' for w in ewts)})")
            print(log_msg(f"  Token {t} [GPU {gpu}]  {' | '.join(row_parts)}", global_dp_rank))
        print(log_msg(f"  ... ({total_tokens} total tokens in memory, no file I/O needed)", global_dp_rank))

    else:
        print(log_msg("No routing data captured!", global_dp_rank))
        print(log_msg("Make sure VLLM_TRACK_ROUTING=1 is set when running.", global_dp_rank))
        print(log_msg(f"Variable Status: {os.environ.get('VLLM_TRACK_ROUTING')}", global_dp_rank))
    
    # ==================================

    # Give engines time to pause their processing loops before exiting
    sleep(1)


if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enforce_eager,
                args.enable_expert_parallel,
                args.trust_remote_code,
                args.max_num_seqs,
                args.max_model_len,
                args.compilation_config,
                args.gpu_memory_utilization,
                args.enable_dbo,
                args.quantization,
                args.minimal_tracking,
                args.save_routing_csv,
                args.num_prompts_per_rank,
            ),
        )
        proc.start()
        procs.append(proc)
    
    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within timeout.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
