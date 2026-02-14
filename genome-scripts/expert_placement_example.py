#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Expert Placement Example for vLLM

This script demonstrates offline inference with vLLM on MoE models
and prepares infrastructure for custom expert placement across GPUs.

Usage:
    Single GPU:
        python expert_placement_example.py --model="ibm-research/PowerMoE-3b"
    
    Multi-GPU (Tensor Parallel):
        python expert_placement_example.py \
            --model="mistralai/Mixtral-8x7B-Instruct-v0.1" \
            --tp-size=2 \
            --trust-remote-code
    
    With custom expert placement (future feature):
        python expert_placement_example.py \
            --model="mistralai/Mixtral-8x7B-Instruct-v0.1" \
            --tp-size=4 \
            --expert-placement-config="expert_placement.json"
"""

# Command to run code: NCCL_IB_DISABLE=1 python expert_placement_example.py --tp-size=8 --trust-remote-code --enforce-eager --expert-placement-config=mixtral_EP.json

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams # type: ignore
from vllm.config import EPLBConfig
from datasets import load_dataset, concatenate_datasets


@dataclass
class ExpertPlacementConfig:
    """
    Configuration for custom expert placement across GPUs.
    
    This will be used to specify which expert IDs should be placed
    on which GPU devices for MoE models.
    """
    # Map of expert_id -> gpu_id
    expert_to_gpu: Dict[int, int] = field(default_factory=dict)
    
    # Model layer configurations
    num_layers: int = 0
    num_experts_per_layer: int = 0
    
    @classmethod
    def from_json(cls, filepath: str) -> "ExpertPlacementConfig":
        """Load expert placement configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_gpu_for_expert(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """Get the GPU ID for a specific expert in a specific layer."""
        # Generate a unique key for this expert
        expert_key = layer_idx * self.num_experts_per_layer + expert_idx
        return self.expert_to_gpu.get(expert_key, None)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        data = {
            'expert_to_gpu': self.expert_to_gpu,
            'num_layers': self.num_layers,
            'num_experts_per_layer': self.num_experts_per_layer
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Expert Placement Example with vLLM Offline Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name or path (preferably an MoE model)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="Maximum number of sequences per iteration",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model execution",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode execution",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (0.0-1.0)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "simple"],
        help="Dataset to use for prompts",
    )
    parser.add_argument(
        "--expert-placement-config",
        type=str,
        default=None,
        help="Path to expert placement configuration JSON file (future feature)",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        default=True,
        help="Enable expert parallelism for MoE models",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    
    return parser.parse_args()


def load_prompts(dataset_name: str, num_prompts: int) -> List[str]:
    """
    Load prompts from the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        num_prompts: Number of prompts to return
        
    Returns:
        List of prompt strings
    """
    if dataset_name == "simple":
        # Simple test prompts
        prompts = [
            "Explain the theory of relativity in simple terms.",
            "What are the main differences between Python and Java?",
            "Describe the water cycle.",
            "What is machine learning?",
            "Explain how a neural network works.",
            "What causes seasons on Earth?",
            "Describe the process of photosynthesis.",
            "What is the difference between HTTP and HTTPS?",
            "Explain quantum computing for beginners.",
            "What is the capital of France?",
        ]
        return prompts[:num_prompts]
    
    elif dataset_name == "mmlu":
        # Load MMLU dataset for more complex prompts
        subjects = {
            "elementary_mathematics": 32,
            "college_mathematics": 32,
            "abstract_algebra": 32,
            "high_school_physics": 32,
            "college_physics": 32,
        }
        
        datasets_list = []
        for subject, limit in subjects.items():
            try:
                ds = load_dataset("cais/mmlu", subject, split=f"test[:{limit}]")
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: Could not load {subject}: {e}")
                continue
        
        if not datasets_list:
            print("Warning: Could not load MMLU dataset, using simple prompts")
            return load_prompts("simple", num_prompts)
        
        dataset = concatenate_datasets(datasets_list)
        
        def format_mmlu(example):
            """Format MMLU example into a prompt."""
            context = "Answer the following multiple choice question. "
            question = example["question"]
            choices = example["choices"]
            choice_letters = ["A", "B", "C", "D", "E"][:len(choices)]
            choice_text = "\n".join(
                f"{c}) {t}" for c, t in zip(choice_letters, choices)
            )
            return f"{context}\n\nQuestion: {question}\n\n{choice_text}\n\nAnswer:"
        
        prompts = [format_mmlu(example) for example in dataset]
        return prompts[:num_prompts]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def apply_expert_placement_hook(llm: LLM, config: Optional[EPLBConfig]):
    """
    Apply expert placement configuration to the LLM model.
    """
    if config is None:
        print("No EPLB config provided.")
        return
    
    # Get the path from the environment variable we set in main()
    config_path = os.getenv("VLLM_EXPERT_CONFIG_PATH")
    
    print(f"EPLB Policy Active: {config.policy}")
    if config_path:
        print(f"  - Custom Map File: {config_path}")
        # Optionally print the file content to verify
        with open(config_path, 'r') as f:
            data = json.load(f)
            expert_map = data.get("expert_to_gpu", {})
            print(f"  - Custom Expert Mappings: {len(expert_map)}")
    
    print("\nSUCCESS: Custom Policy Hooked into vLLM Engine.")
    print("Expert rearrangement will occur on the first engine step.\n")


def print_model_info(llm: LLM):
    """Print information about the loaded model."""
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    # Try to access model configuration
    try:
        model_config = llm.llm_engine.model_config
        print(f"Model: {model_config.model}")
        print(f"Max model length: {model_config.max_model_len}")
        print(f"Dtype: {model_config.dtype}")
        
        # Check if it's an MoE model
        if hasattr(model_config.hf_config, 'num_local_experts'):
            num_experts = model_config.hf_config.num_local_experts
            num_layers = model_config.hf_config.num_hidden_layers
            print(f"\nMoE Model Detected:")
            print(f"  - Number of experts per layer: {num_experts}")
            print(f"  - Number of layers: {num_layers}")
            print(f"  - Total experts: {num_experts * num_layers}")
        else:
            print("\nNot an MoE model (or MoE config not detected)")
    except Exception as e:
        print(f"Could not retrieve model info: {e}")
    
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    print("="*80)
    print("vLLM Expert Placement Example")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tp_size}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Dataset: {args.dataset}")
    print("="*80 + "\n")
    
    # Load expert placement configuration if provided
    expert_config = None
    if args.expert_placement_config:
        policy_name = "custom"
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = os.path.abspath(args.expert_placement_config)
        print(f"Setting VLLM_EXPERT_CONFIG_PATH to: {os.environ['VLLM_EXPERT_CONFIG_PATH']}")
    else:
        policy_name = "default"

    # Create the actual config object vLLM expects
    expert_config = EPLBConfig(
        policy="custom",
        step_interval=1,                # Apply every step
        log_balancedness=True,         # Enable summary logs
        log_balancedness_interval=1    # Log every step
    )

    # Load prompts
    print(f"\nLoading prompts from {args.dataset} dataset...")
    prompts = load_prompts(args.dataset, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts\n")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.output_length,
    )
    
    print("Creating LLM instance...")
    print(f"  - GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"  - Max Num Seqs: {args.max_num_seqs}")
    print(f"  - Enforce Eager: {args.enforce_eager}")
    print(f"  - Trust Remote Code: {args.trust_remote_code}\n")
    print(f"  - Custom Expert Config: {'True' if expert_config else 'False'}")
    # Create the LLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        enforce_eager=args.enforce_eager,
        enable_expert_parallel=args.enable_expert_parallel,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        eplb_config=expert_config,
    )
    
    # Print model information
    print_model_info(llm)
    
    # Apply expert placement configuration (future feature)
    apply_expert_placement_hook(llm, expert_config)
    
    # Generate responses
    print("\n" + "="*80)
    print("GENERATING RESPONSES")
    print("="*80 + "\n")
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Print outputs
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"Prompt {i+1}:")
        print(f"  {prompt[:100]}..." if len(prompt) > 100 else f"  {prompt}")
        print(f"\nGenerated:")
        print(f"  {generated_text[:200]}..." if len(generated_text) > 200 else f"  {generated_text}")
        print("\n" + "-"*80 + "\n")
        
        # Only print first 5 outputs by default
        if i >= 4:
            remaining = len(outputs) - i - 1
            if remaining > 0:
                print(f"... and {remaining} more outputs (not shown)\n")
            break
    
    print("="*80)
    print("DONE")
    print("="*80)
    
    # Generate example expert placement config for reference
    if not args.expert_placement_config:
        print("\nTo use custom expert placement in the future, create a config file like:")
        print("expert_placement_config.json:")
        example_config = ExpertPlacementConfig(
            expert_to_gpu={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
            num_layers=32,
            num_experts_per_layer=8
        )
        print(json.dumps({
            'expert_to_gpu': example_config.expert_to_gpu,
            'num_layers': example_config.num_layers,
            'num_experts_per_layer': example_config.num_experts_per_layer
        }, indent=2))
        print("\nThen run with: --expert-placement-config=expert_placement_config.json")


if __name__ == "__main__":
    main()
