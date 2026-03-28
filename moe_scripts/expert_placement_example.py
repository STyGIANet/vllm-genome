#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Expert Placement Example for vLLM — DP+EP mode.

Launches one process per DP rank (like the data_parallel example), so that
vLLM runs in true Data Parallel + Expert Parallel mode rather than plain
Tensor Parallel.  Each rank sets the VLLM_DP_* env vars before creating its
LLM instance, and the EPLBConfig / VLLM_EXPERT_CONFIG_PATH are forwarded
into every worker so the StaticPlacementPolicy sees them.

Usage:
    Single node, 8 DP ranks, EP enabled, custom placement:
        VLLM_LOGGING_LEVEL=INFO NCCL_IB_DISABLE=1 \\
        python expert_placement_example.py \\
            --dp-size=8 --trust-remote-code --enforce-eager \\
            --expert-placement-config=mixtral_EP.json

    Single node, 4 DP ranks, 2 GPUs per rank (TP=2 within each rank):
        NCCL_IB_DISABLE=1 \\
        python expert_placement_example.py \\
            --dp-size=4 --tp-size=2 --trust-remote-code --enforce-eager \\
            --expert-placement-config=mixtral_EP.json
"""

import json
import os
from dataclasses import dataclass, field
from multiprocessing import Process
from time import sleep
from typing import Dict, List, Optional

from datasets import concatenate_datasets, load_dataset

from vllm import LLM, SamplingParams  # type: ignore
from vllm.config import EPLBConfig
from vllm.utils.network_utils import get_open_port


@dataclass
class ExpertPlacementConfig:
    """
    Configuration for custom expert placement across GPUs.

    Map of expert_id -> gpu_id, plus optional layer/model metadata.
    """

    expert_to_gpu: Dict[int, int] = field(default_factory=dict)
    num_layers: int = 0
    num_experts_per_layer: int = 0

    @classmethod
    def from_json(cls, filepath: str) -> "ExpertPlacementConfig":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def get_gpu_for_expert(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        expert_key = layer_idx * self.num_experts_per_layer + expert_idx
        return self.expert_to_gpu.get(expert_key, None)

    def to_json(self, filepath: str):
        data = {
            "expert_to_gpu": self.expert_to_gpu,
            "num_layers": self.num_layers,
            "num_experts_per_layer": self.num_experts_per_layer,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Expert Placement Example — DP+EP mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name or path (preferably an MoE model)",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=8,
        help="Data parallel size (number of independent engine instances / DP ranks)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size *per DP rank* (GPUs per rank)",
    )
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
        "--num-prompts-per-rank",
        type=int,
        default=10,
        help="Number of prompts to process per DP rank",
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
        help="Path to expert placement configuration JSON file",
    )
    parser.add_argument(
        "--disable-expert-parallel",
        dest="enable_expert_parallel",
        action="store_false",
        help="Disable expert parallelism (default: enabled)",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Seconds before an unresponsive worker process is killed",
    )
    parser.set_defaults(enable_expert_parallel=True)
    return parser.parse_args()


def log_msg(msg: str, dp_rank: int) -> str:
    pid = os.getpid()
    return f"[DP Rank {dp_rank} PID:{pid}] {msg}"


def load_prompts(dataset_name: str, total_prompts: int) -> List[str]:
    if dataset_name == "simple":
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
        # Repeat/truncate to reach requested total
        repeated = (prompts * ((total_prompts // len(prompts)) + 1))[:total_prompts]
        return repeated

    elif dataset_name == "mmlu":
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

        if not datasets_list:
            print("Warning: Could not load MMLU, using simple prompts")
            return load_prompts("simple", total_prompts)

        dataset = concatenate_datasets(datasets_list)

        def format_mmlu(example):
            context = "Answer the following multiple choice question. "
            question = example["question"]
            choices = example["choices"]
            choice_letters = ["A", "B", "C", "D", "E"][: len(choices)]
            choice_text = "\n".join(
                f"{c}) {t}" for c, t in zip(choice_letters, choices)
            )
            return f"{context}\n\nQuestion: {question}\n\n{choice_text}\n\nAnswer:"

        prompts = [format_mmlu(ex) for ex in dataset]
        return prompts[:total_prompts]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_rank(
    model: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    gpus_per_rank: int,
    enforce_eager: bool,
    enable_expert_parallel: bool,
    trust_remote_code: bool,
    max_num_seqs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    expert_placement_config_path: Optional[str],
    dataset: str,
    num_prompts_per_rank: int,
    output_length: int,
):
    """Worker function executed in each DP rank subprocess."""

    # ── DP env vars must be set before LLM() is created ──────────────────────
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Forward the expert config path so StaticPlacementPolicy finds it
    if expert_placement_config_path:
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = expert_placement_config_path

    print(log_msg(f"Starting (dp_size={dp_size}, gpus_per_rank={gpus_per_rank})", global_dp_rank))

    # ── Build EPLBConfig ──────────────────────────────────────────────────────
    eplb_config = EPLBConfig(
        policy="custom",
        step_interval=1,
        log_balancedness=True,
        log_balancedness_interval=1,
    )

    # ── Load prompts for this rank ────────────────────────────────────────────
    total_prompts = dp_size * num_prompts_per_rank
    all_prompts = load_prompts(dataset, total_prompts)
    start = global_dp_rank * num_prompts_per_rank
    end = start + num_prompts_per_rank
    prompts = all_prompts[start:end]
    if not prompts:
        prompts = ["Placeholder prompt."]
    print(log_msg(f"Processing {len(prompts)} prompts (indices {start}–{end - 1})", global_dp_rank))

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=output_length,
    )

    # ── Create LLM ───────────────────────────────────────────────────────────
    print(log_msg("Creating LLM instance...", global_dp_rank))
    llm = LLM(
        model=model,
        tensor_parallel_size=gpus_per_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        enable_eplb=True,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        eplb_config=eplb_config,
    )

    # ── Print model info (rank 0 only to reduce noise) ────────────────────────
    if global_dp_rank == 0:
        try:
            mc = llm.llm_engine.model_config
            print(log_msg(f"Model: {mc.model}", global_dp_rank))
            if hasattr(mc.hf_config, "num_local_experts"):
                print(
                    log_msg(
                        f"MoE: {mc.hf_config.num_local_experts} experts/layer × "
                        f"{mc.hf_config.num_hidden_layers} layers",
                        global_dp_rank,
                    )
                )
        except Exception as e:
            print(log_msg(f"Could not retrieve model info: {e}", global_dp_rank))

    # ── Generate ──────────────────────────────────────────────────────────────
    print(log_msg("Generating...", global_dp_rank))
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs[:5]):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(log_msg(f"Prompt {i + 1}: {prompt[:80]}...", global_dp_rank))
        print(log_msg(f"  -> {generated[:120]}...", global_dp_rank))

    print(log_msg("Done.", global_dp_rank))
    sleep(1)  # Allow engines to drain before exit


def main():
    args = parse_args()

    # Resolve expert placement config to an absolute path before forking,
    # so child processes can find it regardless of cwd changes.
    expert_config_abs: Optional[str] = None
    if args.expert_placement_config:
        expert_config_abs = os.path.abspath(args.expert_placement_config)
        if not os.path.exists(expert_config_abs):
            raise FileNotFoundError(
                f"Expert placement config not found: {expert_config_abs}"
            )
        print(f"Using expert placement config: {expert_config_abs}")

    # ── DP master coordination ────────────────────────────────────────────────
    if args.node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    dp_size = args.dp_size
    node_size = args.node_size
    node_rank = args.node_rank

    assert dp_size % node_size == 0, "dp_size must be divisible by node_size"
    dp_per_node = dp_size // node_size

    print(f"Launching {dp_per_node} DP rank(s) on this node "
          f"(dp_size={dp_size}, tp_size={args.tp_size}, "
          f"expert_parallel={args.enable_expert_parallel})")

    # ── Spawn one process per local DP rank ───────────────────────────────────
    procs = []
    for local_rank, global_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=run_rank,
            args=(
                args.model,
                dp_size,
                local_rank,
                global_rank,
                dp_master_ip,
                dp_master_port,
                args.tp_size,
                args.enforce_eager,
                args.enable_expert_parallel,
                args.trust_remote_code,
                args.max_num_seqs,
                args.max_model_len,
                args.gpu_memory_utilization,
                expert_config_abs,
                args.dataset,
                args.num_prompts_per_rank,
                args.output_length,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} (timeout after {args.timeout}s)")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)


if __name__ == "__main__":
    main()
