#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
combined_launch.py — DP+EP inference with BOTH token routing tracking AND
live expert redistribution via a user-supplied compute_placement() callback.

This is the primary integration script for the moe-merged branch.  It combines:
  • data_parallel_with_tracking.py  — per-step routing capture
  • expert_placement_example.py     — custom expert placement via StaticPlacementPolicy

Usage (8×L4, Mixtral, placement + tracking):
    source /home/nirmal/moe/merged/vllm/.venv/bin/activate
    cd /home/nirmal/moe/merged/vllm

    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    python token_tracing/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --expert-placement-config token_tracing/mixtral_EP_test.json

Tracking only (no placement):
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 \\
    python token_tracing/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager

Expert placement only (no tracking):
    NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    python token_tracing/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --expert-placement-config token_tracing/mixtral_EP_test.json

Environment variables
---------------------
VLLM_TRACK_ROUTING=1         Enable per-step routing capture.
VLLM_EXPERT_CONFIG_PATH      Overrides --expert-placement-config at the worker level.
NCCL_IB_DISABLE=1            Disable InfiniBand (single-node only).
VLLM_LOGGING_LEVEL=INFO      Show EPLB placement logs.
"""

import argparse
import os
from multiprocessing import Process
from time import sleep
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# User callback: replace this with your own expert placement logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_placement(routing: dict) -> dict:
    """Load-balancing placement: reassign experts so each GPU gets equal token load.

    Algorithm (greedy bin-packing):
    1. Count how many tokens each expert received this step (sum across all layers).
    2. Sort experts by load descending.
    3. Use a min-heap over GPUs keyed by current assigned load.
    4. Assign each expert (highest-load first) to the currently least-loaded GPU.

    This is called after every forward pass (prefill + each decode step).
    The returned mapping is fed into StaticPlacementPolicy.set_dynamic_config()
    and takes effect on the NEXT eplb_step().

    Args:
        routing: dict[layer_id, list[capture]]
            Each capture: {'topk_ids': Tensor[T, K], 'topk_weights': Tensor[T, K],
                           'num_tokens': int}

    Returns:
        {"expert_to_gpu": {"<expert_id>": <gpu_id>, ...}}
        Returns {} (no update) if routing is empty or num_gpus cannot be inferred.
    """
    if not routing:
        return {}

    import heapq

    # ── Determine num_experts and num_gpus from DP size env var ──────────────
    dp_size = int(os.environ.get("VLLM_DP_SIZE", "1"))
    tp_size = int(os.environ.get("_COMBINED_TP_SIZE", "1"))
    num_gpus = dp_size * tp_size  # == ep_size

    # Accumulate token counts per expert across all layers and all captures
    expert_load: dict[int, int] = {}
    for layer_id, captures in routing.items():
        for cap in captures:
            topk_ids = cap["topk_ids"]
            # topk_ids may be on GPU — move to CPU for counting
            ids_cpu = topk_ids.cpu().reshape(-1).tolist()
            for eid in ids_cpu:
                expert_load[eid] = expert_load.get(eid, 0) + 1

    if not expert_load:
        return {}

    num_experts = max(expert_load.keys()) + 1

    # Pad any unseen experts with zero load
    for eid in range(num_experts):
        if eid not in expert_load:
            expert_load[eid] = 0

    # ── Greedy bin-packing: sort experts by load desc, assign to min-load GPU ─
    # heap entries: (gpu_current_load, gpu_id)
    gpu_heap = [(0, g) for g in range(num_gpus)]
    heapq.heapify(gpu_heap)

    expert_to_gpu: dict[str, int] = {}
    for expert_id in sorted(expert_load, key=lambda e: -expert_load[e]):
        gpu_load, gpu_id = heapq.heappop(gpu_heap)
        expert_to_gpu[str(expert_id)] = gpu_id
        heapq.heappush(gpu_heap, (gpu_load + expert_load[expert_id], gpu_id))

    # Log load distribution at INFO level (rank 0 only to reduce noise)
    if os.environ.get("VLLM_DP_RANK", "0") == "0":
        gpu_loads = {g: 0 for g in range(num_gpus)}
        for eid_str, gid in expert_to_gpu.items():
            gpu_loads[gid] += expert_load[int(eid_str)]
        load_str = "  ".join(f"GPU{g}:{gpu_loads[g]}" for g in range(num_gpus))
        import logging
        logging.getLogger(__name__).info(
            "[compute_placement] load after rebalance: %s", load_str
        )

    return {"expert_to_gpu": expert_to_gpu}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined token routing tracking + expert placement (DP+EP mode)"
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model name or HuggingFace path",
    )
    # Parallelism
    parser.add_argument("--dp-size", type=int, default=8, dest="dp_size",
                        help="Number of DP ranks (one process + engine per rank)")
    parser.add_argument("--tp-size", type=int, default=1, dest="tp_size",
                        help="Tensor parallel GPUs per DP rank")
    parser.add_argument("--node-size", type=int, default=1, dest="node_size")
    parser.add_argument("--node-rank", type=int, default=0, dest="node_rank")
    parser.add_argument("--master-addr", type=str, default="", dest="master_addr")
    parser.add_argument("--master-port", type=int, default=0, dest="master_port")
    # Engine knobs
    parser.add_argument("--max-num-seqs", type=int, default=128, dest="max_num_seqs")
    parser.add_argument("--max-model-len", type=int, default=2048, dest="max_model_len")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        dest="gpu_memory_utilization")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    parser.add_argument("--disable-expert-parallel", dest="enable_expert_parallel",
                        action="store_false")
    parser.set_defaults(enable_expert_parallel=True)
    # Dataset / generation
    parser.add_argument("--dataset", type=str, default="simple",
                        choices=["simple", "mmlu"],
                        help="Prompt source (simple=hardcoded, mmlu=HuggingFace dataset)")
    parser.add_argument("--num-prompts-per-rank", type=int, default=3,
                        dest="num_prompts_per_rank",
                        help="Prompts per DP rank (keep small when tracking)")
    parser.add_argument("--output-length", type=int, default=10, dest="output_length",
                        help="Max tokens to generate per prompt")
    # Expert placement
    parser.add_argument("--expert-placement-config", type=str, default=None,
                        dest="expert_placement_config",
                        help="Path to placement JSON (enables StaticPlacementPolicy)")
    # Routing output
    parser.add_argument("--save-routing-pt", type=str, default="",
                        dest="save_routing_pt",
                        help="If set, save per-rank routing tensors to <path>_rank<N>.pt")
    # Process management
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds before an unresponsive worker is killed")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(dataset_name: str, total_prompts: int):
    if dataset_name == "simple":
        base = [
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
        return (base * ((total_prompts // len(base)) + 1))[:total_prompts]

    elif dataset_name == "mmlu":
        from datasets import concatenate_datasets, load_dataset

        subjects = {
            "elementary_mathematics": 32,
            "high_school_us_history": 32,
            "college_physics": 32,
        }
        datasets_list = []
        for subject, limit in subjects.items():
            try:
                ds = load_dataset("cais/mmlu", subject, split=f"test[:{limit}]")
                datasets_list.append(ds)
            except Exception as e:
                print(f"Warning: could not load {subject}: {e}")

        if not datasets_list:
            print("Warning: MMLU unavailable, falling back to simple prompts.")
            return load_prompts("simple", total_prompts)

        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets_list)

        def fmt(ex):
            q, choices = ex["question"], ex["choices"]
            letters = ["A", "B", "C", "D"][:len(choices)]
            opts = "\n".join(f"{l}) {t}" for l, t in zip(letters, choices))
            return f"Question: {q}\n{opts}\nAnswer:"

        return [fmt(ex) for ex in dataset][:total_prompts]

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-rank worker
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, rank: int) -> str:
    return f"[DP Rank {rank} PID:{os.getpid()}] {msg}"


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
    save_routing_pt: str,
):
    # ── Set DP env vars (must happen before LLM() is instantiated) ────────────
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["VLLM_TRACK_ROUTING"] = "1"
    os.environ["_COMBINED_TP_SIZE"] = str(gpus_per_rank)  # used by compute_placement

    use_placement = expert_placement_config_path is not None
    if use_placement:
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = expert_placement_config_path

    # ── Imports (deferred so env vars are set first) ──────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.config import EPLBConfig
    from vllm.utils.network_utils import get_open_port

    # ── LLM config ────────────────────────────────────────────────────────────
    eplb_config = EPLBConfig(
        policy="custom" if use_placement else "default",
        step_interval=1,
        log_balancedness=use_placement,
        log_balancedness_interval=1,
    )

    print(log(
        f"Starting — dp_size={dp_size}, gpus_per_rank={gpus_per_rank}, "
        f"tracking=True, placement={use_placement}",
        global_dp_rank,
    ))

    llm = LLM(
        model=model,
        tensor_parallel_size=gpus_per_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        enable_eplb=use_placement,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        eplb_config=eplb_config,
    )

    # ── Wire compute_placement callback into the model runner ─────────────────
    if use_placement:
        try:
            engine_core = llm.llm_engine.engine_core  # type: ignore[attr-defined]
            mr = engine_core.model_runner            # type: ignore[attr-defined]
            mr._compute_placement_callback = compute_placement
            print(log("compute_placement callback attached to model runner.", global_dp_rank))
        except AttributeError as e:
            print(log(f"WARNING: could not attach callback ({e})", global_dp_rank))

    # ── Load prompts for this rank ────────────────────────────────────────────
    total_prompts = dp_size * num_prompts_per_rank
    all_prompts = load_prompts(dataset, total_prompts)
    start = global_dp_rank * num_prompts_per_rank
    prompts = all_prompts[start: start + num_prompts_per_rank] or ["Placeholder."]
    print(log(f"Processing {len(prompts)} prompts (indices {start}–{start + len(prompts) - 1})", global_dp_rank))

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length)

    # ── Reset step counter so step_idx=0 maps to prefill ─────────────────────
    llm.reset_step_counter()

    # ── Generate ──────────────────────────────────────────────────────────────
    print(log("Generating...", global_dp_rank))
    outputs = llm.generate(prompts, sampling_params)

    for i, out in enumerate(outputs[:3]):
        print(log(f"  Prompt: {out.prompt[:80]!r}", global_dp_rank))
        print(log(f"  Output: {out.outputs[0].text[:120]!r}", global_dp_rank))

    # ─────────────────────────────────────────────────────────────────────────
    # Routing analysis
    # ─────────────────────────────────────────────────────────────────────────
    print(log("Draining step snapshots...", global_dp_rank))
    snapshots = llm.drain_step_snapshots()

    if not snapshots:
        print(log("No routing snapshots captured — is VLLM_TRACK_ROUTING=1?", global_dp_rank))
    else:
        print(log(f"Captured {len(snapshots)} step snapshot(s):", global_dp_rank))
        for snap in snapshots:
            layers = sorted(snap["routing"].keys())
            if not layers:
                continue
            first_captures = snap["routing"][layers[0]]
            total_tokens = sum(c["num_tokens"] for c in first_captures)
            print(log(
                f"  step_idx={snap['step_idx']}  "
                f"layers={len(layers)}  "
                f"tokens={total_tokens}",
                global_dp_rank,
            ))

        # ── Flatten all captures into a single [tokens, layers, top_k] tensor ──
        # Combine prefill + all decode steps into one view.
        layer_ids = sorted(snapshots[0]["routing"].keys())
        num_layers = len(layer_ids)

        # Build per-layer lists of all token captures across every step
        per_layer_ids: dict[int, list] = {lid: [] for lid in layer_ids}
        per_layer_wts: dict[int, list] = {lid: [] for lid in layer_ids}

        for snap in snapshots:
            for lid in layer_ids:
                for cap in snap["routing"].get(lid, []):
                    per_layer_ids[lid].append(cap["topk_ids"])
                    per_layer_wts[lid].append(cap["topk_weights"])

        # Stack into tensors
        first_lid = layer_ids[0]
        all_ids = torch.cat(per_layer_ids[first_lid], dim=0)   # [total_tokens, top_k]
        top_k = all_ids.shape[1]
        total_tokens = all_ids.shape[0]

        # Full [total_tokens, num_layers, top_k] tensors
        expert_ids = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.int32)
        expert_weights = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.float32)

        for layer_idx, lid in enumerate(layer_ids):
            layer_ids_cat = torch.cat(per_layer_ids[lid], dim=0)
            layer_wts_cat = torch.cat(per_layer_wts[lid], dim=0)
            expert_ids[:, layer_idx, :] = layer_ids_cat.to(torch.int32)
            expert_weights[:, layer_idx, :] = layer_wts_cat.to(torch.float32)

        origin_gpu = torch.full((total_tokens,), global_dp_rank, dtype=torch.int32)

        print(log(
            f"Routing tensors: {total_tokens} tokens × {num_layers} layers × {top_k} experts/token",
            global_dp_rank,
        ))

        # Per-layer expert usage summary
        print(log("Expert usage summary (first 3 layers):", global_dp_rank))
        for layer_idx in range(min(3, num_layers)):
            layer_data = expert_ids[:, layer_idx, :]  # [tokens, top_k]
            unique, counts = torch.unique(layer_data, return_counts=True)
            top_experts = sorted(zip(unique.tolist(), counts.tolist()),
                                 key=lambda x: -x[1])[:5]
            summary = ", ".join(f"E{e}:{c}" for e, c in top_experts)
            print(log(f"  Layer {layer_ids[layer_idx]}: {summary}", global_dp_rank))

        # ── Save to disk (optional) ───────────────────────────────────────────
        if save_routing_pt:
            save_path = f"{save_routing_pt}_rank{global_dp_rank}.pt"
            torch.save({
                "token_ids":     torch.arange(total_tokens, dtype=torch.int64),
                "origin_gpu":    origin_gpu,
                "expert_ids":    expert_ids,
                "expert_weights": expert_weights,
                "num_tokens":    total_tokens,
                "num_layers":    num_layers,
                "top_k":         top_k,
                "dp_rank":       global_dp_rank,
                "num_steps":     len(snapshots),
            }, save_path)
            print(log(f"Saved routing data → {save_path}", global_dp_rank))

    print(log("Done.", global_dp_rank))
    sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve placement config to absolute path before forking
    expert_config_abs: Optional[str] = None
    if args.expert_placement_config:
        expert_config_abs = os.path.abspath(args.expert_placement_config)
        if not os.path.exists(expert_config_abs):
            raise FileNotFoundError(f"Expert placement config not found: {expert_config_abs}")
        print(f"Expert placement config: {expert_config_abs}")

    # DP master coordination
    if args.node_size == 1:
        from vllm.utils.network_utils import get_open_port
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    dp_size = args.dp_size
    dp_per_node = dp_size // args.node_size
    node_rank = args.node_rank

    print(
        f"Launching {dp_per_node} DP rank(s) on this node "
        f"(dp_size={dp_size}, tp_size={args.tp_size}, "
        f"expert_parallel={args.enable_expert_parallel}, "
        f"tracking=VLLM_TRACK_ROUTING={os.environ.get('VLLM_TRACK_ROUTING', '0')}, "
        f"placement={'enabled' if expert_config_abs else 'disabled'})"
    )

    procs = []
    for local_rank, global_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        p = Process(
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
                args.save_routing_pt,
            ),
        )
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join(timeout=args.timeout)
        if p.exitcode is None:
            print(f"Killing process {p.pid} (timeout after {args.timeout}s)")
            p.kill()
            exit_code = 1
        elif p.exitcode:
            exit_code = p.exitcode

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
