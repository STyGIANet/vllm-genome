#!/usr/bin/env python3
"""
moe_combined.py — DP+EP inference with live token-routing tracking AND
expert redistribution driven by a user-supplied compute_placement() callback.

This script combines both features from the two research branches:
  • token_routing  — captures per-token expert assignments every step
  • expert-placement — applies a custom physical placement via StaticPlacementPolicy

Usage
-----
VLLM_TRACK_ROUTING=1 VLLM_EXPERT_CONFIG_PATH=<json> \
    NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \
    python token_tracing/moe_combined.py \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --dp-size 8 --trust-remote-code --enforce-eager

Environment variables
---------------------
VLLM_TRACK_ROUTING=1        Enable per-step routing capture (required).
VLLM_EXPERT_CONFIG_PATH     Path to placement JSON (optional; the callback
                            overrides this after the first step).
NCCL_IB_DISABLE=1           Disable InfiniBand (use for single-node setups).

The default compute_placement() below is a **passthrough** that returns the
same sequential layout every step.  Replace it with your own function that
reads the routing data and returns an expert→GPU assignment.
"""

import argparse
import multiprocessing
import os
import socket
import sys
from typing import Any


# ---------------------------------------------------------------------------
# User-supplied placement callback
# ---------------------------------------------------------------------------

def compute_placement(routing: dict) -> dict:
    """Example passthrough: keeps experts in their default sequential order.

    Args:
        routing: dict mapping layer_id → list of capture dicts, each with:
            - 'topk_ids':     Tensor[num_tokens, top_k]  (int32, CPU)
            - 'topk_weights': Tensor[num_tokens, top_k]  (float32, CPU)
            - 'num_tokens':   int

    Returns:
        dict with key "expert_to_gpu" mapping expert_id → gpu_rank, e.g.:
            {"expert_to_gpu": {"0": 0, "1": 0, "2": 1, "3": 1, ...}}
        Return an empty dict {} to skip placement update for this step.
    """
    # Example: count token visits per expert across all layers, then assign
    # the most-visited experts to GPU 0, etc.  Here we just return {} to
    # keep whatever placement is currently active (JSON file or default).
    return {}


# ---------------------------------------------------------------------------
# Per-DP-rank worker
# ---------------------------------------------------------------------------

def worker(
    rank: int,
    dp_size: int,
    master_ip: str,
    master_port: int,
    args: argparse.Namespace,
) -> None:
    gpus_per_rank = args.tp_size
    gpu_ids = list(range(rank * gpus_per_rank, (rank + 1) * gpus_per_rank))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_TRACK_ROUTING"] = "1"

    from vllm import LLM, SamplingParams
    from vllm.config import EPLBConfig

    eplb_cfg = EPLBConfig(
        policy="custom",
        step_interval=1,          # rebalance after every forward pass
        log_balancedness=False,
    ) if args.expert_placement else EPLBConfig()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=gpus_per_rank,
        enable_expert_parallel=True,
        enable_eplb=args.expert_placement,
        eplb_config=eplb_cfg,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
    )

    # Wire the compute_placement callback into the model runner on this rank.
    if args.expert_placement:
        try:
            # Access the underlying model runner through the engine internals.
            engine_core = llm.llm_engine.engine_core  # type: ignore[attr-defined]
            model_runner = engine_core.model_runner     # type: ignore[attr-defined]
            model_runner._compute_placement_callback = compute_placement
        except AttributeError:
            print(
                f"[rank {rank}] WARNING: could not attach compute_placement "
                "callback — engine internals may have changed.",
                file=sys.stderr,
            )

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    prompts = [
        "What is the capital of France?",
        "Explain the Pythagorean theorem.",
        "Write a short poem about the sea.",
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    if rank == 0:
        print(f"\n[rank {rank}] Running {len(prompts)} prompts …")

    outputs = llm.generate(prompts, sampling_params)

    if rank == 0:
        for out in outputs:
            print(f"\nPrompt : {out.prompt!r}")
            print(f"Output : {out.outputs[0].text!r}")

    # Drain per-step routing snapshots accumulated during generation.
    snapshots = llm.drain_step_snapshots()
    if rank == 0:
        print(f"\n[rank {rank}] Captured {len(snapshots)} routing step(s).")
        for snap in snapshots[:3]:
            layers = sorted(snap["routing"].keys())
            first_layer = snap["routing"][layers[0]]
            total_tokens = sum(c["num_tokens"] for c in first_layer)
            print(
                f"  step_idx={snap['step_idx']}  "
                f"layers={len(layers)}  "
                f"tokens_in_layer0={total_tokens}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE combined tracking + placement demo")
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--dp-size", type=int, default=1, dest="dp_size")
    parser.add_argument("--tp-size", type=int, default=1, dest="tp_size")
    parser.add_argument("--max-model-len", type=int, default=512, dest="max_model_len")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    parser.add_argument(
        "--expert-placement", action="store_true", dest="expert_placement",
        help="Enable EPLB with the custom placement policy + compute_placement callback",
    )
    args = parser.parse_args()

    master_ip = "127.0.0.1"
    master_port = _get_free_port()

    if args.dp_size == 1:
        # Single process — no need for multiprocessing.
        worker(0, 1, master_ip, master_port, args)
        return

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(args.dp_size):
        p = ctx.Process(
            target=worker,
            args=(rank, args.dp_size, master_ip, master_port, args),
            daemon=False,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"Ranks {failed} exited with non-zero exit code.", file=sys.stderr)
        sys.exit(1)

    print("\nAll ranks completed successfully.")


if __name__ == "__main__":
    main()
