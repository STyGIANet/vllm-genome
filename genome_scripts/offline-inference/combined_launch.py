#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
combined_launch.py - DP+EP inference with token routing tracking and live
expert redistribution via a user-supplied compute_placement() callback.

Primary integration script for the moe-merged branch. Combines:
  * per-step MoE routing capture (from token_routing branch)
  * custom expert placement via StaticPlacementPolicy (from expert-placement branch)

Usage (8xL4, Mixtral, both features):
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    python genome_scripts/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --expert-placement-config genome_scripts/mixtral_EP_test.json

Prompts are drawn from WikiText-2 (HuggingFace datasets).
For multi-turn chatbot simulation see chatbot_launch.py.

Tracking only (no placement):
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 \\
    python genome_scripts/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager

Placement only (no tracking):
    NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    python genome_scripts/combined_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --expert-placement-config genome_scripts/mixtral_EP_test.json

Environment variables
---------------------
VLLM_TRACK_ROUTING=1      Enable per-step routing capture.
NCCL_IB_DISABLE=1         Disable InfiniBand (single-node only).
VLLM_LOGGING_LEVEL=INFO   Show EPLB placement logs from custom_policy.py.
"""

import argparse
import os
import random
from collections import deque
from multiprocessing import Process, Queue
from time import sleep
from typing import Optional

import torch


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined token routing tracking + expert placement (DP+EP mode)"
    )
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    parser.add_argument("--dp-size", type=int, default=8, dest="dp_size")
    parser.add_argument("--tp-size", type=int, default=1, dest="tp_size")
    parser.add_argument("--node-size", type=int, default=1, dest="node_size")
    parser.add_argument("--node-rank", type=int, default=0, dest="node_rank")
    parser.add_argument("--master-addr", type=str, default="", dest="master_addr")
    parser.add_argument("--master-port", type=int, default=0, dest="master_port")
    parser.add_argument("--max-num-seqs", type=int, default=16, dest="max_num_seqs")
    parser.add_argument("--max-model-len", type=int, default=512, dest="max_model_len",
                        help="Keep <=512 on 22 GiB L4s")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024,
                        dest="max_num_batched_tokens",
                        help="Default 1024 avoids OOM on L4s with EP=8 "
                             "(EP all-gather multiplies effective tokens by ep_size)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        dest="gpu_memory_utilization")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    parser.add_argument("--disable-expert-parallel", dest="enable_expert_parallel",
                        action="store_false")
    parser.set_defaults(enable_expert_parallel=True)
    parser.add_argument("--num-prompts-per-rank", type=int, default=3,
                        dest="num_prompts_per_rank")
    parser.add_argument("--dispatch-batch-size", type=int, default=1,
                        dest="dispatch_batch_size",
                        help="Maximum number of real prompts the coordinator assigns "
                             "to one worker in a dispatch round.")
    parser.add_argument("--load-balancing-policy", type=str, default="static",
                        choices=["static", "random"],
                        dest="load_balancing_policy",
                        help="Prompt-to-rank assignment policy. "
                             "'static' preserves the old fixed split. "
                             "'random' dispatches each prompt to a random DP rank.")
    parser.add_argument("--load-balancing-seed", type=int, default=0,
                        dest="load_balancing_seed",
                        help="Random seed used by the load balancer.")
    parser.add_argument("--output-length", type=int, default=10, dest="output_length")
    parser.add_argument("--expert-placement-config", type=str, default=None,
                        dest="expert_placement_config",
                        help="Path to placement JSON (enables StaticPlacementPolicy)")
    parser.add_argument("--callback-placement", action="store_true",
                        dest="callback_placement",
                        help="Enable EPLB driven entirely by compute_placement() callback "
                             "(no JSON required; StaticPlacementPolicy starts from identity)")
    parser.add_argument("--placement-step-interval", type=int, default=32,
                        dest="placement_step_interval",
                        help="Run EPLB rebalance every N forward passes. "
                             "Default 32 avoids PCIe saturation on L4s "
                             "(each full Mixtral rearrangement moves ~90 GB of weights)")
    parser.add_argument("--save-routing-pt", type=str, default="",
                        dest="save_routing_pt",
                        help="Save per-rank routing tensors to <path>_rank<N>.pt")
    parser.add_argument("--save-coactivation-pt", type=str, default="",
                        dest="save_coactivation_pt",
                        help="Save token co-activation graph to <path>_rank<N>.pt "
                             "(nodes are unique (layer, expert) pairs)")
    parser.add_argument("--save-outputs", type=str, default="",
                        dest="save_outputs",
                        help="Write prompts and responses to <path>_rank<N>.txt")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------------

def load_prompts(total_prompts: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [
        text[:300]
        for row in ds
        if len(text := row["text"].strip()) > 100
    ]
    if not texts:
        raise RuntimeError("wikitext dataset returned no usable samples")
    return (texts * ((total_prompts // len(texts)) + 1))[:total_prompts]


def _select_target_rank(
    prompt_idx: int,
    dp_size: int,
    policy: str,
    rng: random.Random,
) -> int:
    if policy == "static":
        return prompt_idx % dp_size
    if policy == "random":
        return rng.randrange(dp_size)
    raise ValueError(f"Unsupported load balancing policy: {policy}")


def build_prompt_queues(
    prompts: list[str],
    dp_size: int,
    policy: str,
    seed: int,
) -> tuple[list[Queue], list[int]]:
    """Assign prompts to per-rank queues using the requested policy."""
    rng = random.Random(seed)
    queues = [Queue() for _ in range(dp_size)]
    counts = [0] * dp_size

    for prompt_idx, prompt in enumerate(prompts):
        target_rank = _select_target_rank(prompt_idx, dp_size, policy, rng)
        queues[target_rank].put((prompt_idx, prompt))
        counts[target_rank] += 1

    for queue in queues:
        queue.put(None)

    return queues, counts


def assign_prompts_for_round(
    pending_prompts: deque[tuple[int, str]],
    dp_size: int,
    policy: str,
    rng: random.Random,
    dispatch_batch_size: int,
) -> list[list[tuple[int, str]]]:
    """Assign the next online dispatch round."""
    assignments: list[list[tuple[int, str]]] = [[] for _ in range(dp_size)]
    deferred: deque[tuple[int, str]] = deque()

    while pending_prompts:
        prompt_id, prompt = pending_prompts.popleft()
        target_rank = _select_target_rank(prompt_id, dp_size, policy, rng)
        if len(assignments[target_rank]) < dispatch_batch_size:
            assignments[target_rank].append((prompt_id, prompt))
        else:
            deferred.append((prompt_id, prompt))

        if all(len(batch) >= dispatch_batch_size for batch in assignments):
            break

    while deferred:
        pending_prompts.appendleft(deferred.pop())

    return assignments


def handle_generate_results(
    global_dp_rank: int,
    prompt_ids: list[int],
    prompts: list[str],
    outputs,
    llm,
    save_routing_pt: str,
    save_coactivation_pt: str,
    save_outputs: str,
    round_idx: int,
) -> None:
    for prompt_id, out in zip(prompt_ids, outputs):
        if prompt_id < 0:
            continue
        prefix = f"[Rank {global_dp_rank}][Round {round_idx}][Prompt {prompt_id}]"
        print(f"{prefix} {out.prompt[:60]!r} -> {out.outputs[0].text[:80]!r}")

    if save_outputs:
        save_path = f"{save_outputs}_rank{global_dp_rank}.txt"
        sep = "=" * 80
        with open(save_path, "a") as f:
            for prompt_id, out in zip(prompt_ids, outputs):
                if prompt_id < 0:
                    continue
                f.write(f"{sep}\n")
                f.write(
                    f"Rank {global_dp_rank}  |  Round {round_idx}  |  "
                    f"Prompt id {prompt_id}\n"
                )
                f.write(f"{sep}\n")
                f.write(out.prompt.strip())
                f.write("\n\n--- Response ---\n")
                f.write(out.outputs[0].text.strip())
                f.write("\n\n")
        print(f"[Rank {global_dp_rank}] Saved outputs -> {save_path}")

    snapshots = llm.drain_step_snapshots()
    if not snapshots:
        return

    real_indices = [i for i, prompt_id in enumerate(prompt_ids) if prompt_id >= 0]
    if not real_indices:
        return

    step_summary = "  ".join(
        f"step{s['step_idx']}:"
        + str(sum(c["num_tokens"] for c in next(iter(s["routing"].values()), [])))
        + "tok"
        for s in snapshots
    )
    print(f"[Rank {global_dp_rank}] Round {round_idx} {len(snapshots)} steps - {step_summary}")

    layer_ids = sorted(snapshots[0]["routing"].keys())
    num_layers = len(layer_ids)

    per_layer_ids: dict[int, list] = {lid: [] for lid in layer_ids}
    per_layer_wts: dict[int, list] = {lid: [] for lid in layer_ids}

    for snap in snapshots:
        for lid in layer_ids:
            for cap in snap["routing"].get(lid, []):
                per_layer_ids[lid].append(cap["topk_ids"])
                per_layer_wts[lid].append(cap["topk_weights"])

    first_lid = layer_ids[0]
    all_ids = torch.cat(per_layer_ids[first_lid], dim=0)
    top_k = all_ids.shape[1]
    total_tokens = all_ids.shape[0]

    expert_ids = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.int32)
    expert_weights = torch.zeros(total_tokens, num_layers, top_k, dtype=torch.float32)

    for layer_idx, lid in enumerate(layer_ids):
        expert_ids[:, layer_idx, :] = torch.cat(per_layer_ids[lid], dim=0).to(torch.int32)
        expert_weights[:, layer_idx, :] = torch.cat(per_layer_wts[lid], dim=0).to(torch.float32)

    usage_parts = []
    for layer_idx in range(min(3, num_layers)):
        layer_data = expert_ids[:, layer_idx, :]
        unique, counts = torch.unique(layer_data, return_counts=True)
        top_experts = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:3]
        usage_parts.append(
            "L" + str(layer_ids[layer_idx]) + ":"
            + ",".join(f"E{e}({c})" for e, c in top_experts)
        )
    print(f"[Rank {global_dp_rank}] Round {round_idx} {total_tokens} tokens - top experts: {' | '.join(usage_parts)}")

    adjacency, node_map = build_token_coactivation_graph(expert_ids)
    upper = torch.triu(adjacency, diagonal=1)
    edge_count = int((upper > 0).sum().item())
    total_weight = int(upper.sum().item())
    print(f"[Rank {global_dp_rank}] Round {round_idx} co-activation graph: "
          f"{adjacency.shape[0]} nodes, {edge_count} edges, weight_sum={total_weight}")

    if save_coactivation_pt:
        save_path = f"{save_coactivation_pt}_rank{global_dp_rank}_round{round_idx}.pt"
        torch.save({
            "adjacency": adjacency,
            "node_map": node_map,
            "num_nodes": adjacency.shape[0],
            "num_edges": edge_count,
            "total_edge_weight": total_weight,
            "num_experts": adjacency.shape[0] // num_layers if num_layers else 0,
            "num_layers": num_layers,
            "dp_rank": global_dp_rank,
            "num_tokens": total_tokens,
            "prompt_ids": [pid for pid in prompt_ids if pid >= 0],
        }, save_path)
        print(f"[Rank {global_dp_rank}] Saved co-activation graph -> {save_path}")

    if save_routing_pt:
        save_path = f"{save_routing_pt}_rank{global_dp_rank}_round{round_idx}.pt"
        origin_gpu = torch.full((total_tokens,), global_dp_rank, dtype=torch.int32)
        torch.save({
            "token_ids":      torch.arange(total_tokens, dtype=torch.int64),
            "origin_gpu":     origin_gpu,
            "expert_ids":     expert_ids,
            "expert_weights": expert_weights,
            "num_tokens":     total_tokens,
            "num_layers":     num_layers,
            "top_k":          top_k,
            "dp_rank":        global_dp_rank,
            "num_steps":      len(snapshots),
            "prompt_ids":     [pid for pid in prompt_ids if pid >= 0],
        }, save_path)
        print(f"[Rank {global_dp_rank}] Saved routing -> {save_path}")


def build_token_coactivation_graph(
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Build a token-level co-activation graph over unique (layer, expert) nodes.

    Each token contributes one count to every unordered pair of unique
    (layer, expert) nodes it invoked anywhere in the model.
    """
    if expert_ids.numel() == 0:
        return torch.zeros(0, 0, dtype=torch.int64), []

    total_tokens, num_layers, _top_k = expert_ids.shape
    num_experts = int(expert_ids.max().item()) + 1
    num_nodes = num_layers * num_experts
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.int64)

    node_map = [
        {
            "node_id": layer_idx * num_experts + expert_id,
            "layer_id": layer_idx,
            "expert_id": expert_id,
        }
        for layer_idx in range(num_layers)
        for expert_id in range(num_experts)
    ]

    layer_offsets = (
        torch.arange(num_layers, dtype=torch.int64).view(num_layers, 1) * num_experts
    )

    for token_idx in range(total_tokens):
        token_nodes = (
            expert_ids[token_idx].to(torch.int64) + layer_offsets
        ).reshape(-1)
        token_nodes = torch.unique(token_nodes, sorted=True)
        if token_nodes.numel() < 2:
            continue

        row_idx, col_idx = torch.triu_indices(
            token_nodes.numel(), token_nodes.numel(), offset=1
        )
        src_nodes = token_nodes[row_idx]
        dst_nodes = token_nodes[col_idx]
        adjacency[src_nodes, dst_nodes] += 1
        adjacency[dst_nodes, src_nodes] += 1

    return adjacency, node_map


# -----------------------------------------------------------------------------
# Per-rank worker
# -----------------------------------------------------------------------------

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
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    expert_placement_config_path: Optional[str],
    callback_placement: bool,
    placement_step_interval: int,
    task_queue: Queue,
    result_queue: Queue,
    output_length: int,
    save_routing_pt: str,
    save_coactivation_pt: str,
    save_outputs: str,
):
    is_rank0 = global_dp_rank == 0

    # -- Set DP env vars (must happen before LLM() is instantiated) ------------
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["VLLM_TRACK_ROUTING"] = "1"

    use_placement = expert_placement_config_path is not None or callback_placement
    if expert_placement_config_path is not None:
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = expert_placement_config_path

    # -- Imports (deferred so env vars are set first) --------------------------
    from vllm import LLM, SamplingParams
    from vllm.config import EPLBConfig

    eplb_config = EPLBConfig(
        policy="custom" if use_placement else "default",
        step_interval=placement_step_interval,
        log_balancedness=False,
        use_async=use_placement,  # async expert transfer - background thread moves
                                  # weights layer-by-layer, main thread only stalls
                                  # for a fast local copy from staging buffer
    )

    llm = LLM(
        model=model,
        tensor_parallel_size=gpus_per_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        enable_eplb=use_placement,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        eplb_config=eplb_config,
        # Keep async scheduling off while experimenting with custom EPLB
        # placement so request progress stays aligned with rebalance cadence.
        async_scheduling=False,
    )

    # -- Wire compute_placement callback into the model runner via RPC ---------
    # Direct attribute access on llm.llm_engine.engine_core doesn't reach the
    # worker subprocess in DP multiprocess mode (engine_core is a SyncMPClient
    # proxy). Instead we use collective_rpc to tell each worker to import
    # placement_fns.compute_placement in its own address space.
    if use_placement:
        placement_fns_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "placement_fns.py"
        )
        llm.collective_rpc(
            "register_placement_callback",
            args=(placement_fns_path, "compute_placement"),
        )
        if is_rank0:
            print("[Rank 0] compute_placement callback registered on all workers.")

    if is_rank0:
        print(f"[Rank 0] All engines ready. placement={'on' if use_placement else 'off'}, "
              f"tracking=on")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length)
    round_idx = 0

    while True:
        command, payload = task_queue.get()
        if command == "shutdown":
            break
        if command != "run_round":
            raise ValueError(f"Unknown worker command: {command}")

        prompt_tasks = payload
        if prompt_tasks:
            prompt_ids = [prompt_id for prompt_id, _prompt in prompt_tasks]
            prompts = [prompt for _prompt_id, prompt in prompt_tasks]
        else:
            prompt_ids = [-1]
            prompts = ["Placeholder."]

        llm.reset_step_counter()
        outputs = llm.generate(prompts, sampling_params)
        handle_generate_results(
            global_dp_rank=global_dp_rank,
            prompt_ids=prompt_ids,
            prompts=prompts,
            outputs=outputs,
            llm=llm,
            save_routing_pt=save_routing_pt,
            save_coactivation_pt=save_coactivation_pt,
            save_outputs=save_outputs,
            round_idx=round_idx,
        )
        result_queue.put({
            "rank": global_dp_rank,
            "round_idx": round_idx,
            "num_real_prompts": len([pid for pid in prompt_ids if pid >= 0]),
            "prompt_ids": [pid for pid in prompt_ids if pid >= 0],
        })
        round_idx += 1

    sleep(1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    expert_config_abs: Optional[str] = None
    if args.expert_placement_config:
        expert_config_abs = os.path.abspath(args.expert_placement_config)
        if not os.path.exists(expert_config_abs):
            raise FileNotFoundError(f"Expert placement config not found: {expert_config_abs}")

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
    n = args.num_prompts_per_rank

    all_prompts = load_prompts(dp_size * n)
    pending_prompts: deque[tuple[int, str]] = deque(enumerate(all_prompts))
    result_queue: Queue = Queue()
    task_queues = [Queue() for _ in range(dp_size)]

    use_placement = expert_config_abs is not None or args.callback_placement
    tracking = os.environ.get("VLLM_TRACK_ROUTING", "0") == "1"
    if expert_config_abs:
        placement_label = f"on ({expert_config_abs})"
    elif args.callback_placement:
        placement_label = "on (callback-only)"
    else:
        placement_label = "off"
    print(
        f"Launching dp_size={dp_size} tp_size={args.tp_size} "
        f"tracking={'on' if tracking else 'off'} "
        f"placement={placement_label} "
        f"lb_policy={args.load_balancing_policy}"
    )
    print(f"Coordinator managing {len(all_prompts)} prompts.")

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
                args.max_num_batched_tokens,
                args.max_model_len,
                args.gpu_memory_utilization,
                expert_config_abs,
                args.callback_placement,
                args.placement_step_interval,
                task_queues[global_rank],
                result_queue,
                args.output_length,
                args.save_routing_pt,
                args.save_coactivation_pt,
                args.save_outputs,
            ),
        )
        p.start()
        procs.append(p)

    dispatch_rng = random.Random(args.load_balancing_seed)
    round_idx = 0
    while pending_prompts:
        assignments = assign_prompts_for_round(
            pending_prompts=pending_prompts,
            dp_size=dp_size,
            policy=args.load_balancing_policy,
            rng=dispatch_rng,
            dispatch_batch_size=args.dispatch_batch_size,
        )
        assigned_count = sum(len(batch) for batch in assignments)
        if assigned_count == 0:
            raise RuntimeError("Coordinator failed to assign any prompts in a round.")

        print(
            f"Coordinator round {round_idx}: "
            + "  ".join(f"rank{rank}={len(batch)}" for rank, batch in enumerate(assignments))
        )
        for rank, batch in enumerate(assignments):
            task_queues[rank].put(("run_round", batch))

        results = []
        for _ in range(dp_size):
            results.append(result_queue.get(timeout=args.timeout))
        results.sort(key=lambda item: item["rank"])
        print(
            f"Coordinator completed round {round_idx}: "
            + "  ".join(
                f"rank{item['rank']} done={item['num_real_prompts']}"
                for item in results
            )
        )
        round_idx += 1

    for task_queue in task_queues:
        task_queue.put(("shutdown", None))

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
