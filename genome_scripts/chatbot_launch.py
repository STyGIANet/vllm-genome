#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
chatbot_launch.py — Multi-turn chatbot simulation with KV prefix caching.

Simulates real chatbot usage: each rank holds N independent chat sessions.
Each session runs --num-turns sequential generate() calls, appending the
model's response to the conversation history before the next user message.
Because vLLM's prefix cache stores past KV states, later turns reuse cached
computation for all tokens before the new user message.

Per-turn expert load is printed so you can observe how routing distributions
shift as the prefix cache warms up (hypothesis: cached prefixes skip early
layers → different expert activation profile → different optimal placement).

Usage (8×L4, Mixtral):
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \\
    python genome_scripts/chatbot_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --num-chats 2 --num-turns 4 --output-length 32

With live expert placement driven by the callback:
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \\
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \\
    python genome_scripts/chatbot_launch.py \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        --dp-size 8 --trust-remote-code --enforce-eager \\
        --callback-placement --placement-step-interval 32 \\
        --num-chats 2 --num-turns 4 --output-length 32

Environment variables
---------------------
VLLM_TRACK_ROUTING=1      Enable per-step routing capture.
NCCL_IB_DISABLE=1         Disable InfiniBand (single-node only).
VLLM_LOGGING_LEVEL=INFO   Show EPLB placement logs.
"""

import argparse
import os
from multiprocessing import Barrier, Process
from time import sleep
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Conversation templates
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is (opening_message, [follow_up_1, follow_up_2, ...]).
# Follow-ups cycle if --num-turns exceeds their count.
_CHAT_SESSIONS = [
    (
        "What is the capital of France?",
        ["Tell me more about its history.", "What is the population there?", "What language do they speak?"],
    ),
    (
        "Explain how neural networks learn.",
        ["What is backpropagation?", "How do activation functions work?", "What is overfitting?"],
    ),
    (
        "What causes the seasons on Earth?",
        ["How does the tilt affect temperature?", "Are seasons the same in both hemispheres?", "What about near the equator?"],
    ),
    (
        "How does photosynthesis work?",
        ["What role does chlorophyll play?", "Where does the oxygen come from?", "Can it happen at night?"],
    ),
    (
        "What is the difference between RAM and storage?",
        ["Which one is faster?", "What happens when RAM is full?", "How much RAM do modern phones have?"],
    ),
    (
        "How do airplanes generate lift?",
        ["What is Bernoulli's principle?", "How do flaps change lift?", "What happens during a stall?"],
    ),
    (
        "What is quantum entanglement?",
        ["Can it be used for communication?", "How is it experimentally observed?", "What is superposition?"],
    ),
    (
        "How does the immune system fight viruses?",
        ["What are T cells?", "How do vaccines train immunity?", "What is herd immunity?"],
    ),
]


def _format_turn(history: str, user_msg: str) -> str:
    """Append a user message to the conversation history."""
    return history + f"User: {user_msg}\nAssistant:"


def _initial_prompt(session_idx: int) -> str:
    opening, _ = _CHAT_SESSIONS[session_idx % len(_CHAT_SESSIONS)]
    return _format_turn("", opening)


def _follow_up(session_idx: int, turn: int) -> str:
    """Return the follow-up user message for a given session and turn index."""
    _, follow_ups = _CHAT_SESSIONS[session_idx % len(_CHAT_SESSIONS)]
    return follow_ups[(turn - 1) % len(follow_ups)]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-turn chatbot simulation with KV prefix caching (DP+EP mode)"
    )
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--dp-size", type=int, default=8, dest="dp_size")
    parser.add_argument("--tp-size", type=int, default=1, dest="tp_size")
    parser.add_argument("--node-size", type=int, default=1, dest="node_size")
    parser.add_argument("--node-rank", type=int, default=0, dest="node_rank")
    parser.add_argument("--master-addr", type=str, default="", dest="master_addr")
    parser.add_argument("--master-port", type=int, default=0, dest="master_port")
    parser.add_argument("--max-num-seqs", type=int, default=16, dest="max_num_seqs")
    parser.add_argument("--max-model-len", type=int, default=512, dest="max_model_len",
                        help="Keep ≤512 on 22 GiB L4s")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024,
                        dest="max_num_batched_tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        dest="gpu_memory_utilization")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    parser.add_argument("--disable-expert-parallel", dest="enable_expert_parallel",
                        action="store_false")
    parser.set_defaults(enable_expert_parallel=True)
    parser.add_argument("--num-chats", type=int, default=2, dest="num_chats",
                        help="Number of independent chat sessions per DP rank")
    parser.add_argument("--num-turns", type=int, default=4, dest="num_turns",
                        help="Number of conversation turns per chat session")
    parser.add_argument("--output-length", type=int, default=32, dest="output_length")
    parser.add_argument("--expert-placement-config", type=str, default=None,
                        dest="expert_placement_config",
                        help="Path to placement JSON (enables StaticPlacementPolicy)")
    parser.add_argument("--callback-placement", action="store_true",
                        dest="callback_placement",
                        help="Enable EPLB driven by compute_placement() callback")
    parser.add_argument("--placement-step-interval", type=int, default=32,
                        dest="placement_step_interval")
    parser.add_argument("--save-outputs", type=str, default="",
                        dest="save_outputs",
                        help="Write conversation turns to <path>_rank<N>.txt")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-rank worker
# ─────────────────────────────────────────────────────────────────────────────

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
    num_chats: int,
    num_turns: int,
    output_length: int,
    save_outputs: str,
    barrier: Barrier,
):
    is_rank0 = global_dp_rank == 0

    # ── Set DP env vars (must happen before LLM() is instantiated) ────────────
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["VLLM_TRACK_ROUTING"] = "1"

    use_placement = expert_placement_config_path is not None or callback_placement
    if expert_placement_config_path is not None:
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = expert_placement_config_path

    # ── Imports (deferred so env vars are set first) ──────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.config import EPLBConfig

    eplb_config = EPLBConfig(
        policy="custom" if use_placement else "default",
        step_interval=placement_step_interval,
        log_balancedness=False,
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
        enable_prefix_caching=True,
        # Async scheduling de-syncs EP ranks → NCCL deadlock in all_reduce.
        async_scheduling=False,
    )

    if use_placement:
        placement_fns_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "placement_fns.py"
        )
        llm.collective_rpc(
            "register_placement_callback",
            args=(placement_fns_path, "compute_placement"),
        )
        if is_rank0:
            print("[Rank 0] compute_placement callback registered.")

    if is_rank0:
        print(f"[Rank 0] Engines ready. chats_per_rank={num_chats} turns={num_turns} "
              f"placement={'on' if use_placement else 'off'}")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length)

    # Assign sessions to this rank: rank R owns session indices offset by global_dp_rank.
    # e.g. rank 0 → sessions 0, 8, 16 ... (with num_chats=2 → sessions 0, 8)
    session_indices = [
        global_dp_rank * num_chats + i for i in range(num_chats)
    ]

    # ── Initialise conversation histories (one per session) ───────────────────
    histories = [_initial_prompt(s) for s in session_indices]

    # turn_log: list of (session_idx, turn, user_msg, response, cache_label)
    turn_log: list[tuple[int, int, str, str, str]] = []

    llm.reset_step_counter()

    # ── Multi-turn loop ───────────────────────────────────────────────────────
    for turn in range(num_turns):
        # Synchronise all DP ranks before generate().  Without this barrier the
        # ranks finish their inter-turn post-processing at different times and
        # the first rank to call generate() starts NCCL ep_group.all_reduce()
        # operations inside _on_routing_step() before other ranks' EngineCores
        # have received any requests, causing a permanent NCCL deadlock.
        barrier.wait()

        outputs = llm.generate(histories, sampling_params)

        # Drain routing snapshots captured during this turn's generate() call.
        snapshots = llm.drain_step_snapshots()

        # Compute total tokens processed this turn.
        total_tokens = sum(
            sum(c["num_tokens"] for c in next(iter(s["routing"].values()), []))
            for s in snapshots
        )

        # Count token-to-expert assignments from topk_ids across all steps and
        # layers.  (expert_load is only available inside the placement callback;
        # drain_step_snapshots returns topk_ids / topk_weights / num_tokens.)
        expert_counts: dict[int, int] = {}
        for snap in snapshots:
            for captures in snap["routing"].values():
                for c in captures:
                    for expert_id in c["topk_ids"].reshape(-1).tolist():
                        expert_id = int(expert_id)
                        expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1

        if expert_counts:
            top_experts = sorted(expert_counts.items(), key=lambda x: -x[1])[:4]
            load_str = " ".join(f"E{e}({cnt})" for e, cnt in top_experts)
        else:
            load_str = "(no routing data)"

        cache_label = "cold" if turn == 0 else "warm"
        print(f"[Rank {global_dp_rank}] turn={turn} tokens={total_tokens} "
              f"cache={cache_label} top_experts: {load_str}")

        # ── Append model response and next user message to each history ───────
        for i, (out, sess_idx) in enumerate(zip(outputs, session_indices)):
            response = out.outputs[0].text.strip()

            # Derive the user message for this turn for logging.
            if turn == 0:
                user_msg = _CHAT_SESSIONS[sess_idx % len(_CHAT_SESSIONS)][0]
            else:
                user_msg = _follow_up(sess_idx, turn)
            turn_log.append((sess_idx, turn, user_msg, response, cache_label))

            if turn < num_turns - 1:
                follow_up = _follow_up(sess_idx, turn + 1)
                histories[i] = (
                    histories[i] + response + f"\nUser: {follow_up}\nAssistant:"
                )
            else:
                print(f"[Rank {global_dp_rank}] session={sess_idx} "
                      f"turn={turn} → {response[:80]!r}")

    if save_outputs:
        save_path = f"{save_outputs}_rank{global_dp_rank}.txt"
        sep = "=" * 80
        cur_session = None
        with open(save_path, "w") as f:
            for sess_idx, turn, user_msg, response, cache_label in turn_log:
                if sess_idx != cur_session:
                    cur_session = sess_idx
                    f.write(f"\n{'#' * 80}\n")
                    f.write(f"# Session {sess_idx}  (Rank {global_dp_rank})\n")
                    f.write(f"{'#' * 80}\n\n")
                f.write(f"{sep}\n")
                f.write(f"Turn {turn + 1} of {num_turns}  [{cache_label}]\n")
                f.write(f"{sep}\n")
                f.write(f"User: {user_msg}\n\n")
                f.write(f"Assistant: {response}\n\n")
        print(f"[Rank {global_dp_rank}] Saved outputs → {save_path}")

    sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    use_placement = expert_config_abs is not None or args.callback_placement
    print(
        f"Launching dp_size={dp_size} tp_size={args.tp_size} "
        f"chats_per_rank={args.num_chats} turns={args.num_turns} "
        f"placement={'on' if use_placement else 'off'}"
    )

    # All DP ranks must enter generate() together each turn.  Without this
    # barrier the first rank to finish inter-turn post-processing starts NCCL
    # operations inside its EngineCore before other EngineCores have work,
    # causing a permanent EP all_reduce deadlock.
    barrier = Barrier(dp_size)

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
                args.num_chats,
                args.num_turns,
                args.output_length,
                args.save_outputs,
                barrier,
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
