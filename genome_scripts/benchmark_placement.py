#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
benchmark_placement.py — Compare throughput across MoE expert placement strategies.

Measures generate() wall time and tokens/second for each scenario. Scenarios
differ in whether routing tracking and/or live expert placement are enabled,
and (for placement scenarios) how frequently EPLB fires.

Scenarios
---------
  baseline      Plain DP+EP: no routing capture, no expert movement.
                Represents the throughput ceiling — no instrumentation overhead.
  tracking      VLLM_TRACK_ROUTING=1 with no placement.
                Isolates per-step routing capture overhead from EPLB cost.
  interval_64   Tracking + callback placement, EPLB every 64 forward passes.
  interval_32   Tracking + callback placement, EPLB every 32 forward passes (default).
  interval_16   Tracking + callback placement, EPLB every 16 forward passes.
  interval_8    Tracking + callback placement, EPLB every 8 forward passes (aggressive).
  interval_1    EPLB every forward pass.
                WARNING: on PCIe-only L4 hardware each EPLB move takes 1–5 s.
                With 64 decode steps this scenario may take > 30 minutes. Use
                --timeout accordingly.

Quick start
-----------
  source merged/vllm/.venv/bin/activate
  cd merged/vllm

  NCCL_IB_DISABLE=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \\
  python genome_scripts/benchmark_placement.py \\
      --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --dp-size 8 --trust-remote-code --enforce-eager \\
      --num-prompts-per-rank 4 --output-length 64 \\
      --output-dir benchmark_results

Run a subset of scenarios
--------------------------
  python genome_scripts/benchmark_placement.py ... \\
      --scenarios baseline tracking interval_32 interval_8

All available scenario names
-----------------------------
  baseline  tracking  interval_64  interval_32  interval_16  interval_8  interval_1

Outputs (written to --output-dir)
-----------------------------------
  results.json          Per-rank timing data and aggregated stats.
  throughput.png        Bar chart: tokens/s per scenario.
  latency.png           Bar chart: generate() wall time per scenario.
  interval_sweep.png    Line chart: throughput vs. placement-step-interval.
  relative_overhead.png Bar chart: throughput relative to baseline (%).
"""

import argparse
import json
import os
import socket
import statistics
import time
from collections import OrderedDict
from multiprocessing import Process
from pathlib import Path
from time import sleep
from typing import Optional


# ─── Latency helpers ─────────────────────────────────────────────────────────

def _percentile(vals: list, p: float) -> float:
    """p-th percentile (p in 0..100) of a non-empty list."""
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(p / 100.0 * len(s))))
    return s[idx]


def _safe_mean(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _safe_std(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)

# ─── Scenario catalogue ──────────────────────────────────────────────────────
# Each entry defines: whether to capture routing, whether to run EPLB, and how
# often EPLB fires. track_routing=False disables the all-reduce barrier on
# every forward pass, so the baseline truly reflects raw DP+EP throughput.

ALL_SCENARIOS: OrderedDict = OrderedDict([
    ("baseline", {
        "track_routing": False,
        "callback_placement": False,
        "placement_step_interval": None,
        "description": "Plain DP+EP — no tracking, no expert movement",
        "color": "#4C72B0",
        "label": "Baseline\n(no features)",
    }),
    ("tracking", {
        "track_routing": True,
        "callback_placement": False,
        "placement_step_interval": None,
        "description": "Routing capture only — VLLM_TRACK_ROUTING=1, no EPLB",
        "color": "#DD8452",
        "label": "Tracking\n(no placement)",
    }),
    ("interval_64", {
        "track_routing": True,
        "callback_placement": True,
        "placement_step_interval": 64,
        "description": "Tracking + callback placement, EPLB every 64 forward passes",
        "color": "#55A868",
        "label": "Placement\ninterval=64",
    }),
    ("interval_32", {
        "track_routing": True,
        "callback_placement": True,
        "placement_step_interval": 32,
        "description": "Tracking + callback placement, EPLB every 32 forward passes",
        "color": "#C44E52",
        "label": "Placement\ninterval=32",
    }),
    ("interval_16", {
        "track_routing": True,
        "callback_placement": True,
        "placement_step_interval": 16,
        "description": "Tracking + callback placement, EPLB every 16 forward passes",
        "color": "#8172B2",
        "label": "Placement\ninterval=16",
    }),
    ("interval_8", {
        "track_routing": True,
        "callback_placement": True,
        "placement_step_interval": 8,
        "description": "Tracking + callback placement, EPLB every 8 forward passes (aggressive)",
        "color": "#937860",
        "label": "Placement\ninterval=8",
    }),
    ("interval_1", {
        "track_routing": True,
        "callback_placement": True,
        "placement_step_interval": 1,
        "description": (
            "EPLB every forward pass — on PCIe L4s each move takes 1–5 s; "
            "64 decode steps → very long run"
        ),
        "color": "#DA8BC3",
        "label": "Placement\ninterval=1\n(extreme)",
    }),
])

DEFAULT_SCENARIOS = [
    "baseline", "tracking", "interval_64", "interval_32", "interval_16", "interval_8",
]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoE expert placement benchmark — compare throughput across strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model & parallelism
    p.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    p.add_argument("--dp-size", type=int, default=8, dest="dp_size")
    p.add_argument("--tp-size", type=int, default=1, dest="tp_size")
    p.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")
    p.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    p.add_argument("--disable-expert-parallel", dest="enable_expert_parallel",
                   action="store_false")
    p.set_defaults(enable_expert_parallel=True)

    # Memory
    p.add_argument("--max-num-seqs", type=int, default=16, dest="max_num_seqs")
    p.add_argument("--max-model-len", type=int, default=512, dest="max_model_len")
    p.add_argument("--max-num-batched-tokens", type=int, default=1024,
                   dest="max_num_batched_tokens")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   dest="gpu_memory_utilization")

    # Workload
    p.add_argument("--num-prompts-per-rank", type=int, default=4,
                   dest="num_prompts_per_rank",
                   help="WikiText-2 prompts per DP rank. More = longer run, richer data.")
    p.add_argument("--output-length", type=int, default=64, dest="output_length",
                   help="Max new tokens to generate per prompt. Longer = more decode steps "
                        "= more EPLB events and more differentiated results.")

    # Benchmark control
    p.add_argument(
        "--scenarios", nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=list(ALL_SCENARIOS.keys()),
        help=f"Which scenarios to run. Defaults: {DEFAULT_SCENARIOS}. "
             f"All options: {list(ALL_SCENARIOS.keys())}",
        metavar="SCENARIO",
    )
    p.add_argument("--output-dir", default="benchmark_results", dest="output_dir",
                   help="Directory for results.json and plots (created if absent).")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Seconds to wait for each subprocess before force-killing. "
                        "Default 1800 s accommodates PCIe-heavy EPLB on L4 VMs. "
                        "Increase further if running interval_1.")
    p.add_argument("--inter-scenario-cooldown", type=int, default=65,
                   dest="inter_scenario_cooldown",
                   help="Seconds to sleep between scenarios. Linux tcp_fin_timeout "
                        "is 60 s by default, so vLLM's internal NCCL rendezvous "
                        "ports stay in TIME_WAIT for 60 s after a scenario exits. "
                        "The default 65 s gives a 5 s margin over that. "
                        "Set to 0 to disable (risks EADDRINUSE on later scenarios).")

    return p.parse_args()


# ─── Prompt loading ───────────────────────────────────────────────────────────

def load_prompts(total: int) -> list:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [
        text[:300]
        for row in ds
        if len(text := row["text"].strip()) > 100
    ]
    if not texts:
        raise RuntimeError("wikitext dataset returned no usable samples")
    return (texts * ((total // len(texts)) + 1))[:total]


# ─── Per-rank worker ──────────────────────────────────────────────────────────

def run_rank(
    # Scenario identity
    scenario_name: str,
    track_routing: bool,
    callback_placement: bool,
    placement_step_interval: Optional[int],
    timing_file: str,
    # DP configuration
    model: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    gpus_per_rank: int,
    # Engine options
    enforce_eager: bool,
    enable_expert_parallel: bool,
    trust_remote_code: bool,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    # Workload
    prompts: list,
    output_length: int,
) -> None:
    """Worker function — one process per DP rank. Writes timing JSON to timing_file."""
    is_rank0 = global_dp_rank == 0

    # ── DP env vars (must be set before vLLM is imported) ────────────────────
    os.environ["VLLM_DP_RANK"]        = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"]  = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"]        = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"]   = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Explicitly control VLLM_TRACK_ROUTING — don't rely on caller's env because
    # the user may have it set globally (e.g. VLLM_TRACK_ROUTING=1 in their shell).
    os.environ["VLLM_TRACK_ROUTING"] = "1" if track_routing else "0"

    use_placement = callback_placement
    interval = placement_step_interval or 32  # EPLBConfig needs an int; value unused when eplb off

    # ── Deferred imports ─────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.config import EPLBConfig

    eplb_config = EPLBConfig(
        policy="custom" if use_placement else "default",
        step_interval=interval,
        log_balancedness=False,
        use_async=use_placement,  # async transfer: background thread moves weights,
                                  # main thread only waits for fast local buffer copy
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
        # async_scheduling de-syncs EP ranks at different forward-pass counts →
        # NCCL all_reduce deadlock. Must be off whenever tracking is enabled.
        async_scheduling=False,
        # Enable per-request timing so RequestOutput.metrics is populated with
        # first_token_latency (TTFT) and first_token_ts / last_token_ts (decode).
        # The LLM() default sets disable_log_stats=True, which would leave
        # metrics=None. We override it here to collect online-inference latencies.
        disable_log_stats=False,
    )

    # ── Register placement callback (when EPLB is active) ────────────────────
    if use_placement:
        placement_fns_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "placement_fns.py"
        )
        llm.collective_rpc(
            "register_placement_callback",
            args=(placement_fns_path, "compute_placement"),
        )
        if is_rank0:
            print(f"  [{scenario_name}][Rank 0] compute_placement callback registered.")

    if is_rank0:
        print(
            f"  [{scenario_name}][Rank 0] Engine ready — "
            f"tracking={'on' if track_routing else 'off'}  "
            f"placement={'on (interval=' + str(interval) + ')' if use_placement else 'off'}"
        )

    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=output_length
    )

    # ── Timed generate() ─────────────────────────────────────────────────────
    # We reset the step counter then time only generate(), excluding model load
    # and the one-time callback registration above.
    llm.reset_step_counter()

    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.perf_counter()
    generate_time = t_end - t_start

    # ── Post-generate analysis ────────────────────────────────────────────────
    tokens_generated = sum(
        len(o.outputs[0].token_ids) for o in outputs if o.outputs
    )

    # ── Per-request latency metrics (from RequestOutput.metrics) ─────────────
    # Requires disable_log_stats=False (set above). Fields used:
    #   first_token_latency  — TTFT: wall-clock delta from arrival to first token.
    #   first_token_ts       — monotonic engine-core timestamp of first token.
    #   last_token_ts        — monotonic engine-core timestamp of last token.
    # Both *_ts fields are from the same monotonic clock in the same process,
    # so (last_token_ts - first_token_ts) = decode_duration is valid.
    request_ttfts:  list[float] = []
    request_tpots:  list[float] = []
    request_e2e:    list[float] = []

    for out in outputs:
        m = out.metrics
        if m is None or m.first_token_latency == 0.0:
            continue
        n_gen = m.num_generation_tokens
        ttft  = m.first_token_latency          # seconds, wall-clock
        request_ttfts.append(ttft)

        if n_gen > 1 and m.first_token_ts > 0.0 and m.last_token_ts > 0.0:
            decode_dur = m.last_token_ts - m.first_token_ts   # seconds, monotonic
            tpot = decode_dur / (n_gen - 1)
            request_tpots.append(tpot)
            request_e2e.append(ttft + decode_dur)
        else:
            # Single-token output: E2E ≈ TTFT
            request_e2e.append(ttft)

    def _ms(v):
        return round(v * 1000, 3) if v is not None else None

    ttft_mean_ms = _ms(statistics.mean(request_ttfts)) if request_ttfts else None
    ttft_p50_ms  = _ms(_percentile(request_ttfts, 50)) if request_ttfts else None
    ttft_p99_ms  = _ms(_percentile(request_ttfts, 99)) if request_ttfts else None
    tpot_mean_ms = _ms(statistics.mean(request_tpots)) if request_tpots else None
    tpot_p50_ms  = _ms(_percentile(request_tpots, 50)) if request_tpots else None
    tpot_p99_ms  = _ms(_percentile(request_tpots, 99)) if request_tpots else None
    e2e_mean_ms  = _ms(statistics.mean(request_e2e))   if request_e2e  else None
    e2e_p50_ms   = _ms(_percentile(request_e2e,  50))  if request_e2e  else None
    e2e_p99_ms   = _ms(_percentile(request_e2e,  99))  if request_e2e  else None

    # drain_step_snapshots() is always safe — returns [] when tracking=off
    snapshots = llm.drain_step_snapshots()
    num_steps = len(snapshots)
    num_eplb_events = (num_steps // interval) if use_placement else 0

    tok_per_sec = tokens_generated / generate_time if generate_time > 0 else 0.0

    has_lat = ttft_mean_ms is not None
    lat_str = (f"  TTFT {ttft_mean_ms:.0f} ms  TPOT {tpot_mean_ms:.0f} ms (p99 {tpot_p99_ms:.0f} ms)"
               if has_lat else "  (latency metrics unavailable)")
    print(
        f"  [{scenario_name}][Rank {global_dp_rank}] "
        f"{tokens_generated} tok in {generate_time:.1f} s "
        f"= {tok_per_sec:.1f} tok/s | "
        f"{num_steps} steps, {num_eplb_events} EPLB events"
    )
    print(f"  [{scenario_name}][Rank {global_dp_rank}]{lat_str}")

    # ── Write timing JSON ─────────────────────────────────────────────────────
    timing = {
        "dp_rank": global_dp_rank,
        "scenario": scenario_name,
        # Throughput
        "generate_time_s": generate_time,
        "tokens_generated": tokens_generated,
        "tok_per_sec": tok_per_sec,
        "num_prompts": len(outputs),
        # Routing / EPLB
        "num_steps": num_steps,
        "num_eplb_events": num_eplb_events,
        "placement_step_interval": placement_step_interval,
        "track_routing": track_routing,
        # Online-inference latencies (all in milliseconds)
        "ttft_mean_ms":  ttft_mean_ms,
        "ttft_p50_ms":   ttft_p50_ms,
        "ttft_p99_ms":   ttft_p99_ms,
        "tpot_mean_ms":  tpot_mean_ms,
        "tpot_p50_ms":   tpot_p50_ms,
        "tpot_p99_ms":   tpot_p99_ms,
        "e2e_mean_ms":   e2e_mean_ms,
        "e2e_p50_ms":    e2e_p50_ms,
        "e2e_p99_ms":    e2e_p99_ms,
    }
    Path(timing_file).write_text(json.dumps(timing, indent=2))

    sleep(1)  # let NCCL teardown finish before process exits


# ─── Scenario runner ──────────────────────────────────────────────────────────

def _log_gpu_processes(label: str) -> None:
    """Log current CUDA processes via nvidia-smi to detect unexpected GPU users.

    Prints a warning if any processes are found — helps distinguish between
    benchmark crashes caused by port/NCCL issues and those caused by another
    user's workload occupying GPU memory or NCCL network ports on the same host.
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if lines:
            print(f"  [GPU monitor — {label}] WARNING: {len(lines)} CUDA process(es) detected:")
            for line in lines:
                parts = (line + ",,").split(",", 2)
                pid, name, mem = parts[0], parts[1], parts[2]
                print(f"    PID {pid.strip()}  {name.strip():<35}  {mem.strip()} MiB")
        else:
            print(f"  [GPU monitor — {label}] GPUs clear — no CUDA processes detected.")
    except FileNotFoundError:
        print(f"  [GPU monitor — {label}] nvidia-smi not found — skipping GPU check.")
    except Exception as e:
        print(f"  [GPU monitor — {label}] nvidia-smi error: {e}")


def _pick_port(exclude: set) -> int:
    """Pick a free TCP port using stdlib only (no vllm import in the parent process).

    Using get_open_port() from vllm.utils.network_utils imports vLLM in the parent
    process, which opens internal sockets that are inherited by every forked child.
    When all 8 children exit they leave multiple copies of those sockets in TIME_WAIT,
    eventually causing a recycled port to hit EADDRINUSE on a later scenario.

    We avoid the vllm import entirely and also skip recently-used ports.
    """
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in exclude:
            return port
    # Fallback: return whatever we got last (shouldn't happen in practice)
    return port


def run_scenario(
    scenario_name: str,
    scenario_cfg: dict,
    args: argparse.Namespace,
    rank_prompts: list,
    timing_dir: str,
    used_ports: set,
) -> list:
    """Spawn one subprocess per DP rank for this scenario. Returns list of timing dicts.

    ``used_ports`` is mutated in place: every port tried (success or failure) is
    added so the next scenario's port selection skips it.
    """
    dp_master_ip = "127.0.0.1"
    MAX_ATTEMPTS = 3

    for attempt in range(MAX_ATTEMPTS):
        dp_master_port = _pick_port(exclude=used_ports)
        used_ports.add(dp_master_port)

        procs = []
        for rank in range(args.dp_size):
            timing_file = os.path.join(timing_dir, f"{scenario_name}_rank{rank}.json")
            # Remove stale file from a previous (failed) run
            if os.path.exists(timing_file):
                os.remove(timing_file)

            p = Process(
                target=run_rank,
                args=(
                    scenario_name,
                    scenario_cfg["track_routing"],
                    scenario_cfg["callback_placement"],
                    scenario_cfg["placement_step_interval"],
                    timing_file,
                    args.model,
                    args.dp_size,
                    rank,          # local_dp_rank  (single-node → same as global)
                    rank,          # global_dp_rank
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
                    rank_prompts[rank],
                    args.output_length,
                ),
            )
            p.start()
            procs.append(p)

        t_attempt = time.monotonic()
        failed = 0
        for p in procs:
            p.join(timeout=args.timeout)
            if p.exitcode is None:
                print(f"  WARNING: Killing process {p.pid} (timeout after {args.timeout} s)")
                p.kill()
                failed += 1
            elif p.exitcode != 0:
                failed += 1
        elapsed = time.monotonic() - t_attempt

        all_failed = failed == len(procs)
        fast_failure = elapsed < 60  # startup failure, not a runtime crash

        if all_failed and fast_failure and attempt < MAX_ATTEMPTS - 1:
            print(
                f"  All {len(procs)} ranks failed within {elapsed:.0f}s on port "
                f"{dp_master_port} (likely EADDRINUSE). "
                f"Retrying with a fresh port (attempt {attempt + 2}/{MAX_ATTEMPTS})..."
            )
            sleep(3)
            continue

        if failed:
            print(f"  WARNING: {failed}/{len(procs)} rank(s) failed in scenario '{scenario_name}'")
        break

    # Collect whatever timing files were written
    timings = []
    for rank in range(args.dp_size):
        timing_file = os.path.join(timing_dir, f"{scenario_name}_rank{rank}.json")
        if os.path.exists(timing_file):
            with open(timing_file) as f:
                timings.append(json.load(f))

    return timings


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(timings: list) -> dict:
    """Compute mean and std-dev across ranks for all metrics."""
    if not timings:
        return {}

    def _std(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    tok_per_sec = [t["tok_per_sec"] for t in timings]
    gen_times   = [t["generate_time_s"] for t in timings]
    steps       = [t["num_steps"] for t in timings]
    eplb_events = [t["num_eplb_events"] for t in timings]

    agg: dict = {
        # Throughput
        "tok_per_sec_mean":     statistics.mean(tok_per_sec),
        "tok_per_sec_std":      _std(tok_per_sec),
        "generate_time_mean":   statistics.mean(gen_times),
        "generate_time_std":    _std(gen_times),
        "total_tokens":         sum(t["tokens_generated"] for t in timings),
        # EPLB
        "num_steps_mean":       statistics.mean(steps),
        "num_eplb_events_mean": statistics.mean(eplb_events),
        "num_ranks_collected":  len(timings),
    }

    # Latency metrics — averaged across ranks (each rank runs different prompts,
    # so per-rank p99s are already across several requests; we average those).
    for key in (
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms",
        "tpot_mean_ms", "tpot_p50_ms", "tpot_p99_ms",
        "e2e_mean_ms",  "e2e_p50_ms",  "e2e_p99_ms",
    ):
        vals = [t.get(key) for t in timings if t.get(key) is not None]
        agg[key] = round(statistics.mean(vals), 3) if vals else None

    return agg


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(results: dict, all_scenarios: dict) -> None:
    """Print a formatted results table to stdout."""
    names = [n for n in all_scenarios if n in results and results[n].get("aggregate")]
    if not names:
        print("No results to display.")
        return

    def _fmt_ms(v):
        return f"{v:.0f} ms" if v is not None else "N/A"

    header = (
        f"{'Scenario':<20}  {'Tok/s (mean±std)':<20}  "
        f"{'Gen time':<16}  {'TTFT mean':>10}  "
        f"{'TPOT mean':>10}  {'TPOT p99':>10}  {'EPLB events':>12}"
    )
    sep = "─" * len(header)
    print()
    print("  " + sep)
    print("  " + header)
    print("  " + sep)
    for name in names:
        agg   = results[name]["aggregate"]
        tps   = f"{agg['tok_per_sec_mean']:.1f}±{agg['tok_per_sec_std']:.1f}"
        gtime = f"{agg['generate_time_mean']:.1f}±{agg['generate_time_std']:.1f} s"
        ttft  = _fmt_ms(agg.get("ttft_mean_ms"))
        tpot  = _fmt_ms(agg.get("tpot_mean_ms"))
        tp99  = _fmt_ms(agg.get("tpot_p99_ms"))
        eplb  = f"{agg['num_eplb_events_mean']:.0f}"
        print(
            f"  {name:<20}  {tps:<20}  {gtime:<16}  "
            f"{ttft:>10}  {tpot:>10}  {tp99:>10}  {eplb:>12}"
        )
    print("  " + sep)
    print()
    print("  TTFT  = Time to First Token (prefill latency proxy)")
    print("  TPOT  = Time Per Output Token (mean decode step duration)")
    print("  TPOT p99 = worst-case decode step — EPLB stalls dominate this metric")
    print()


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_dir: str, all_scenarios: dict) -> None:
    """Generate throughput, latency, interval sweep, and relative-overhead plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available — skipping plots.")
        print("Install with: pip install matplotlib numpy")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.7,
    })

    os.makedirs(output_dir, exist_ok=True)

    # Ordered list of completed scenarios (preserve scenario catalogue order)
    names  = [n for n in all_scenarios if n in results and results[n].get("aggregate")]
    if not names:
        print("No completed scenarios — no plots generated.")
        return

    labels   = [all_scenarios[n]["label"] for n in names]
    colors   = [all_scenarios[n]["color"] for n in names]
    tps      = np.array([results[n]["aggregate"]["tok_per_sec_mean"]   for n in names])
    tps_std  = np.array([results[n]["aggregate"]["tok_per_sec_std"]    for n in names])
    gts      = np.array([results[n]["aggregate"]["generate_time_mean"] for n in names])
    gts_std  = np.array([results[n]["aggregate"]["generate_time_std"]  for n in names])

    baseline_tps = (
        results["baseline"]["aggregate"]["tok_per_sec_mean"]
        if "baseline" in results and results["baseline"].get("aggregate")
        else None
    )
    x = np.arange(len(names))
    bar_width = 0.6
    fig_w = max(8, len(names) * 1.5)

    # ── Helper: annotate bars ────────────────────────────────────────────────
    def _annotate(ax, bars, vals, fmt="{:.1f}", offset_frac=0.03):
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_range * offset_frac,
                fmt.format(v),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    # ── 1. Throughput bar chart (tok/s) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bars = ax.bar(x, tps, yerr=tps_std, width=bar_width, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    if baseline_tps is not None:
        ax.axhline(baseline_tps, color="#222", linestyle="--", linewidth=1.2,
                   alpha=0.6, label=f"Baseline ({baseline_tps:.1f} tok/s)")
        ax.legend(fontsize=9)
    ax.set_ylim(0, max(tps + tps_std) * 1.25)
    _annotate(ax, bars, tps)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tokens / second  (mean across DP ranks)", fontsize=10)
    ax.set_title("MoE Placement Benchmark — Throughput", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "throughput.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ── 2. Generate wall-time bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bars = ax.bar(x, gts, yerr=gts_std, width=bar_width, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    ax.set_ylim(0, max(gts + gts_std) * 1.25)
    _annotate(ax, bars, gts, fmt="{:.1f}s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("generate() wall time in seconds  (mean across DP ranks)", fontsize=10)
    ax.set_title("MoE Placement Benchmark — Generate Latency", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "latency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ── 5. Interval sweep line chart ──────────────────────────────────────────
    interval_names = [
        n for n in names
        if all_scenarios[n]["placement_step_interval"] is not None
    ]
    if len(interval_names) >= 2:
        intervals = np.array([all_scenarios[n]["placement_step_interval"] for n in interval_names])
        i_tps     = np.array([results[n]["aggregate"]["tok_per_sec_mean"] for n in interval_names])
        i_std     = np.array([results[n]["aggregate"]["tok_per_sec_std"]  for n in interval_names])

        # Sort ascending by interval so the line goes left→right
        order     = np.argsort(intervals)
        intervals = intervals[order]
        i_tps     = i_tps[order]
        i_std     = i_std[order]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(intervals, i_tps, yerr=i_std,
                    fmt="o-", color="#2E86AB", linewidth=2, markersize=7,
                    capsize=5, elinewidth=1.5, label="Placement (callback)")

        # Annotate each point
        for xi, yi in zip(intervals, i_tps):
            ax.annotate(
                f"{yi:.1f}", xy=(xi, yi),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold",
            )

        # Horizontal baselines
        if baseline_tps is not None:
            ax.axhline(baseline_tps, color="#4C72B0", linestyle="--", linewidth=1.5,
                       alpha=0.85, label=f"Baseline ({baseline_tps:.1f} tok/s)")
        if "tracking" in results and results["tracking"].get("aggregate"):
            tr = results["tracking"]["aggregate"]["tok_per_sec_mean"]
            ax.axhline(tr, color="#DD8452", linestyle=":", linewidth=1.5,
                       alpha=0.85, label=f"Tracking-only ({tr:.1f} tok/s)")

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks(intervals)
        ax.set_xlabel("EPLB placement-step-interval (forward passes between rebalances)",
                      fontsize=10)
        ax.set_ylabel("Tokens / second  (mean across DP ranks)", fontsize=10)
        ax.set_title("Effect of EPLB Rebalance Frequency on Throughput",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(output_dir, "interval_sweep.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    else:
        print("  Skipping interval_sweep.png (fewer than 2 interval scenarios completed).")

    # ── 4. TTFT bar chart ────────────────────────────────────────────────────
    ttft_means = [results[n]["aggregate"].get("ttft_mean_ms") for n in names]
    ttft_p99s  = [results[n]["aggregate"].get("ttft_p99_ms")  for n in names]
    if any(v is not None for v in ttft_means):
        safe_means = [v if v is not None else 0.0 for v in ttft_means]
        safe_p99s  = [v if v is not None else 0.0 for v in ttft_p99s]

        fig, ax = plt.subplots(figsize=(fig_w, 5))
        bw = 0.35
        bars_m = ax.bar(x - bw / 2, safe_means, width=bw, label="Mean",
                        color=colors, edgecolor="white", linewidth=0.5)
        bars_p = ax.bar(x + bw / 2, safe_p99s, width=bw, label="p99",
                        color=[c + "80" for c in colors],  # ~50% alpha hex
                        edgecolor="white", linewidth=0.5, hatch="//")
        for bar, v in zip(bars_m, safe_means):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        for bar, v in zip(bars_p, safe_p99s):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Time to First Token (ms)", fontsize=10)
        ax.set_title("MoE Placement Benchmark — TTFT (Prefill Latency)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(safe_means), max(safe_p99s)) * 1.25)
        plt.tight_layout()
        out = os.path.join(output_dir, "ttft.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    else:
        print("  Skipping ttft.png (no latency data — was disable_log_stats=False set?)")

    # ── 5. TPOT bar chart ─────────────────────────────────────────────────────
    tpot_means = [results[n]["aggregate"].get("tpot_mean_ms") for n in names]
    tpot_p99s  = [results[n]["aggregate"].get("tpot_p99_ms")  for n in names]
    if any(v is not None for v in tpot_means):
        safe_means_t = [v if v is not None else 0.0 for v in tpot_means]
        safe_p99s_t  = [v if v is not None else 0.0 for v in tpot_p99s]

        fig, ax = plt.subplots(figsize=(fig_w, 5))
        bw = 0.35
        bars_m = ax.bar(x - bw / 2, safe_means_t, width=bw, label="Mean (avg decode step)",
                        color=colors, edgecolor="white", linewidth=0.5)
        bars_p = ax.bar(x + bw / 2, safe_p99s_t, width=bw, label="p99 (worst decode step = EPLB stall)",
                        color=[c + "80" for c in colors],
                        edgecolor="white", linewidth=0.5, hatch="//")
        for bar, v in zip(bars_m, safe_means_t):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        for bar, v in zip(bars_p, safe_p99s_t):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Time Per Output Token — TPOT (ms)", fontsize=10)
        ax.set_title(
            "MoE Placement Benchmark — TPOT (Decode Latency)\n"
            "p99 captures EPLB stalls: one rebalance ≈ 1–5 s on PCIe L4",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(safe_means_t), max(safe_p99s_t)) * 1.25)
        plt.tight_layout()
        out = os.path.join(output_dir, "tpot.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    else:
        print("  Skipping tpot.png (no TPOT data — need >1 output token per request)")

    # ── 7. Relative overhead bar chart ────────────────────────────────────────
    if baseline_tps is not None and baseline_tps > 0:
        rel     = tps     / baseline_tps * 100
        rel_std = tps_std / baseline_tps * 100

        fig, ax = plt.subplots(figsize=(fig_w, 5))
        bars = ax.bar(x, rel, yerr=rel_std, width=bar_width, capsize=5,
                      color=colors, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 1.5, "ecolor": "#333"})
        ax.axhline(100, color="#222", linestyle="--", linewidth=1.2,
                   alpha=0.6, label="Baseline (100%)")
        ax.set_ylim(0, max(rel + rel_std) * 1.20)
        _annotate(ax, bars, rel, fmt="{:.1f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Throughput relative to baseline (%)", fontsize=10)
        ax.set_title("MoE Placement Benchmark — Overhead vs Baseline",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(output_dir, "relative_overhead.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
    else:
        print("  Skipping relative_overhead.png (baseline scenario not in results).")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    scenarios_to_run = [s for s in args.scenarios if s in ALL_SCENARIOS]
    if not scenarios_to_run:
        print("No valid scenarios selected. Use --scenarios with one or more of:")
        print(" ", list(ALL_SCENARIOS.keys()))
        return

    output_dir = args.output_dir
    timing_dir = os.path.join(output_dir, "timing")
    os.makedirs(timing_dir, exist_ok=True)

    # Load prompts once — all scenarios see identical input for a fair comparison.
    n = args.num_prompts_per_rank
    print(f"Loading {args.dp_size * n} WikiText-2 prompts...")
    all_prompts  = load_prompts(args.dp_size * n)
    rank_prompts = [
        all_prompts[r * n: (r + 1) * n] or ["Placeholder."]
        for r in range(args.dp_size)
    ]

    print(
        f"\nBenchmark settings:\n"
        f"  model:               {args.model}\n"
        f"  dp_size:             {args.dp_size}\n"
        f"  tp_size:             {args.tp_size}\n"
        f"  num_prompts_per_rank:{n}\n"
        f"  output_length:       {args.output_length}\n"
        f"  scenarios:           {scenarios_to_run}\n"
        f"  output_dir:          {output_dir}\n"
    )

    print("NOTE: Each scenario runs in fresh subprocesses (clean CUDA contexts).")
    print("      Results reflect generate() time only — model loading is excluded.\n")

    results: dict = {}
    # Tracks every port used across all scenarios so _pick_port() never re-selects
    # one that may still be in TCP TIME_WAIT from a previous scenario's NCCL group.
    used_ports: set = set()
    cooldown = args.inter_scenario_cooldown

    for idx, scenario_name in enumerate(scenarios_to_run, 1):
        cfg = ALL_SCENARIOS[scenario_name]
        bar = "=" * 64
        print(f"\n{bar}")
        print(f"  Scenario {idx}/{len(scenarios_to_run)}: {scenario_name}")
        print(f"  {cfg['description']}")
        print(bar)

        _log_gpu_processes(f"before '{scenario_name}'")
        timings = run_scenario(scenario_name, cfg, args, rank_prompts, timing_dir, used_ports)
        _log_gpu_processes(f"after '{scenario_name}'")

        if timings:
            agg = aggregate(timings)
            results[scenario_name] = {
                "config":    {k: v for k, v in cfg.items() if k not in ("color", "label")},
                "timings":   timings,
                "aggregate": agg,
            }
            print(
                f"\n  Result: {agg['tok_per_sec_mean']:.1f} ± {agg['tok_per_sec_std']:.1f} tok/s  "
                f"({agg['generate_time_mean']:.1f} ± {agg['generate_time_std']:.1f} s)  "
                f"| ranks collected: {agg['num_ranks_collected']}/{args.dp_size}"
            )
        else:
            print(f"\n  WARNING: No timing data for scenario '{scenario_name}' — all ranks failed.")

        # Inter-scenario cooldown — let NCCL TCP connections from this scenario's
        # 8 child processes age through TIME_WAIT before the next port selection.
        if idx < len(scenarios_to_run) and cooldown > 0:
            print(
                f"\n  Cooling down {cooldown} s before next scenario "
                f"(NCCL TIME_WAIT connections from {scenario_name} need to clear)..."
            )
            sleep(cooldown)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(results, ALL_SCENARIOS)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_file}")

    # ── Generate plots ────────────────────────────────────────────────────────
    completed = sum(1 for r in results.values() if r.get("aggregate"))
    if completed >= 2:
        print("\nGenerating plots...")
        plot_results(results, output_dir, ALL_SCENARIOS)
    elif completed == 1:
        print("\nOnly 1 scenario completed — skipping plots (need at least 2 to compare).")
    else:
        print("\nNo scenarios completed — no plots generated.")

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
