# Benchmark Placement — Output Guide

`benchmark_placement.py` runs a series of MoE placement configurations back-to-back and
produces a JSON results file plus six comparison plots. This document explains every
output, every metric, and how to interpret what you see.

---

## Quick reference: output files

| File | What it contains |
|---|---|
| `results.json` | Per-rank timing + aggregated stats for every completed scenario |
| `throughput.png` | Bar chart — tokens/second per scenario |
| `latency.png` | Bar chart — total `generate()` wall time per scenario |
| `ttft.png` | Bar chart — Time to First Token (mean and p99) per scenario |
| `tpot.png` | Bar chart — Time Per Output Token: mean and p99 side-by-side |
| `interval_sweep.png` | Line chart — tok/s vs. placement-step-interval (log₂ x-axis) |
| `relative_overhead.png` | Bar chart — throughput as % of the baseline scenario |
| `timing/` | Raw per-rank JSON files, one per rank per scenario |

---

## Metric definitions

### Throughput

**tok/s** — total output tokens generated divided by the `generate()` wall time.

```
tok/s = tokens_generated / generate_time_s
```

- Measured per rank; the plot shows mean ± std across all 8 DP ranks.
- Each rank runs different prompts independently, so variance reflects natural
  load differences between ranks, not measurement noise.
- `generate_time_s` starts just before `llm.generate(prompts, ...)` returns and
  ends just after — it includes all EPLB stalls, all_reduce barriers, and routing
  capture overhead, but excludes model loading and callback registration.

### Generate wall time

Total wall-clock seconds for the `generate()` call on each rank.  The inverse of
throughput, shown separately because some scenarios generate different numbers of
output tokens (if the model hits `max_tokens` vs. an EOS token early).

---

### TTFT — Time to First Token

Time from when a request was submitted to vLLM until the first output token is
produced. Measured per request, then averaged across all requests on the rank.

```
TTFT per request = first_token_latency  (pre-computed wall-clock delta in vLLM)
```

**What drives TTFT in this system:**

1. **Prefill compute** — all prompt tokens processed in a single forward pass.
   Longer prompts = longer TTFT.
2. **EPLB at prefill time** — if an EPLB rebalance fires on the same step as
   the prefill forward pass, the prefill is stalled until weight transfers finish.
   With `placement_step_interval=1` this happens on every step including prefill.
3. **Queue time** — in offline batch mode, all prompts are submitted at once and
   the scheduler processes them in batches, so some requests wait in the queue.

**Expected pattern:** TTFT should increase slightly for placement scenarios vs.
the baseline (prefill step may coincide with an EPLB rebalance), and increase
further for very low intervals.

**p99 TTFT** — useful for detecting if some requests happened to land on an EPLB
step and were stalled for several seconds.

---

### TPOT — Time Per Output Token

Average time between consecutive output tokens for a single request, measured
over the entire decode phase.

```
decode_duration = last_token_ts - first_token_ts   (both monotonic engine-core timestamps)

TPOT per request = decode_duration / (num_output_tokens - 1)
```

This is the single most diagnostic metric for EPLB overhead. Here is why:

- Each decode step produces one token per request.
- An EPLB rebalance blocks the decode loop: all 8 ranks synchronise via NCCL,
  transfer expert weights with P2P, and then `cuda.synchronize()` before
  continuing. On PCIe-only L4 hardware this adds **1–5 seconds** to that one
  decode step.
- Mean TPOT averages over many decode steps (most of which are fast), so it
  rises gradually as interval decreases.
- **p99 TPOT** captures the worst step — almost certainly an EPLB step —
  and shows the true tail latency. A request processed during a rebalance
  could wait seconds for its next token.

**Expected pattern:**

| Scenario | Mean TPOT | p99 TPOT |
|---|---|---|
| baseline | low (e.g. 20 ms) | similar to mean |
| tracking | slightly higher (all_reduce barrier on every step) | similar to mean |
| interval_64 | slightly higher | **spike** — 1–5 s on PCIe L4 |
| interval_32 | higher still | spike |
| interval_8 | noticeably higher | spike |
| interval_1 | very high | every step is a rebalance stall |

**TPOT vs ITL:** In the literature, ITL (inter-token latency) measures the
time between *consecutive* output tokens as observed by the end user in a
streaming context. TPOT as computed here is equivalent — it is the mean ITL
over the full decode sequence for one request.

---

### E2E Latency — End-to-End Request Latency

Total time from request arrival to completion, per request.

```
E2E latency ≈ TTFT + decode_duration
            = first_token_latency + (last_token_ts - first_token_ts)
```

> **Note on time bases:** `first_token_latency` is a wall-clock delta (frontend
> process), while `last_token_ts - first_token_ts` uses monotonic engine-core
> timestamps (same process, so the subtraction is valid). The sum is an accurate
> approximation of total wall time per request.

E2E latency is the user-visible metric in production online serving. High TPOT
(from EPLB stalls) inflates E2E proportionally to the number of decode steps:
a 3-second EPLB stall adds 3 seconds to the E2E latency of every request being
decoded at that moment.

---

## Results JSON structure

```jsonc
{
  "baseline": {
    "config": {
      "track_routing": false,
      "callback_placement": false,
      "placement_step_interval": null,
      "description": "..."
    },
    "timings": [
      {
        "dp_rank": 0,
        "scenario": "baseline",
        // Throughput
        "generate_time_s": 12.3,
        "tokens_generated": 256,
        "tok_per_sec": 20.8,
        "num_prompts": 4,
        // EPLB
        "num_steps": 0,
        "num_eplb_events": 0,
        "placement_step_interval": null,
        "track_routing": false,
        // Online-inference latencies (milliseconds, null if unavailable)
        "ttft_mean_ms":  142.1,
        "ttft_p50_ms":   138.5,
        "ttft_p99_ms":   195.3,
        "tpot_mean_ms":  22.4,
        "tpot_p50_ms":   21.8,
        "tpot_p99_ms":   28.1,
        "e2e_mean_ms":   5500.2,
        "e2e_p50_ms":    5410.0,
        "e2e_p99_ms":    5980.3
      },
      // ... one entry per rank (0–7)
    ],
    "aggregate": {
      // Throughput
      "tok_per_sec_mean":     20.8,
      "tok_per_sec_std":      1.2,
      "generate_time_mean":   12.3,
      "generate_time_std":    0.4,
      "total_tokens":         2048,
      // EPLB
      "num_steps_mean":       0,
      "num_eplb_events_mean": 0,
      "num_ranks_collected":  8,
      // Latencies — averaged across ranks (null if unavailable)
      "ttft_mean_ms":  142.1,
      "ttft_p50_ms":   138.5,
      "ttft_p99_ms":   195.3,
      "tpot_mean_ms":  22.4,
      "tpot_p50_ms":   21.8,
      "tpot_p99_ms":   28.1,
      "e2e_mean_ms":   5500.2,
      "e2e_p50_ms":    5410.0,
      "e2e_p99_ms":    5980.3
    }
  },
  "tracking": { ... },
  "interval_32": { ... }
}
```

Latency fields are `null` when:
- `num_generation_tokens == 0` (no tokens produced, e.g. timeout)
- `first_token_ts == 0.0` (stats not populated — should not happen since `disable_log_stats=False`)

---

## Plot guide

### `throughput.png`

Bar chart of mean tok/s per scenario with error bars (std across ranks). A
dashed horizontal line marks the baseline value for easy comparison. Numbers
above bars show the mean value.

**Reading it:** The gap between the baseline bar and any placement scenario bar
is the direct throughput cost of that configuration. A small gap at interval=64
and a large gap at interval=8 indicates that EPLB transfer time dominates at
high frequencies.

---

### `latency.png`

Bar chart of mean `generate()` wall time in seconds. Error bars are std across
ranks. This is the total time your script blocks waiting for `llm.generate()`.

**Reading it:** This is `1 / throughput` scaled by total tokens, so it tells
the same story as `throughput.png` but in a form more useful if you care about
batch completion time rather than token rate.

---

### `ttft.png`

Two bars per scenario (side-by-side):
- **Solid** — TTFT mean (average across requests and ranks)
- **Hatched** — TTFT p99 (worst-case across requests, averaged across ranks)

**Reading it:** A jump in p99 TTFT for placement scenarios vs. baseline indicates
that some prefill steps coincided with an EPLB rebalance. If TTFT mean is flat but
p99 is elevated, EPLB occasionally stalls the very first forward pass of a request.

---

### `tpot.png`

Two bars per scenario (side-by-side):
- **Solid** — TPOT mean (average over decode steps per request, averaged across ranks)
- **Hatched** — TPOT p99 (worst single decode step, averaged across ranks)

**This is the most diagnostic plot for EPLB overhead.**

- **Mean TPOT rising** as interval decreases = EPLB overhead spreading across
  all decode steps because more steps include a stall.
- **p99 TPOT spiking** = one decode step per N steps is slow (the EPLB step).
  On PCIe L4 hardware, p99 TPOT for `interval_8` might be 2–5 s vs.
  a mean TPOT of 50–100 ms. This is the "invisible tax" on online serving:
  most tokens arrive quickly, but one token per rebalance interval is delayed
  by the full weight-transfer duration.

---

### `interval_sweep.png`

Line chart: tok/s on the y-axis, placement-step-interval on the x-axis (log₂
scale so that 64 → 32 → 16 → 8 are evenly spaced). Dashed horizontal lines
mark the baseline and tracking-only throughputs for reference.

**Reading it:**
- A flat line = EPLB cost is negligible even at low intervals (expected on
  NVLink hardware where a full Mixtral rebalance takes ~25 ms).
- A steeply falling line = EPLB transfers dominate throughput at low intervals
  (expected on PCIe-only L4 VMs where each rebalance costs 1–5 s).
- The crossing point where the line meets the "tracking-only" baseline is the
  interval below which EPLB overhead begins to hurt throughput.

---

### `relative_overhead.png`

Bar chart of throughput as a percentage of the baseline. Baseline = 100%.
A dashed line marks 100% for easy reading.

**Reading it:** A value of 90% for `interval_32` means you pay a 10% throughput
penalty for running EPLB every 32 steps. This is the "cost" side of the
trade-off — the "benefit" side (better load balance → faster expert compute) is
not captured here, because the current benchmark does not distinguish hot vs.
cold experts in the workload.

---

## How metrics are collected

Each scenario spawns 8 subprocesses (one per DP rank) with fresh CUDA contexts.
Inside each subprocess:

1. **Model loads** (not timed).
2. **`llm.generate(prompts, ...)`** is timed with `time.perf_counter()`.
3. All EPLB events, NCCL all_reduces, and routing captures happen *inside*
   `generate()`, so they are fully included in the timing.
4. After `generate()`, `RequestOutput.metrics` fields are read for each request
   to compute TTFT, TPOT, and E2E latency.

Latency metrics come from vLLM's internal `RequestStateStats` object, enabled by
passing `disable_log_stats=False` to `LLM()`. The timestamps use two clocks:
- `first_token_latency` — wall-clock `time.time()` delta (frontend process)
- `first_token_ts` / `last_token_ts` — monotonic timestamps from the engine core
  process (valid to subtract from each other; do not mix with `arrival_time`)

---

## Hardware notes and expected numbers

### 8× NVIDIA L4 (PCIe-only, cloud VM)

One Mixtral-8x7B EPLB full rebalance (8 experts, all moved) costs ~1–5 s on
PCIe Gen 4 in a virtualised environment. Expected results:

| Scenario | tok/s | TPOT mean | TPOT p99 |
|---|---|---|---|
| baseline | ~40–60 | ~20 ms | ~30 ms |
| tracking | ~38–58 | ~22 ms | ~35 ms |
| interval_64 | ~35–55 | ~25 ms | **~2 s** |
| interval_32 | ~30–50 | ~30 ms | **~2 s** |
| interval_8 | ~20–35 | ~80 ms | **~2 s** |
| interval_1 | very low | >1 s | **~2–5 s** |

The TPOT p99 is roughly constant across placement scenarios (≈ cost of one
EPLB transfer), while mean TPOT rises because more steps include a stall.

### 8× NVIDIA RTX Pro 6000 Blackwell (NVLink)

One Mixtral EPLB rebalance takes ~25 ms over NVLink 5th-gen.

| Scenario | tok/s | TPOT mean | TPOT p99 |
|---|---|---|---|
| baseline | ~200+ | <5 ms | <8 ms |
| interval_8 | ~195+ | <10 ms | **~25 ms** |
| interval_1 | ~180+ | ~30 ms | **~25 ms** |

On NVLink hardware `interval_1` becomes practical — the penalty is a single
25 ms stall every decode step instead of a 2–5 s stall.

---

## Re-plotting from saved results

If you already have a `results.json` from a previous run, you can re-generate
plots without re-running inference:

```python
import json, sys
sys.path.insert(0, "genome_scripts")
from benchmark_placement import plot_results, ALL_SCENARIOS

with open("benchmark_results/results.json") as f:
    results = json.load(f)

plot_results(results, "benchmark_results", ALL_SCENARIOS)
```

---

## Adding new scenarios

Edit the `ALL_SCENARIOS` `OrderedDict` at the top of `benchmark_placement.py`.
Each entry needs:

```python
"my_scenario": {
    "track_routing": True,        # Set VLLM_TRACK_ROUTING=1 in subprocesses
    "callback_placement": True,   # Enable EPLB with compute_placement() callback
    "placement_step_interval": 4, # Fire EPLB every N forward passes
    "description": "...",
    "color": "#RRGGBB",           # Matplotlib bar color
    "label": "My\nScenario",      # Axis label (use \n for line breaks)
},
```

To test a custom `compute_placement()` function, edit `placement_fns.py` and
run with `--scenarios baseline my_scenario` to compare against the no-op baseline.
