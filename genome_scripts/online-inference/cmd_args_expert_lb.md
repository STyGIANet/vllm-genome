# Online Inference Command Args

This is the companion flag reference for the online-serving LB architecture in
[README.md](README.md).

It explains the launch scripts in `genome_scripts/` and the LB / EPLB flags
that matter for online inference.

## Launcher Files

The current launchers are:

- [../run.sh](../run.sh)
  - canonical single-host launch for the current LB experiments
- [../vllm-run.sh](../vllm-run.sh)
  - multinode `mpirun` wrapper
- [../vllm-ns-worker.sh](../vllm-ns-worker.sh)
  - per-node worker launcher used by `vllm-run.sh`

## Quick Start

Single-host:

```bash
bash ../run.sh deepseek-ai/deepseek-moe-16b-chat
```

Multinode:

```bash
bash ../vllm-run.sh deepseek-ai/deepseek-moe-16b-chat
```

The multinode path depends on local cluster tooling such as `mpirun` and
`nswrap`, plus the host list baked into `vllm-run.sh`.

## Current Canonical Single-Host Recipe

`../run.sh` currently enables:

- `--tensor-parallel-size 1`
- `--data-parallel-size 8`
- `--enable-expert-parallel`
- `--all2all-backend deepep_high_throughput`
- `--api-server-count 1`
- `--enable-prefix-affinity-routing`
- `--prefix-affinity-only-prefill`
- `--prefix-learning-algorithm prefixtrie`
- `--enable-kv-block-prefix-routing`
- `--enable-load-score-routing`
- `--expert-affinity-routing-weight 1`
- `--kv-block-prefix-routing-weight 0.5`
- `--load-score-routing-weight 0.5`
- `--enable-eplb`
- `--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0}'`
- `--placement-callback-path ${PLACEMENT_PATH}`
- `--placement-callback-func compute_placement`
- `--max-pending-requests-per-engine 256`
- `--load-balancer-debug`

So `run.sh` is the best source of truth for the currently favored online setup.

## Flag Groups

All of the LB-specific CLI flags are defined in:

- `vllm/config/model.py`
- `vllm/engine/arg_utils.py`

### 1. Expert-Affinity Routing

- `--enable-prefix-affinity-routing`
  - turns on expert-affinity learning and expert-score routing
- `--prefix-affinity-only-prefill`
  - learns from prompt prefill only, instead of waiting until request finish
- `--prefix-learning-algorithm prefixtrie`
  - selects the prompt-side retrieval algorithm
  - the current tree only supports `prefixtrie`
- `--expert-affinity-routing-weight`
  - weight of the expert-locality term in the coordinator score
- `--prefix-affinity-learning-queue-size`
  - bounds the frontend fallback learning queue
  - owner-known updates now bypass most of this path, but the queue still
    matters for fallback learning work

### 2. KV Prefix Routing

- `--enable-kv-block-prefix-routing`
  - enables exact KV block-prefix scoring
- `--kv-block-prefix-routing-weight`
  - weight of the KV-locality term in the combined score

### 3. Load Score Routing

- `--enable-load-score-routing`
  - enables normalized load-based scoring
- `--load-score-routing-weight`
  - weight of the load term in the combined score

The load score is based on the same signal used in the docs and code path:

- `waiting * 4 + running`

### 4. Admission Control

- `--max-pending-requests-per-engine`
  - caps how many requests may be pending for a DP engine before frontend
    dispatch waits

This is one of the main guardrails against severe per-engine imbalance.

### 5. EPLB and Custom Placement

- `--enable-eplb`
  - enables EPLB
- `--eplb-config`
  - JSON config for EPLB, including:
    - `policy`
    - `use_async`
    - `step_interval`
    - `window_size`
    - `num_redundant_experts`
- `--placement-callback-path`
  - Python file that exports the placement callback
- `--placement-callback-func`
  - symbol name to load from that file

Current scripts point this at:

- `genome_scripts/expert-placement/placement_fns.py`

and use:

- `compute_placement`

### 6. Routed-Expert Return Path

- `--enable-return-routed-experts`
  - legacy / diagnostic routed-expert return path
  - not part of the current `run.sh` recipe

### 7. Debugging and Profiling

- `--load-balancer-debug`
  - enables detailed route, prefix-learning, and placement debug logs

This is useful when diagnosing correctness, but it can distort performance.

## Multinode-Specific Notes

`../vllm-run.sh` and `../vllm-ns-worker.sh` add the cluster launch layer:

- `mpirun` across hosts
- `RANK`, `WORLD_SIZE`, `HEAD_RANK`, and `HEAD_IP`
- `--data-parallel-address`
- `--data-parallel-rpc-port`
- `--data-parallel-size-local 1`
- `--data-parallel-start-rank ${RANK}`
- `--headless` on non-head ranks

`vllm-run.sh` also forwards:

- `VLLM_EXTRA_ARGS`
- `VLLM_SKIP_DEEPEP_PROFILE`
- `VLLM_SKIP_DEEPEP_WARMUP`
- `VLLM_SKIP_DEEPEP_DUMMY_BATCH`
- `CUDA_LAUNCH_BLOCKING`

One important current difference:

- `../run.sh` explicitly sets `num_redundant_experts` to `0`
- `../vllm-ns-worker.sh` does not currently hardcode that field in its default
  `--eplb-config`

If you want exact parity between the single-host and multinode launch paths,
pass the matching EPLB config through `VLLM_EXTRA_ARGS`.

## Code Path Map

### Prefix Learning / Expert-Affinity Index

- frontend request entry:
  - `vllm/v1/engine/core_client.py`
- coordinator scoring:
  - `vllm/v1/engine/coordinator.py`
- scheduler-side prefix outputs:
  - `vllm/v1/core/sched/scheduler.py`
- worker-side routed-expert capture and owner computation:
  - `vllm/v1/worker/gpu_model_runner.py`
- expert-affinity index and owner-cache helpers:
  - `vllm/v1/prefix_router.py`

### Routing Capture to Placement / METIS

- fused MoE top-k capture:
  - `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
- per-step filtering / graph build:
  - `vllm/v1/worker/gpu_model_runner.py`
- EPLB graph accumulation and async rearrangement:
  - `vllm/distributed/eplb/eplb_state.py`
- custom placement policy:
  - `vllm/distributed/eplb/policy/custom_policy.py`
- callback implementation used by scripts:
  - `genome_scripts/expert-placement/placement_fns.py`

### KV Prefix Routing

- scheduler KV cache events:
  - `vllm/v1/core/sched/scheduler.py`
- exact KV prefix index:
  - `vllm/v1/prefix_router.py`
- KV scoring:
  - `vllm/v1/engine/coordinator.py`

### Load Score Routing

- local fast path and frontend mirrors:
  - `vllm/v1/engine/core_client.py`
- combined coordinator score:
  - `vllm/v1/engine/coordinator.py`
