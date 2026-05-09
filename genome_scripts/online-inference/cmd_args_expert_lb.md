# Online Inference Command Args

This is the companion flag reference for the online-serving LB architecture in
[../../vllm/genome/README.md](../../vllm/genome/README.md).

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

`../run.sh` is the current single-host source of truth.

What it enables today:

- `--tensor-parallel-size 1`
- `--data-parallel-size 8`
- `--data-parallel-size-local 8`
- `--nnodes 1`
- `--enable-expert-parallel`
- `--trust_remote_code`
- `--all2all-backend nccl_alltoall`
- `--max_num_batched_tokens ${MAX_NUM_BATCHED_TOKENS:-8192}`
- `--api-server-count 1`
- `--enable-eplb`
- `--eplb-config '{"use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0}'`

What it also exports in the shell:

- `NCCL_IB_DISABLE=1`
- `VLLM_SERVER_DEV_MODE=1`
- `VLLM_SKIP_DEEPEP_WARMUP=1`

What is present in `run.sh` but currently commented out:

- `--placement-callback-path ${PLACEMENT_PATH}`
- `--placement-callback-func compute_placement`
- `--max-pending-requests-per-engine 256`
- `--enable-load-score-routing`
- `--enable-kv-block-prefix-routing`
- `--enable-prefix-affinity-routing`
- `--prefix-affinity-only-prefill`
- `--prefix-learning-algorithm prefixtrie`
- `--load-balancer-debug`
- `--moe-dispatch-traffic-dump-dir ${TRAFFIC_DIR}`
- `--placement-routing-dump-dir ${TRACE_DIR}`

One important subtlety:

- `run.sh` still passes `--expert-affinity-routing-weight 1`,
  `--kv-block-prefix-routing-weight 0.5`, and
  `--load-score-routing-weight 0.5`
- these weights do nothing unless the corresponding routing modes are enabled

So the current canonical single-host recipe is a throughput-oriented NCCL
all-to-all plus EPLB baseline, not the older "all LB features on" recipe.

## Flag Groups

The CLI surface for these options is defined in:

- `vllm/engine/arg_utils.py`

The model-side defaults live in:

- `vllm/config/model.py`

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

Current `run.sh` status:

- expert-affinity weights are passed
- the mode itself is currently commented out

### 2. KV Prefix Routing

- `--enable-kv-block-prefix-routing`
  - enables exact KV block-prefix scoring
- `--kv-block-prefix-routing-weight`
  - weight of the KV-locality term in the combined score

Current `run.sh` status:

- the weight is passed
- the mode itself is currently commented out

### 3. Load Score Routing

- `--enable-load-score-routing`
  - enables normalized load-based scoring
- `--load-score-routing-weight`
  - weight of the load term in the combined score

The load score is based on the same signal used in the docs and code path:

- `waiting * 4 + running`

Current `run.sh` status:

- the weight is passed
- the mode itself is currently commented out

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

Current `run.sh` status:

- the callback path and function are currently commented out
- the canonical single-host script currently uses plain EPLB without the custom
  placement callback

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
- `--enable-elastic-ep`
- `--all2all-backend nixl_ep`

`vllm-run.sh` also forwards:

- `VLLM_EXTRA_ARGS`
- `VLLM_SKIP_DEEPEP_PROFILE`
- `VLLM_SKIP_DEEPEP_WARMUP`
- `VLLM_SKIP_DEEPEP_DUMMY_BATCH`
- `CUDA_LAUNCH_BLOCKING`

Current multinode/single-host differences worth remembering:

- `../run.sh` uses `nccl_alltoall`
- `../vllm-ns-worker.sh` uses `nixl_ep` and `--enable-elastic-ep`
- `../run.sh` explicitly sets `num_redundant_experts` to `0`
- `../vllm-ns-worker.sh` does not currently hardcode that field in its default
  `--eplb-config`

If you want exact parity between the single-host and multinode launch paths,
you need to pass the matching transport and EPLB config through
`VLLM_EXTRA_ARGS`.

## Code Path Map

### Prefix Learning / Expert-Affinity Index

- frontend request entry:
  - `vllm/v1/engine/core_client.py`
  - `vllm/genome/engine/dplb_routing.py`
- coordinator scoring:
  - `vllm/v1/engine/coordinator.py`
  - `vllm/genome/engine/coordinator_routing.py`
- scheduler-side prefix outputs:
  - `vllm/v1/core/sched/scheduler.py`
- worker-side routed-expert capture and owner computation:
  - `vllm/v1/worker/gpu_model_runner.py`
  - `vllm/genome/gpu_model_runner_prefix_learning.py`
  - `vllm/genome/gpu_model_runner_routing.py`
- expert-affinity index and owner-cache helpers:
  - `vllm/v1/prefix_router.py`
  - `vllm/genome/prefix_learning.py`

### Routing Capture to Placement / METIS

- fused MoE top-k capture:
  - `vllm/model_executor/layers/fused_moe/router/base_router.py`
  - `vllm/genome/fused_moe/router_capture.py`
- per-step filtering / graph build:
  - `vllm/v1/worker/gpu_model_runner.py`
  - `vllm/genome/gpu_model_runner_routing.py`
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
  - `vllm/genome/engine/coordinator_routing.py`

### Load Score Routing

- local fast path and frontend mirrors:
  - `vllm/v1/engine/core_client.py`
  - `vllm/genome/engine/dplb_routing.py`
- combined coordinator score:
  - `vllm/v1/engine/coordinator.py`
  - `vllm/genome/engine/coordinator_routing.py`
