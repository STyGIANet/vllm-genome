# Description
Short path map for expert-based routing, METIS placement, and serving load balancing.

# Prefill Token Expert To Radix Tree
Request arrives through serving load balancer in `vllm/v1/engine/core_client.py`.

Coordinator scoring path in `vllm/v1/engine/coordinator.py` is used when expert-affinity or KV routing is active.

With `--prefix-affinity-only-prefill`, routed experts are emitted after prefill from `vllm/v1/core/sched/scheduler.py`.

Without that flag, expert-affinity learning happens at request finish in `vllm/v1/engine/core_client.py`.

Owner GPU is computed from current expert placement in `vllm/v1/worker/worker_base.py`.

Prefix is inserted into worker radix tree in `vllm/v1/worker/worker_base.py`.

Coordinator radix mirror is updated in `vllm/v1/engine/coordinator.py`.

Radix tree data structure is in `vllm/v1/prefix_router.py`. Cause for confusion: This radix tree is not for Kv-cache aware load balancing, we use it for exper-aware load balancing.

Radix tree refresh after EPLB step uses placement epoch from `vllm/distributed/eplb/policy/custom_policy.py`.

# Token Experts To Compact Cooccurrence Graph To METIS
Per forward pass top-k expert ids are captured in `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`.

Prefill-only filtering and compact graph edge build happen in `vllm/v1/worker/gpu_model_runner.py`.

Per-forward graph increments are accumulated in `vllm/distributed/eplb/eplb_state.py`.

At EPLB rearrangement, the compact graph is synced and handed to custom policy in `vllm/distributed/eplb/eplb_state.py` and `vllm/distributed/eplb/policy/custom_policy.py`.

METIS partitioning runs in `genome_scripts/placement_fns.py`.

Placement trigger is the normal EPLB step in `vllm/distributed/eplb/eplb_state.py`.

# Expert Affinity Load Balancer
Prefix match score path is `vllm/v1/prefix_router.py` to `vllm/v1/engine/coordinator.py` to `vllm/v1/engine/core_client.py`.

Learning path is `vllm/v1/core/sched/scheduler.py` to `vllm/v1/engine/core_client.py` to `vllm/v1/worker/worker_base.py`.

# KV Prefix Load Balancer
Live KV cache events come from `vllm/v1/core/sched/scheduler.py`.

Exact KV prefix index is in `vllm/v1/prefix_router.py`.

KV prefix score is computed in `vllm/v1/engine/coordinator.py`.

# Load Score Load Balancer
Load signal is `waiting * 4 + running`.

Load-only routing stays on the local fast path in `vllm/v1/engine/core_client.py`.

When expert or KV routing is active, load score is also part of the coordinator combined score in `vllm/v1/engine/coordinator.py`.

# Serving Flags
Main flags are in `vllm/config/model.py` and `vllm/engine/arg_utils.py`.

`--enable-prefix-affinity-routing` enables expert-affinity routing.

`--prefix-affinity-only-prefill` learns expert-affinity after prefill only.

`--enable-kv-block-prefix-routing` enables exact KV prefix routing.

`--enable-load-score-routing` enables normalized load score.

`--expert-affinity-routing-weight`, `--kv-block-prefix-routing-weight`, and `--load-score-routing-weight` set the weighted sum.

`--max-pending-requests-per-engine` enables frontend-side admission queueing and caps how many requests can be pending at an engine before dispatch waits.

`--placement-callback-path` and `--placement-callback-func` register the online EPLB custom placement callback.

`--enable-return-routed-experts` is used for expert-affinity learning and is auto-enabled by prefix-affinity routing.

`--enable-eplb` and `--eplb-config` control EPLB, including `step_interval`.

`--load-balancer-debug` prints expert score, KV score, load score, final score, and routed-expert capture logs during serving.


Example run: see `vllm-run.sh` and `vllm-ns-worker.sh`
