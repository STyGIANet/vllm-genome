# Genome Refactor Notes

Goal: keep long-lived custom behavior merge-friendly against upstream `vllm`
by moving custom support code into `vllm/genome/...` and reducing inline edits
in high-churn upstream files.

Principles:

- Prefer new sibling modules under `vllm/genome/...` over embedding large
  helper classes inside upstream runtime files.
- Keep core-file edits limited to:
  - imports
  - feature selection / delegation
  - thin hook calls
- Preserve behavior and performance; avoid changing public request/response
  flow unless required.

Completed extractions in this pass:

- `vllm/genome/prefix_learning.py`
  - `PrefixLearningAsyncStepJob`
  - `PrefixLearningPendingCopyJob`
  - `AsyncPrefixLearningOwnerLearner`
- `vllm/genome/engine/load_balancing_types.py`
  - `PREFIX_ROUTER_LEARNING_TIMEOUT_S`
  - `InFlightRequestInfo`
  - `QueuedDispatchRequest`
  - `PrefixLearningWorkItem`
  - `PrefixLearningContext`
  - `PrefixRouterUpdate`
- `vllm/genome/engine/coordinator_routing.py`
  - custom coordinator-side prefix-affinity / KV-prefix / load-score routing
    state and scoring helpers
- `vllm/genome/engine/dplb_routing.py`
  - custom DPLB client routing, scoring, route-query, and learning helpers
- `vllm/genome/fused_moe/router_capture.py`
  - routed-expert capture helpers
  - EPLB expert-load recording helpers

Primary remaining extraction targets:

- `vllm/v1/engine/core_client.py`
  - move the remaining custom dispatch / output / abort / elastic-EP helpers
    into `vllm/genome/engine/...`
- `vllm/model_executor/layers/fused_moe/router/base_router.py`
  - verify whether any remaining genome-specific hooks still need separation
- `vllm/v1/worker/gpu_model_runner.py`
  - continue peeling prefix-learning and placement-capture support into
    `vllm/genome/...`

Operational rule for follow-up passes:

- When a custom block is large and mostly self-contained, move the whole block
  first.
- Only after extraction, shrink upstream files by replacing embedded logic
  with imports / delegation.
