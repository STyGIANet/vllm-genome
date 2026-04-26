# Load-Balancer Architecture

## Quick Start

The fastest way to bring up the online-inference stack in this repo is the
single-host launcher:

```bash
../run.sh deepseek-ai/deepseek-moe-16b-chat
```

In a separate shell, send requests with:

```bash
python3 send-prompts.py
# Change the following to match the server address and model used in the run.sh launch:
#HOST = "http://0.0.0.0:8000"
#MODEL = "deepseek-ai/deepseek-moe-16b-chat"
```

To evaluate placement and LB across different LB weights:
```bash
python3 weights-experiment.py # poisson arrival of prompts, modify poisson rate and lb weights range in the script
```

`../run.sh` is the current canonical launch script for this LB/EPLB setup. It
starts `vllm serve` with:

- DP=8 and expert parallel enabled
- expert-affinity, KV-prefix, and load-score routing enabled
- custom EPLB placement via `placement_fns.py`
- prefix learning enabled with `--prefix-learning-algorithm prefixtrie`
- debug LB logging enabled

For the full flag-by-flag explanation, see
[cmd_args_expert_lb.md](cmd_args_expert_lb.md).

For the multinode launcher path, see:

- [../vllm-run.sh](../vllm-run.sh)
- [../vllm-ns-worker.sh](../vllm-ns-worker.sh)

```text
                                  +------------------------------+
                                  | FastAPI / OpenAI API Server  |
                                  | vllm/entrypoints/openai/*    |
                                  +--------------+---------------+
                                                 |
                                                 v
                                  +------------------------------+
                                  | AsyncLLM / EngineClient      |
                                  | vllm/v1/engine/async_llm.py  |
                                  +--------------+---------------+
                                                 |
                         internal DP LB          |         utility / dev APIs
                         enabled                 |         (weights, EPLB step)
                                                 v
                    +------------------------------------------------------+
                    | DPLBAsyncMPClient frontend                          |
                    | vllm/v1/engine/core_client.py                       |
                    | - builds route queries                              |
                    | - keeps local LB mirrors                            |
                    | - dispatches requests to engine cores               |
                    | - consumes outputs / prefix updates                 |
                    +--------------+-----------------------+---------------+
                                   |                       |
               route query (DEALER/ROUTER)                 | request / output sockets
                                   |                       |
                                   v                       v
                    +--------------------------+   +------------------------------+
                    | DP Coordinator Process   |   | EngineCore processes         |
                    | vllm/v1/engine/          |   | vllm/v1/engine/core.py       |
                    | coordinator.py           |   | + scheduler + worker         |
                    | - combines 3 LB signals  |   |                              |
                    | - tracks global mirrors  |   |                              |
                    +--------------+-----------+   +--------------+---------------+
                                   |                              |
                      stats / prefix / KV events                  | scheduler output
                                   |                              v
                                   |                 +------------------------------+
                                   |                 | GPUModelRunner / Worker      |
                                   |                 | vllm/v1/worker/*             |
                                   |                 | - executes MoE forward       |
                                   |                 | - captures routing           |
                                   |                 | - learns prefix owners       |
                                   |                 | - steps EPLB                 |
                                   |                 +--------------+---------------+
                                   |                                |
                                   |                                v
                                   |                 +------------------------------+
                                   |                 | Fused MoE Router / Layer     |
                                   |                 | vllm/model_executor/layers/  |
                                   |                 | fused_moe/*                  |
                                   |                 | - top-k experts per token    |
                                   |                 | - placement capture          |
                                   |                 +--------------+---------------+
                                   |                                |
                                   |                                v
                                   |                 +------------------------------+
                                   |                 | EPLB / Custom Placement      |
                                   |                 | vllm/distributed/eplb/*      |
                                   |                 | - load windows               |
                                   |                 | - coactivation graph         |
                                   |                 | - physical<->logical map     |
                                   |                 | - async weight transfer      |
                                   |                 +------------------------------+
                                   |
                                   +-- prefix placement update / KV cache events / load stats
```

## 1. Scope

This document describes the **online-serving** load-balancing architecture in
this tree. The architecture has four closely related pieces:

1. **Request routing across DP engines**
2. **Prefix-affinity learning and routing**
3. **KV block-prefix routing**
4. **Expert placement / EPLB / custom placement callback**

Companion flag and launcher reference:

- [cmd_args_expert_lb.md](cmd_args_expert_lb.md)

The relevant code lives primarily in:

- `vllm/entrypoints/*` for HTTP entry and dev APIs
- `vllm/v1/engine/*` for frontend routing, coordinator, and handovers
- `vllm/v1/core/sched/*` for scheduler-side storage of LB-related outputs
- `vllm/v1/worker/*` for worker-side MoE routing capture and prefix learning
- `vllm/model_executor/layers/fused_moe/*` for actual MoE top-k routing
- `vllm/distributed/eplb/*` for placement and background expert transfers

This document is specifically about the **internal DP LB** path, i.e. the
topology where one frontend can route across multiple DP engine cores.

## 2. High-Level Pieces

### 2.1 HTTP / API entry

The online server is built in:

- `vllm/entrypoints/openai/api_server.py`
- `vllm/entrypoints/serve/__init__.py`

Main responsibilities:

- build the FastAPI app
- register standard API routers
- register dev/control routers
- create `AsyncLLM`

Important LB-related API routers:

- `vllm/entrypoints/serve/load_balancer/api_router.py`
  - `GET /load_balancer/weights`
  - `POST /load_balancer/weights`
- `vllm/entrypoints/serve/eplb/api_router.py`
  - `GET /eplb/step_interval`
  - `POST /eplb/step_interval`

These are dev-mode control surfaces gated by `VLLM_SERVER_DEV_MODE`.

### 2.2 Async engine wrapper

Main file:

- `vllm/v1/engine/async_llm.py`

Responsibilities:

- constructs `AsyncLLM`
- creates the frontend client via `EngineCoreClient.make_async_mp_client(...)`
- exposes runtime control methods to the API layer
- registers the custom placement callback on workers for online serving

LB/EPLB-specific methods:

- `maybe_register_placement_callback()`
- `get_runtime_load_balancer_weights()`
- `update_runtime_load_balancer_weights()`
- `get_runtime_eplb_step_interval()`
- `update_runtime_eplb_step_interval()`

It also sets:

- `VLLM_CAPTURE_ROUTING_FOR_PLACEMENT=1` when `placement_callback_path` is set

That environment variable turns on worker-side MoE routing capture for
placement/EPLB.

### 2.3 Frontend client

Main file:

- `vllm/v1/engine/core_client.py`

The internal-DP-LB frontend implementation is:

- `DPLBAsyncMPClient`

This class is the central online LB frontend. It:

- receives engine/coordinator addresses during launch
- sends route queries to the coordinator
- optionally falls back to local scoring if coordinator routing fails
- dispatches requests to engine cores
- tracks in-flight requests
- maintains local mirrors for expert-affinity and KV-prefix routing
- consumes worker/scheduler prefix-learning outputs
- sends prefix updates to the coordinator

### 2.4 DP coordinator

Main file:

- `vllm/v1/engine/coordinator.py`

Main classes:

- `DPCoordinator`
- `DPCoordinatorProc`

This process is the global control-plane component for internal DP LB. It:

- tracks per-engine load stats
- receives route queries from frontends
- computes weighted route scores
- maintains global prefix-affinity mirrors
- maintains global exact KV block-prefix mirrors
- rebroadcasts placement updates
- coordinates DP wave wakeups

### 2.5 Engine core / scheduler / worker

Main files:

- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/outputs.py`
- `vllm/v1/engine/__init__.py`

This is the execution side. The scheduler and worker are where:

- routed experts are captured
- prefix-learning owners/pairs are produced
- KV cache events are emitted
- placement map changes are detected and exported

### 2.6 MoE router / placement capture

Main files:

- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

This is where per-token MoE top-k experts are chosen and, when enabled,
captured into the per-step routing buffer used by:

- offline routing tracking
- placement/EPLB capture
- worker-side prefix learning

### 2.7 EPLB / custom placement

Main files:

- `vllm/distributed/eplb/eplb_state.py`
- `vllm/distributed/eplb/async_worker.py`
- `vllm/distributed/eplb/policy/custom_policy.py`

This layer:

- accumulates expert-load windows
- optionally accumulates coactivation edges for graph-based placement
- computes a new `physical_to_logical_map`
- asynchronously transfers expert weights
- exposes placement epoch changes back upward for prefix owner-cache refresh

## 3. End-to-End Request Flow

### 3.1 HTTP request enters the server

1. FastAPI app is built in `vllm/entrypoints/openai/api_server.py`.
2. `build_async_engine_client()` creates `AsyncLLM`.
3. `AsyncLLM` creates an `EngineCoreClient`.
4. For internal DP LB, `EngineCoreClient.make_async_mp_client(...)` selects
   `DPLBAsyncMPClient`.

Relevant files:

- `vllm/entrypoints/openai/api_server.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core_client.py`

### 3.2 Request becomes `EngineCoreRequest`

`AsyncLLM` uses:

- `InputProcessor`
- `OutputProcessor`

and converts user input into:

- `EngineCoreRequest`

Relevant types/files:

- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/async_llm.py`

### 3.3 Frontend chooses a DP engine

The frontend client either:

- asks the coordinator for a route decision, or
- computes a local fallback score

Coordinator query path:

- `_build_coordinator_route_request()`
- `_select_core_engine_for_request_async()`

Fallback local path:

- `_select_core_engine_for_request()`

Relevant file:

- `vllm/v1/engine/core_client.py`

### 3.4 Request is dispatched

After engine choice:

- `_dispatch_request_to_engine(...)` sends the request to the chosen engine
- in some EP+DP cases, shadow requests are also sent to non-chosen engines

Relevant file:

- `vllm/v1/engine/core_client.py`

### 3.5 Scheduler and worker execute the batch

The worker path is:

1. `GPUModelRunner.execute_model(...)`
2. MoE router executes top-k routing
3. worker post-processes:
   - prefix-learning capture
   - routing-step hooks
   - EPLB step
4. `sample_tokens(...)` creates `ModelRunnerOutput`
5. scheduler folds that into `EngineCoreOutputs`
6. frontend client consumes `EngineCoreOutputs`

Relevant files:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/outputs.py`
- `vllm/v1/engine/__init__.py`

## 4. The Three LB Signals

The current online LB combines three independent score families:

1. **Expert-affinity score**
2. **KV block-prefix score**
3. **Load score**

The weighted sum is then tie-broken by current load.

### 4.1 Expert-affinity routing

#### Purpose

Predict which DP rank is best for a prompt based on previous prompts whose
prefill MoE routing was observed.

#### Current online index

Current `prefix_learning_algorithm` is:

- `prefixtrie`

Implementation:

- `vllm/v1/prefix_router.py`

Current score:

- `longest_prefix_length(prompt_token_ids) / total_prompt_tokens`

#### Learning path

Worker-side learning starts from actual MoE routing.

Flow:

1. MoE router captures `topk_ids` for prompt tokens.
2. Worker stores prompt-only routing traces for the current step.
3. A background prefix learner reduces those traces into per-request owner
   counts using the current owner cache derived from EPLB placement.
4. Worker emits either:
   - immediate `prefix_learning_owner_by_req`
   - or late `async_prefix_learning_owner_by_req`
   - or fallback `prefix_learning_pairs_by_req`
5. Scheduler stores these in request-local dictionaries.
6. Frontend applies local prefix updates and batches them to coordinator.

Worker-side files:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
- `vllm/model_executor/layers/fused_moe/layer.py`

Scheduler/output files:

- `vllm/v1/outputs.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/__init__.py`

Frontend/coordinator files:

- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/prefix_router.py`

#### What is actually learned

Current path still learns a **target rank** per prompt, not a logical-expert
histogram. The owner computation uses:

- `build_owner_cache_from_physical_to_logical_map(...)`
- `compute_owner_from_layer_expert_pairs(...)`
- `compute_owner_from_routed_experts(...)`

in:

- `vllm/v1/prefix_router.py`

That means expert-affinity is currently:

- prompt tokens -> learned rank label -> prefix trie retrieval

not:

- prompt tokens -> predicted expert histogram -> live placement projection

#### Frontend update path

When prefix learning produces a rank owner:

1. frontend keeps prompt tokens in `prefix_learning_context_by_request`
2. `process_engine_outputs(...)` extracts owner updates
3. `_apply_prefix_router_owner_update_batch(...)` updates the local mirror
4. `_send_prefix_router_update_batch(...)` sends batched updates to coordinator

Relevant file:

- `vllm/v1/engine/core_client.py`

#### Coordinator mirror

Coordinator receives:

- `PREFIX_ROUTER_UPDATE`
- `PREFIX_ROUTER_UPDATE_BATCH`

and updates its own expert-affinity mirrors.

Relevant file:

- `vllm/v1/engine/coordinator.py`

#### Routing-time score lookup

Frontend fallback:

- `DPLBAsyncMPClient._get_expert_affinity_scores(...)`

Coordinator:

- `DPCoordinatorProc._get_expert_affinity_scores(...)`

Both use the same mirror structure and the same `index.score(prompt_token_ids)`.

### 4.2 KV block-prefix routing

#### Purpose

Route requests toward engines that already hold a matching cached KV prefix.

#### Data structure

Per-rank exact block-prefix trie/index:

- `ExactBlockPrefixIndex`

Implementation:

- `vllm/v1/prefix_router.py`

#### Update path

1. scheduler emits `KVEventBatch` in `EngineCoreOutputs.kv_cache_event_batch`
2. frontend and coordinator consume these events
3. each side updates per-rank `ExactBlockPrefixIndex`

Files:

- `vllm/v1/engine/__init__.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/prefix_router.py`

#### Routing-time score

The score is:

- `matched_blocks / total_blocks`

Frontend fallback:

- `_get_kv_block_prefix_scores(...)` in `core_client.py`

Coordinator:

- `_get_kv_block_prefix_scores(...)` in `coordinator.py`

### 4.3 Load score routing

#### Purpose

Prefer less-loaded engines and break ties between engines that have similar
expert/KV affinity.

#### Source of truth

Per-engine request counts:

- `waiting`
- `running`

These come from `scheduler_stats` in `EngineCoreOutputs`.

Files:

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/coordinator.py`
- `vllm/v1/engine/core_client.py`

#### Score

Raw load:

- `waiting * 4 + running`

Normalized load score:

- inverse-minmax normalization to `[0, 1]`

Implementations:

- `DPLBAsyncMPClient._get_load_scores(...)`
- `DPCoordinatorProc._get_load_scores(...)`

#### Tie-break

After weighted expert/KV/load scores are combined, the system still resolves
ties by current load:

- frontend fallback: `_choose_engine_by_load(...)` in `core_client.py`
- coordinator: `_choose_engine_by_load(...)` in `coordinator.py`

## 5. How Final Routing Decision Is Computed

The route decision is:

1. compute enabled score maps
2. scale each by its runtime weight
3. sum them per engine index
4. find the maximum score
5. among tied best engines, pick least loaded engine

Coordinator implementation:

- `DPCoordinatorProc._route_request(...)`

Frontend fallback implementation:

- `DPLBAsyncMPClient._select_core_engine_for_request(...)`

Relevant request type:

- `CoordinatorRouteRequest` in `vllm/v1/engine/__init__.py`

Route query builder:

- `DPLBAsyncMPClient._build_coordinator_route_request(...)`

## 6. Coordinator / Frontend / Engine Handovers

### 6.1 Address handoff and process launch

Launch path:

- `vllm/v1/engine/utils.py`

Important types:

- `EngineZmqAddresses`
- `launch_core_engines(...)`

The launch utility:

- starts engine processes
- optionally starts `DPCoordinator`
- returns:
  - engine input/output socket addresses
  - coordinator publish/query addresses

### 6.2 Frontend socket roles

Main setup lives in:

- `vllm/v1/engine/core_client.py`

Important addresses:

- `input_socket`: frontend -> engine requests
- `output_socket`: engine -> frontend outputs
- `stats_update_address`: coordinator publish socket for stats/prefix/KV updates
- `route_query_address`: coordinator route query socket

### 6.3 Coordinator socket roles

Main setup lives in:

- `vllm/v1/engine/coordinator.py`

Sockets:

- XPUB for publishing stats/updates to frontends
- ROUTER for route queries from frontends
- PULL for engine outputs
- XPUB for engine wave coordination

## 7. MoE Routing Capture and Worker Learning

### 7.1 Where experts are chosen

Actual MoE top-k routing happens in:

- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

Key point:

- router computes `topk_ids` and `topk_weights`

### 7.2 Routing capture buffers

Global step-local capture structures live in:

- `vllm/model_executor/layers/fused_moe/layer.py`

Important globals/functions:

- `_ROUTING_DATA`
- `_STEP_ROUTING_SNAPSHOTS`
- `_is_tracking_enabled()`
- `_is_placement_capture_enabled()`
- `get_routing_data()`
- `clear_routing_data()`
- `push_step_snapshot()`

### 7.3 What is captured

Each MoE layer can append records like:

- `topk_ids`
- `topk_weights` when full tracking is enabled
- `num_tokens`
- `prefill_ranges`
- `prefill_req_ids`

Capture path:

- `FusedTopKRouter._compute_routing(...)`

### 7.4 Worker-side routing step

`GPUModelRunner._on_routing_step()` is called:

- after each model forward pass
- immediately before `eplb_step()`

Current responsibilities:

1. drain prefix-learning copy jobs
2. accumulate prefix-learning step capture
3. optionally build coactivation edges for callback-graph placement
4. optionally push step snapshots for offline tracking
5. clear routing data

Implementation:

- `vllm/v1/worker/gpu_model_runner.py`

### 7.5 Prefix-learning async learner

Current worker-side prefix learning is partially decoupled from the hot path.

Structures:

- `_PrefixLearningAsyncStepJob`
- `_PrefixLearningPendingCopyJob`
- `_AsyncPrefixLearningOwnerLearner`

Implementation:

- `vllm/v1/worker/gpu_model_runner.py`

Current shape:

1. router hook captures prompt-only `topk_ids`
2. worker enqueues CPU-copy jobs for the prompt-only traces
3. background learner aggregates owner counts by request
4. frontend later receives owner updates through normal engine outputs

This is a real decoupled path, but it still depends on worker-side capture.

## 8. Placement / EPLB / Custom Callback

### 8.1 Core EPLB state

Main file:

- `vllm/distributed/eplb/eplb_state.py`

This manages:

- expert-load sliding windows
- coactivation edge accumulation
- current `physical_to_logical_map`
- current `logical_to_physical_map`
- async transfer bookkeeping

### 8.2 Custom placement callback

Main file:

- `vllm/distributed/eplb/policy/custom_policy.py`

The custom policy class is:

- `StaticPlacementPolicy`

Despite the name, this class now supports:

- one-shot dynamic config updates
- graph metadata injection
- custom callback-based placement generation

Important methods/state:

- `set_compute_placement_callback(...)`
- `set_graph_metadata(...)`
- `snapshot_for_rebalance()`
- `requires_callback_graph()`
- `get_prefix_router_epoch()`
- `_build_map_from_config(...)`

### 8.3 Placement callback registration

Online registration path:

1. server starts `AsyncLLM`
2. `AsyncLLM.maybe_register_placement_callback()` validates config
3. it calls `collective_rpc("register_placement_callback", ...)`
4. workers register the callback for the custom policy

Main file:

- `vllm/v1/engine/async_llm.py`

### 8.4 Rearrangement flow

`EplbState.rearrange(...)`:

1. maps physical load windows into logical-expert load windows
2. all-reduces those windows across EP ranks
3. optionally all-reduces coactivation edge vectors
4. calls the selected EPLB policy to compute a new
   `physical_to_logical_map`
5. either:
   - commits synchronously
   - or prepares async transfer state and wakes async worker

Main file:

- `vllm/distributed/eplb/eplb_state.py`

### 8.5 Async transfer worker

Background transfer thread:

- `vllm/distributed/eplb/async_worker.py`

Responsibilities:

- wait for rearrangement event
- run callback-based map computation in worker thread only when needed
- transfer one layer at a time into buffer
- set `ep_buffer_ready`

### 8.6 Placement update flowing back upward

Worker side:

- `GPUModelRunner._take_prefix_router_placement_update()`

This emits:

- `epoch`
- `physical_to_logical_map`

through:

- `ModelRunnerOutput.prefix_router_placement_update`
- scheduler
- `EngineCoreOutputs.prefix_router_placement_update`

Frontend and coordinator both consume it and refresh their owner caches.

Files:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/outputs.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/coordinator.py`

## 9. Prefix Learning vs Placement: Current Boundary

Current boundary is:

- prefix-learning **mirrors** are token-side (`prefixtrie`)
- owner computation is **placement-aware** because it uses the live
  `physical_to_logical_map`
- placement epoch changes clear prefix-affinity state and refresh owner caches

Current owner-cache builder:

- `build_owner_cache_from_physical_to_logical_map(...)`

File:

- `vllm/v1/prefix_router.py`

Current invariant:

- prefix-learning state is only trusted within a stable placement epoch

That is why:

- frontend/coordinator clear prefix-affinity mirrors on placement epoch change
- worker clears in-flight prefix-learning state on owner-map change

## 10. Runtime Control APIs

### 10.1 LB weights

HTTP router:

- `vllm/entrypoints/serve/load_balancer/api_router.py`

AsyncLLM methods:

- `get_runtime_load_balancer_weights()`
- `update_runtime_load_balancer_weights()`

Frontend methods:

- `get_runtime_load_balancer_weights()`
- `update_runtime_load_balancer_weights()`

These change runtime weights for:

- `expert_affinity_routing_weight`
- `kv_block_prefix_routing_weight`
- `load_score_routing_weight`

### 10.2 EPLB step interval

HTTP router:

- `vllm/entrypoints/serve/eplb/api_router.py`

AsyncLLM methods:

- `get_runtime_eplb_step_interval()`
- `update_runtime_eplb_step_interval()`

Worker control-plane support:

- `vllm/v1/worker/worker_base.py`

## 11. Current Implementation Summary by Concern

### Entry / API

- `vllm/entrypoints/openai/api_server.py`
- `vllm/entrypoints/serve/__init__.py`
- `vllm/entrypoints/serve/load_balancer/api_router.py`
- `vllm/entrypoints/serve/eplb/api_router.py`

### Frontend client / request routing

- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/utils.py`
- `vllm/v1/engine/__init__.py`

### Coordinator

- `vllm/v1/engine/coordinator.py`

### Scheduler / outputs

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/outputs.py`
- `vllm/v1/engine/__init__.py`

### Worker / model runner

- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/worker_base.py`

### MoE routing / capture

- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/v1/prefix_router.py`

### Placement / EPLB

- `vllm/distributed/eplb/eplb_state.py`
- `vllm/distributed/eplb/async_worker.py`
- `vllm/distributed/eplb/policy/custom_policy.py`

## 12. Practical Reading Order

If you want to understand the system quickly, read in this order:

1. `vllm/entrypoints/openai/api_server.py`
2. `vllm/v1/engine/async_llm.py`
3. `vllm/v1/engine/core_client.py`
4. `vllm/v1/engine/coordinator.py`
5. `vllm/v1/engine/__init__.py`
6. `vllm/v1/outputs.py`
7. `vllm/v1/core/sched/scheduler.py`
8. `vllm/v1/worker/gpu_model_runner.py`
9. `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
10. `vllm/v1/prefix_router.py`
11. `vllm/distributed/eplb/eplb_state.py`
12. `vllm/distributed/eplb/policy/custom_policy.py`

## 13. Current Design Reality

Today, the routing system is:

- **request routing** across DP engines
- using three score families
- where expert-affinity is learned online from actual MoE routing
- but retrieved via a token-side predictor (`prefixtrie`)
- and where EPLB placement changes continuously refresh the owner map beneath it

That means the architecture is already split into:

- **prediction side**: prompt-token mirrors
- **ground-truth side**: actual routed experts and live placement

Any future redesign of expert-affinity should respect this split:

- prompt-side predictor
- placement-aware owner projection
- clear epoch boundaries when placement changes
