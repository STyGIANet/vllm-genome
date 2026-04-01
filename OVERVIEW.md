# moe-merged branch — System Overview

## What this branch does

Two capabilities run together on top of vLLM's DP+EP stack:

| Feature | What it does |
|---|---|
| **Token routing tracking** | Captures `{topk_ids, topk_weights, num_tokens}` for every token, every MoE layer, every forward pass |
| **JSON-driven expert placement** | `StaticPlacementPolicy` reads a config file and physically moves expert weights via NCCL P2P |
| **Live `compute_placement()` callback** | Routing load is all-reduced across EP ranks; callback receives global load and returns a new expert→GPU mapping applied before the next EPLB step |

---

## How it all fits together

```
Each forward pass (execute_model)
│
├─ fused_topk_router.py  ← captures {topk_ids, topk_weights, num_tokens}
│    └─ writes into _ROUTING_DATA[layer_id]   (EP-rank-aware: local tokens only)
│
├─ _on_routing_step()  (gpu_model_runner.py)
│    │
│    ├─ get_routing_data()          ← read this step's captures
│    ├─ _aggregate_routing_load()   ← build load[layers, experts] tensor
│    │    ├─ uses eplb_state for shape if routing_snapshot is empty
│    │    ├─ ep_group.all_reduce(load)  ← NCCL barrier: ALL 8 ranks call this
│    │    │                               every step, even with a zero tensor
│    │    └─ returns {} if global sum == 0 (no real tokens → skip callback)
│    │
│    ├─ compute_placement(global_snapshot)  ← user callback (placement_fns.py)
│    │    └─ greedy bin-packing on expert_load → {"expert_to_gpu": {...}}
│    │
│    ├─ StaticPlacementPolicy.set_dynamic_config(placement)
│    ├─ push_step_snapshot()        ← archive for drain_step_snapshots()
│    └─ clear_routing_data()        ← reset for next forward pass
│
└─ eplb_step()  (fires every --placement-step-interval forward passes)
     │
     ├─ EplbState.step()  →  EplbState.rearrange()
     │    ├─ policy.rebalance_experts()  ← StaticPlacementPolicy
     │    │    └─ _build_map_from_config()
     │    │         ├─ if _dynamic_config set → use it (from compute_placement)
     │    │         └─ else → read VLLM_EXPERT_CONFIG_PATH JSON, advance _step
     │    │
     │    ├─ rearrange_expert_weights_inplace()  (rebalance_execute.py)
     │    │    └─ torch.distributed.batch_isend_irecv()  ← NCCL P2P
     │    │
     │    ├─ torch.cuda.synchronize()    ← wait for all P2P transfers
     │    └─ _commit_eplb_maps()         ← update routing maps
     │
     └─ Next forward pass uses new expert locations
```

---

## Correctness invariants

**Placement consistency** — `compute_placement()` must return an identical
mapping on all 8 EP ranks. Guaranteed by the `all_reduce` in
`_aggregate_routing_load()`: every rank gets the same summed load tensor and
therefore computes the same placement. If ranks diverge the NCCL P2P sends/recvs
in `eplb_step()` will mismatch and deadlock.

**NCCL collective participation** — `ep_group.all_reduce()` is a barrier. Every
EP rank must call it on the same logical step, including ranks on dummy batches
with no real tokens. `_aggregate_routing_load()` always calls `all_reduce` (with
a zero tensor when needed) and returns `{}` only *after* the collective.

**Async scheduling must be off** — vLLM auto-enables async scheduling with the
multiproc executor. This de-syncs EP ranks so they hit `all_reduce` at different
forward-pass counts → NCCL deadlock. `combined_launch.py` passes
`async_scheduling=False` to `LLM()` to disable it.

---

## Implementing `compute_placement()`

Edit `genome_scripts/placement_fns.py`. The function is called after every
forward pass (when real tokens were processed) and must return the **same dict
on every EP rank**.

### Function signature

```python
def compute_placement(routing: dict) -> dict:
    ...
```

### Input: `routing`

```
routing: dict[layer_id: int, list[capture]]
```

`layer_id` is the MoE layer index (e.g. 0–31 for Mixtral-8x7B).
Each `capture` is a dict with:

| Key | Type | Description |
|---|---|---|
| `expert_load` | `Tensor[num_experts]` | **Global** token count per expert, already all-reduced across all EP ranks. Use this for placement decisions. |
| `num_gpus` | `int` | EP group size = `dp_size × tp_size`. This is the number of GPU slots to assign experts to. |
| `topk_ids` | `Tensor[T, K]` | Expert indices chosen for this rank's T tokens (K = top-k). **Local to this rank only** — do NOT use for placement (different ranks see different values). |
| `topk_weights` | `Tensor[T, K]` | Router softmax weights for each `topk_ids` entry. Local to this rank. |
| `num_tokens` | `int` | Number of tokens on this rank this step. |

### Output

```python
{"expert_to_gpu": {"<expert_id>": <gpu_id>, ...}}
```

- Keys are **string** expert IDs (e.g. `"0"`, `"1"`, ..., `"7"` for Mixtral).
- Values are **int** GPU IDs in `[0, num_gpus)`.
- Every expert must appear in the dict (partial assignments are not supported).
- Return `{}` to skip the update and keep the current placement.

### Minimal example

```python
def compute_placement(routing: dict) -> dict:
    if not routing:
        return {}
    first_cap = next(iter(routing.values()))[0]
    if 'expert_load' not in first_cap:
        return {}

    num_gpus = first_cap['num_gpus']
    num_experts = first_cap['expert_load'].shape[0]

    # Sum load across all layers
    import torch
    layer_loads = [cap['expert_load'] for caps in routing.values() for cap in caps]
    totals = torch.stack(layer_loads).sum(dim=0)  # [num_experts]

    # Assign expert i to GPU (i % num_gpus) — round-robin, ignores load
    return {"expert_to_gpu": {str(i): i % num_gpus for i in range(num_experts)}}
```

### Important constraints

1. **Must be deterministic across ranks.** Use only `expert_load` (globally
   identical after all_reduce). Do not branch on `topk_ids` or `num_tokens`
   (per-rank values that differ across ranks).

2. **Must assign all experts.** Partial dicts cause key errors in
   `StaticPlacementPolicy._build_map_from_config()`.

3. **Called every forward pass** when there are real tokens. Keep it fast.
   The result is stored and applied at the next `eplb_step()`, so latency
   here does not block inference.

4. **Return `{}` to skip.** The current JSON-driven placement is kept unchanged
   and the JSON step counter does not advance.

---

## Launch commands

All commands from `merged/vllm/` with venv active:

```bash
source merged/vllm/.venv/bin/activate
cd merged/vllm
```

### Both features (primary use case)

```bash
VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
python genome_scripts/combined_launch.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dp-size 8 --trust-remote-code --enforce-eager \
    --expert-placement-config genome_scripts/mixtral_EP_test.json \
    --placement-step-interval 32 \
    --num-prompts-per-rank 2 \
    --output-length 32
```

`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200` is needed on the current PCIe-only L4
setup. See hardware notes below.

### Tracking only (no expert movement)

```bash
VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 \
python genome_scripts/combined_launch.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dp-size 8 --trust-remote-code --enforce-eager \
    --num-prompts-per-rank 2 --output-length 32
```

### JSON-driven placement only (no live callback)

```bash
NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
python genome_scripts/combined_launch.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dp-size 8 --trust-remote-code --enforce-eager \
    --expert-placement-config genome_scripts/mixtral_EP_test.json \
    --placement-step-interval 32
```

### DeepSeek-MoE-16B (64 experts, 8 per GPU)

```bash
VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 VLLM_LOGGING_LEVEL=INFO \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
python genome_scripts/combined_launch.py \
    --model deepseek-ai/deepseek-moe-16b-chat \
    --dp-size 8 --trust-remote-code --enforce-eager \
    --expert-placement-config genome_scripts/deepseek_EP_test.json \
    --placement-step-interval 32
```

### Unit tests (no GPU needed)

```bash
python genome_scripts/test_combined.py
```

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | — | HuggingFace model ID |
| `--dp-size N` | — | DP ranks (one engine per rank) |
| `--tp-size M` | 1 | Tensor-parallel GPUs per rank |
| `--expert-placement-config <path>` | None | JSON placement config; enables EPLB |
| `--placement-step-interval N` | 32 | EPLB rebalance every N forward passes |
| `--dataset wikitext\|chatbot` | wikitext | Prompt source (see below) |
| `--cache-repeat-factor N` | 4 | Chatbot mode: repeat each base prompt N times to warm the KV prefix cache |
| `--num-prompts-per-rank N` | 3 | Prompts per DP rank |
| `--output-length N` | 10 | Max output tokens |
| `--save-routing-pt <path>` | None | Save routing tensors to `<path>_rank<N>.pt` |
| `--timeout N` | 600 | Process join timeout (seconds) |

| Env var | Description |
|---|---|
| `VLLM_TRACK_ROUTING=1` | Enable per-step routing capture |
| `NCCL_IB_DISABLE=1` | Disable InfiniBand (single-node only) |
| `VLLM_LOGGING_LEVEL=INFO` | Show EPLB placement logs |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=N` | Override 300 s default (needed on PCIe-only L4) |

---

## Datasets

| Mode | Flag | Description |
|---|---|---|
| **wikitext** | `--dataset wikitext` | Samples from WikiText-2 (train split). Filters paragraphs shorter than 100 chars; first 300 chars used as prompt. Good diversity of topics; exercises different expert specialisations across layers. |
| **chatbot** | `--dataset chatbot` | A small fixed pool of conversational prompts, each repeated `--cache-repeat-factor` times (default 4). Enables `enable_prefix_caching=True` automatically so repeated requests warm the KV cache. Use this to measure the impact of prefix cache hit rate on expert routing distributions. |

The chatbot mode is designed to test the hypothesis: *cached prefixes skip early-layer computation → different expert load profile → different optimal placement*. Compare `expert_load` distributions between a cold run and a warm-cache run to observe the effect.

---

## Placement JSON format

### Static (same mapping every rebalance)
```json
{"expert_to_gpu": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}}
```

### Multi-step cycling
```json
{"steps": [
    {"expert_to_gpu": {"0":7,"1":6,"2":5,"3":4,"4":3,"5":2,"6":1,"7":0}},
    {"expert_to_gpu": {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7}}
]}
```

### Per-layer override within a step
```json
{"steps": [
    {
        "expert_to_gpu": {"0":7,"1":6,"2":5,"3":4,"4":3,"5":2,"6":1,"7":0},
        "layer_configs": {"15": {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7}}
    }
]}
```

When `compute_placement()` returns a non-empty dict it overrides the JSON for
that one rebalance cycle; `_step` does not advance. On the next rebalance the
JSON config resumes from where it left off.

---

## Modified vLLM files

| File | What changed |
|---|---|
| `genome_scripts/combined_launch.py` | Primary launch script |
| `genome_scripts/placement_fns.py` | Default `compute_placement()` — greedy bin-packing on `expert_load` |
| `genome_scripts/test_combined.py` | 13 unit tests (no GPU needed) |
| `vllm/v1/worker/gpu_model_runner.py` | `_aggregate_routing_load()` (EP all-reduce, called every step); `_on_routing_step()` |
| `vllm/distributed/eplb/policy/custom_policy.py` | `StaticPlacementPolicy`: reads JSON, cycles steps, `set_dynamic_config()` override |
| `vllm/distributed/eplb/eplb_state.py` | `cuda.synchronize()` before `_commit_eplb_maps()`; `_step` save/restore around profile call |
| `vllm/distributed/eplb/rebalance_execute.py` | `torch.distributed.batch_isend_irecv` instead of pynccl; full all-to-all warmup in profile |
| `vllm/v1/worker/worker_base.py` | `drain_step_snapshots_serialized()`, `reset_step_counter()` RPCs |
| `vllm/model_executor/layers/fused_moe/layer.py` | `_ROUTING_DATA`, `_STEP_ROUTING_SNAPSHOTS` globals and accessors |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | EP-rank-aware token capture (avoids double-counting in DP+EP all-gather mode) |

---

## Hardware notes

### Current: 8× NVIDIA L4 (PCIe-only cloud VM)

```
VRAM:    24 GB GDDR6 per GPU (192 GB total)
NVLink:  None
Topology: 2 NUMA nodes — GPU 0–3 on NUMA 0, GPU 4–7 on NUMA 1
          Cross-NUMA P2P: ~10–15 GB/s (PCIe Gen 4, virtualised)
Time for full Mixtral EPLB (256 expert moves): ~1–2 s bare metal, longer in VM
```

Required workarounds:
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200` — VM PCIe overhead
- `--placement-step-interval 32` — amortise transfer cost across decode steps

### Prospective: 8× NVIDIA RTX Pro 6000 Blackwell

```
VRAM:    96 GB GDDR7 per GPU (768 GB total)
PCIe:    Gen 5 x16 (~50–60 GB/s)
NVLink:  5th-gen Bridge (~900 GB/s per connected pair)
```

| Scenario | Full Mixtral EPLB transfer time |
|---|---|
| Current L4 cloud VM | ~1–5 s |
| RTX Pro 6000, PCIe Gen 5 | ~0.2 s |
| RTX Pro 6000, NVLink | ~25 ms |

On RTX Pro 6000: remove `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, set
`--placement-step-interval 1` (feasible with NVLink). No code changes needed.
