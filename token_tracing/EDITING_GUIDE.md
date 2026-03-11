# Token Inference Tracking

This workspace is centered on one active pipeline: capture MoE token routing from a local vLLM checkout during offline inference, keep the routing data available in memory, optionally save consolidated `.pt` files per data-parallel rank, and analyze the results with heatmap and Sankey visualizations.

The source of truth for the current workflow is:

- `vllm/token_tracing/data_parallel_with_tracking.py`
- `vllm/token_tracing/analyze_routing_paths.py`
- `vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/vllm/v1/worker/worker_base.py`
- `vllm/vllm/entrypoints/llm.py`

Most other tracing files in this repo are exploratory or older prototypes.

## High-Level Flow

The routing capture path works like this:

1. `data_parallel_with_tracking.py` starts one Python process per DP rank and runs `LLM.generate(...)`.
2. During each MoE forward pass, `fused_topk_router.py` captures `topk_ids`, `topk_weights`, and `num_tokens` for the current layer.
3. `layer.py` stores those captures in the process-local `_ROUTING_DATA` dictionary.
4. `worker_base.py` converts GPU tensors to CPU tensors and serializes them to raw bytes with `torch.save(...)` so vLLM RPC does not corrupt tensor payloads.
5. `llm.py` gathers the serialized payloads from all DP workers, deserializes them, and returns proper CPU tensors via `get_routing_data_distributed()`.
6. `data_parallel_with_tracking.py` consolidates the captures into dense tensors of shape `[num_tokens, num_layers, top_k]` and optionally saves one `.pt` file per rank.
7. `analyze_routing_paths.py` reads the `.pt` files and produces the heatmap and Sankey outputs.

## Important Files

### Your main script

#### `vllm/token_tracing/data_parallel_with_tracking.py`

This is the main entry point you should edit first.

What it does:

- Parses runtime flags such as model name, DP/TP sizes, prompt count, and output path.
- Loads and formats a small MMLU subset.
- Splits prompts across DP ranks.
- Runs generation with `LLM.generate(...)`.
- Calls `llm.clear_routing_data()` before generation.
- Calls `llm.get_routing_data_distributed()` after generation.
- Consolidates per-layer captures into `expert_ids` and `expert_weights` tensors.
- Prints diagnostics and saves per-rank `.pt` files.

Edit this file when you want to change:

- Which prompts or dataset are used.
- How many prompts each rank processes.
- Sampling parameters such as `max_tokens`, `temperature`, and `top_p`.
- What metadata gets saved in each `.pt` file.
- How routing statistics are summarized or printed.

The most important output tensors created here are:

- `expert_ids`: `[num_tokens, num_layers, top_k]`
- `expert_weights`: `[num_tokens, num_layers, top_k]`
- `token_ids`: `[num_tokens]`
- `origin_gpu`: `[num_tokens]`

### Analysis and visualization

#### `vllm/token_tracing/analyze_routing_paths.py`

This is the offline analysis script for saved routing files.

What it currently provides:

- Consolidation across multiple `routing_data_rank*.pt` files.
- Top-path extraction using top-1 expert routes.
- A heatmap of expert usage by layer.
- A Sankey diagram of token flow between experts across consecutive layers.

Edit this file when you want to change:

- Which routing definition you analyze, top-1 only or top-k aware.
- Path counting logic.
- Plot styling, thresholds, or saved formats.
- Rank diagnostics shown in the figure title or table.

### Baseline reference

#### `vllm/token_tracing/data_parallel.py`

This is the closest baseline or earlier copy of the distributed offline inference driver without the current routing-capture pipeline. Use it as a comparison target if you want to see what was added for tracking.

### Older or exploratory helpers

These are not the current production path:

- `vllm/token_tracing/routing_tracker.py`: older hook-based approach.
- `vllm/token_tracing/simple_tracking_example.py`: example for the hook-based approach.
- `tracer.py`, `old_tracer.py`, and some CSV files at the workspace root: earlier OLMoE experiments, useful for historical context but not part of the current vLLM-integrated path.

## Important vLLM Files

### `vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

This is where the actual routing capture happens.

Current responsibility:

- After fused top-k routing is computed, it appends a capture record into `_ROUTING_DATA[self._layer_id]`.

Current capture payload:

- `topk_ids`
- `topk_weights`
- `num_tokens`

Edit this file when you want to change:

- Which routing fields are captured.
- Whether capture happens before or after renormalization.
- Whether you store additional tensors, for example router logits or expert groups.

Important constraint:

- Do not send raw CUDA tensors through vLLM RPC. Keep local captures as tensors here, but serialize safely later in the worker layer.

### `vllm/vllm/model_executor/layers/fused_moe/layer.py`

This file owns the in-memory routing store.

Current responsibility:

- Defines `_ROUTING_DATA`.
- Exposes `_is_tracking_enabled()`.
- Exposes `get_routing_data()` and `clear_routing_data()`.

Edit this file when you want to change:

- The global in-memory storage model.
- The enable or disable condition, currently `VLLM_TRACK_ROUTING`.
- Reset semantics between runs.

Important constraint:

- This storage is process-local. Each worker has its own copy until the data is collected through the worker and `LLM` APIs.

### `vllm/vllm/v1/worker/worker_base.py`

This is the RPC safety boundary.

Current responsibility:

- `get_routing_data_serialized()` reads `_ROUTING_DATA`.
- Moves tensors from GPU to CPU.
- Uses `torch.save(...)` into `BytesIO`.
- Returns a byte payload plus lightweight metadata.

Edit this file when you want to change:

- How worker-local routing data is prepared for transport.
- Compression or alternate serialization.
- Which metadata fields travel alongside tensor bytes.

Important constraint:

- This file solves the earlier RPC corruption issue. If you change the transport format here, you must make the matching change in `llm.py`.

### `vllm/vllm/entrypoints/llm.py`

This is the user-facing collection API.

Current responsibility:

- `get_routing_data_distributed()` calls `collective_rpc("get_routing_data_serialized")`.
- Deserializes the byte payloads.
- Concatenates captures from all DP workers.
- Returns proper CPU tensors to the tracing script.
- `clear_routing_data()` clears worker-local state through RPC.

Edit this file when you want to change:

- The shape or contract returned to your tracing scripts.
- How worker results are merged across DP ranks.
- Whether captures are returned as one consolidated tensor per layer or as original per-forward-pass segments.

Important constraint:

- Keep the return contract stable for `data_parallel_with_tracking.py`. Right now it expects `routing_data[layer_id]` to be a list of capture dictionaries containing tensors and `num_tokens`.

## How To Edit `data_parallel_with_tracking.py`

If you are changing your experiment logic, this is usually the only file you should need to touch.

### Common edits

#### Change the dataset or prompt formatting

Edit:

- The `load_dataset(...)` call.
- The nested `format_mmlu(...)` function.

#### Change prompt volume per rank

Edit:

- The `--num-prompts-per-rank` argument.
- The `start_idx` and `end_idx` slicing logic if you want a different assignment scheme.

#### Change generation length while tracking

Edit:

- `max_tokens_for_tracking`.

Why it matters:

- Every generated token creates another forward pass and another routing capture per MoE layer. Increasing `max_tokens` increases routing volume quickly.

#### Change what gets saved

Edit the `torch.save({...}, save_path)` payload near the end of the file.

If you add new fields, keep them simple tensors or Python scalars so the analysis script can load them directly.

#### Change the tensor consolidation logic

Edit the loop that iterates over `sorted(routing_data.keys())` and concatenates each layer's captures with `torch.cat(...)`.

This is the right place to:

- Keep captures segmented by forward pass.
- Build token-level metadata.
- Filter layers or experts before writing to disk.

## How To Edit the vLLM Code Safely

The main rule is: edit one layer of the pipeline at a time and keep the contracts aligned.

### If you want to capture more routing information

1. Add the field in `fused_topk_router.py` when the routing decision is available.
2. Store it in `_ROUTING_DATA` without changing unrelated fields.
3. Serialize it in `worker_base.py`.
4. Deserialize and merge it in `llm.py`.
5. Consume it in `data_parallel_with_tracking.py`.

### If you want to change transport or memory behavior

Edit these files together:

- `vllm/vllm/v1/worker/worker_base.py`
- `vllm/vllm/entrypoints/llm.py`

Those two files define the transport contract. Changing only one will break collection.

### If you only want to change routing capture on or off

Edit:

- `_is_tracking_enabled()` in `layer.py`
- Or just set `VLLM_TRACK_ROUTING=1` or `0` when running.

### What does not need a rebuild

For the files above, you are only editing Python. You usually only need to:

- Restart the Python process.
- Re-run the tracing script.

You do not need a CUDA rebuild unless you start editing files under `vllm/csrc/` or other compiled components.

## Running the Current Workflow

From the workspace root:

```bash
source .venv/bin/activate
VLLM_TRACK_ROUTING=1 python vllm/token_tracing/data_parallel_with_tracking.py \
	--trust-remote-code \
	--enforce-eager \
	--save-routing-csv routing_data
```

That produces files like:

- `routing_data_rank0.pt`
- `routing_data_rank1.pt`
- `...`

Then analyze them with:

```bash
source .venv/bin/activate
python vllm/token_tracing/analyze_routing_paths.py \
	--glob "routing_data_rank*.pt" \
	--save routing_heatmap.png \
	--sankey-save routing_sankey.html
```

## Output Files You Should Care About

Generated artifacts:

- `routing_data_rank*.pt`: consolidated per-rank routing tensors.
- `routing_heatmap.png`: expert usage by layer.
- `routing_sankey.html`: token flow between experts across layers.

Useful fields inside each `.pt` file:

- `expert_ids`
- `expert_weights`
- `num_tokens`
- `num_layers`
- `top_k`
- `dp_rank`

## Practical Notes

- The token counts you see in saved `.pt` files are total routed tokens accumulated across captures, not just raw prompt length.
- With tracking enabled, generation length dominates volume quickly because each generated token creates another MoE traversal.
- If analysis breaks after a vLLM edit, first verify the contract between `worker_base.py`, `llm.py`, and `data_parallel_with_tracking.py`.
- If RPC transport starts returning corrupted arrays again, inspect the serialization boundary first rather than the analysis script.

## Recommended Edit Order

If you are changing behavior, use this order:

1. `vllm/token_tracing/data_parallel_with_tracking.py`
2. `vllm/vllm/entrypoints/llm.py`
3. `vllm/vllm/v1/worker/worker_base.py`
4. `vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`
5. `vllm/vllm/model_executor/layers/fused_moe/layer.py`
6. `vllm/token_tracing/analyze_routing_paths.py`

That order matches the direction of data flow from experiment driver, to collection, to capture internals, then back to offline analysis.
