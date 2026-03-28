# Genome Scripts

## expert_placement_example.py

This script is an offline inference example for running Mixture-of-Experts models with vLLM while wiring in vLLM's Expert Parallel Load Balancing (EPLB) path.

At a high level it does four things:

1. Parses runtime options such as model name, tensor parallel size, prompt source, and an optional expert-placement JSON file.
2. Builds prompt batches from either a small in-file prompt list or the MMLU dataset.
3. Creates a vLLM `LLM` instance with expert parallelism enabled and an `EPLBConfig` attached.
4. Calls `llm.generate(...)` and prints a small subset of the outputs.

## Execution flow

The main flow in `expert_placement_example.py` is:

1. `parse_args()` reads the CLI options.
2. `main()` optionally stores the placement JSON path in the `VLLM_EXPERT_CONFIG_PATH` environment variable.
3. `main()` constructs `EPLBConfig(policy="custom", ...)`.
4. `load_prompts()` prepares prompt text.
5. `SamplingParams(...)` defines generation controls such as `temperature`, `top_p`, and `max_tokens`.
6. `LLM(...)` boots the vLLM engine.
7. `print_model_info()` reads model metadata from `llm.llm_engine.model_config`.
8. `apply_expert_placement_hook()` prints diagnostic information about the active EPLB policy and the configured expert map.
9. `llm.generate(prompts, sampling_params)` runs the actual inference.

## What the script itself owns

The script contains a small local `ExpertPlacementConfig` dataclass. That class is only used as a convenience structure for example JSON output and local lookup helpers.

The actual placement behavior is not implemented in this dataclass. The real expert placement logic lives inside vLLM's EPLB policy code.

## vLLM interfaces used by the script

The script directly interfaces with the following vLLM files and symbols.

| vLLM file | Symbol(s) used | Why it matters |
| --- | --- | --- |
| `vllm/vllm/__init__.py` | `LLM`, `SamplingParams` re-exports | The script imports `LLM` and `SamplingParams` from the top-level `vllm` package. |
| `vllm/vllm/entrypoints/llm.py` | `LLM.__init__`, `LLM.generate` | This is the public offline inference entrypoint. The script passes model, TP size, eager mode, expert parallel flags, memory settings, and `eplb_config` here. |
| `vllm/vllm/v1/engine/llm_engine.py` | `LLMEngine.from_engine_args`, `LLMEngine.model_config` | `LLM` delegates engine construction here. The script also reads `llm.llm_engine.model_config` for model metadata. |
| `vllm/vllm/sampling_params.py` | `SamplingParams` | Holds decoding parameters used by `llm.generate(...)`. |
| `vllm/vllm/config/parallel.py` | `EPLBConfig`, `ParallelConfig` | Defines the EPLB configuration object and validates that EPLB is only used when expert parallelism is enabled and the distributed setup is large enough. |
| `vllm/vllm/distributed/eplb/policy/__init__.py` | `EPLB_POLICIES` | Maps `policy="custom"` to the custom static placement policy class. |
| `vllm/vllm/distributed/eplb/eplb_state.py` | `EplbState` policy selection and `rebalance_experts(...)` dispatch | This is where vLLM chooses the EPLB policy from `parallel_config.eplb_config.policy` and invokes the policy during expert rebalancing. |
| `vllm/vllm/distributed/eplb/policy/custom_policy.py` | `StaticPlacementPolicy.rebalance_experts`, `_build_map_from_config` | This is the code that actually reads `VLLM_EXPERT_CONFIG_PATH`, parses the JSON file, and builds the physical-to-logical expert map used for rearrangement. |
| `vllm/vllm/distributed/eplb/rebalance_execute.py` | expert weight movement helpers | After the policy computes a new mapping, this layer performs the actual cross-rank expert-weight transfers. |

## Actual custom placement path

The custom placement path is split across the example script and vLLM internals.

### In the script

- If `--expert-placement-config` is set, `main()` stores the absolute path in `VLLM_EXPERT_CONFIG_PATH`.
- `main()` creates `EPLBConfig(policy="custom", step_interval=1, ...)`.
- `apply_expert_placement_hook()` only prints status information. It does not modify vLLM internals.

### In vLLM

- `ParallelConfig.eplb_config` carries the `EPLBConfig` into the engine.
- `EplbState` reads `parallel_config.eplb_config.policy` and selects a policy from `EPLB_POLICIES`.
- For `policy="custom"`, vLLM selects `StaticPlacementPolicy`.
- `StaticPlacementPolicy.rebalance_experts(...)` loads the JSON path from `VLLM_EXPERT_CONFIG_PATH`.
- `_build_map_from_config(...)` builds a layer-by-layer physical-to-logical mapping.
- `rebalance_execute.py` applies the resulting mapping by moving expert weights between EP ranks.

## Expert placement JSON shape

The current custom EPLB policy in vLLM expects JSON in this style:

```json
{
  "expert_to_gpu": {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1
  },
  "layer_configs": {
    "0": {
      "0": 0,
      "1": 1
    }
  }
}
```

Notes:

- `expert_to_gpu` is treated as a global default mapping.
- `layer_configs` can override that mapping per layer.
- The custom policy reads string keys from JSON and converts them to integers.

## Important behavior notes

- The script always constructs an `EPLBConfig` with `policy="custom"`, even when `--expert-placement-config` is not provided. In that case, vLLM's custom policy falls back to a sequential default map if the environment variable is missing or the file path does not exist.
- The local `policy_name` variable in `main()` is currently informational only; it is not used when building `EPLBConfig`.
- `print_model_info()` detects MoE models by inspecting `llm.llm_engine.model_config.hf_config` for `num_local_experts`.
- `apply_expert_placement_hook()` is a visibility helper. The real rearrangement happens inside vLLM on the first EPLB step.

## Non-vLLM dependencies

The script also depends on:

- `datasets.load_dataset` and `datasets.concatenate_datasets` for MMLU prompt loading.
- Standard-library `json`, `os`, and `argparse` for configuration and I/O.

## Summary

`expert_placement_example.py` is best understood as a thin experiment driver around vLLM's existing offline inference and EPLB infrastructure.

It does not implement expert movement itself. Instead, it:

- collects CLI and dataset inputs,
- configures `LLM`, `SamplingParams`, and `EPLBConfig`,
- exposes the custom JSON file path through `VLLM_EXPERT_CONFIG_PATH`, and
- relies on vLLM's EPLB policy and rebalance execution code to perform the real expert placement work.