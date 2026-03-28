#!/usr/bin/env python3
"""
Expert placement verification tests for Mixtral-8x7B and DeepSeek-MoE-16B.

─────────────────────────────────────────────────────────────────────────────
HOW GPU PLACEMENT IS TRACKED
─────────────────────────────────────────────────────────────────────────────
vLLM's EPLB pipeline has three layers of state that let us verify the JSON
is actually being used:

  1. StaticPlacementPolicy.rebalance_experts() (custom_policy.py)
       Reads VLLM_EXPERT_CONFIG_PATH, builds a physical_to_logical_map tensor
       of shape [num_moe_layers, num_physical_slots].  Physical slots are
       partitioned by GPU: GPU k owns slots [k*S .. (k+1)*S - 1] where
       S = num_physical_slots // num_gpus.  The policy logs at INFO level:

           "Expert Mapping per GPU (Sample for Layer 0):"
           "  Rank 0: Experts [56, 57, 58, 59, 60, 61, 62, 63]"  ← reversed
           "  Rank 7: Experts [0, 1, 2, 3, 4, 5, 6, 7]"

       This is the primary verification signal captured by this test.

  2. EplbState (distributed/eplb/eplb_state.py)
       Receives the maps from the policy and calls rearrange() on each
       FusedMoE layer.  rearrange() physically moves the expert weight
       tensors in GPU memory so each rank only holds its assigned experts.
       After rearrange(), the model's actual weight layout matches the map.

  3. FusedMoE layer routing (model_executor/layers/fused_moe/layer.py)
       Stores logical_to_physical_map (inverse of physical_to_logical_map).
       During inference, the router uses this map so tokens dispatched to
       logical expert E land on the physical slot that currently holds E —
       i.e. on the right GPU.

The unit tests here directly call _build_map_from_config() with known inputs
and assert the returned physical_to_logical_map matches the JSON exactly.
The integration test confirms the live log output matches too.

─────────────────────────────────────────────────────────────────────────────
Usage:
    python test_expert_placement.py            # unit tests only (fast, no GPU)
    python test_expert_placement.py --full     # + live inference for both models
    python test_expert_placement.py --mixtral  # + live Mixtral only
    python test_expert_placement.py --deepseek # + live DeepSeek only
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import subprocess
import sys
import traceback

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if VLLM_ROOT not in sys.path:
    sys.path.insert(0, VLLM_ROOT)

from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy  # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def experts_on_gpu(m: torch.Tensor, gpu: int, slots_per_gpu: int, layer: int = 0):
    """Return sorted logical expert IDs held by `gpu` in `layer`."""
    start = gpu * slots_per_gpu
    return sorted(m[layer, start:start + slots_per_gpu].tolist())


def build_map(config_path, num_layers, num_experts, num_gpus, step=0):
    return StaticPlacementPolicy._build_map_from_config(
        config_path, num_layers, num_experts, num_gpus, step=step)


def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        if detail:
            print(f"         {detail}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Mixtral unit tests  (8 experts / 8 GPUs → 1 per GPU)
# ─────────────────────────────────────────────────────────────────────────────

MX_JSON   = os.path.join(SCRIPT_DIR, "mixtral_EP_test.json")
MX_LAYERS, MX_EXPERTS, MX_GPUS, MX_SLOTS = 32, 8, 8, 1


def test_mixtral_step0_reversal():
    """Step 0: full reversal — GPU k holds expert (7-k)."""
    print("\n[Unit / Mixtral] Step 0 — full reversal")
    m = build_map(MX_JSON, MX_LAYERS, MX_EXPERTS, MX_GPUS, step=0)
    ok = True
    for g in range(MX_GPUS):
        ok &= check(experts_on_gpu(m, g, MX_SLOTS) == [7 - g],
                    f"GPU {g} holds expert {7 - g}",
                    f"got {experts_on_gpu(m, g, MX_SLOTS)}")
    return ok


def test_mixtral_step0_layer15_rotation():
    """Step 0, layer 15: GPU k holds expert (k-1) mod 8."""
    print("\n[Unit / Mixtral] Step 0 — layer 15 rotation (+1 mod 8)")
    m = build_map(MX_JSON, MX_LAYERS, MX_EXPERTS, MX_GPUS, step=0)
    ok = True
    for g in range(MX_GPUS):
        expected = [(g - 1) % 8]
        got = experts_on_gpu(m, g, MX_SLOTS, layer=15)
        ok &= check(got == expected,
                    f"Layer 15, GPU {g} holds expert {expected[0]}",
                    f"got {got}")
    diff = any(
        experts_on_gpu(m, g, MX_SLOTS, layer=15) != experts_on_gpu(m, g, MX_SLOTS, layer=0)
        for g in range(MX_GPUS)
    )
    ok &= check(diff, "Layer 15 differs from layer 0 (per-layer override active)")
    return ok


def test_mixtral_step1_linear():
    """Step 1: GPU k holds expert k (default linear)."""
    print("\n[Unit / Mixtral] Step 1 — default linear layout")
    m = build_map(MX_JSON, MX_LAYERS, MX_EXPERTS, MX_GPUS, step=1)
    ok = True
    for g in range(MX_GPUS):
        ok &= check(experts_on_gpu(m, g, MX_SLOTS) == [g],
                    f"GPU {g} holds expert {g}",
                    f"got {experts_on_gpu(m, g, MX_SLOTS)}")
    return ok


def test_mixtral_cycling():
    """Step 2 wraps back to step 0."""
    print("\n[Unit / Mixtral] Multi-step cycling (step 2 == step 0)")
    m0 = build_map(MX_JSON, MX_LAYERS, MX_EXPERTS, MX_GPUS, step=0)
    m2 = build_map(MX_JSON, MX_LAYERS, MX_EXPERTS, MX_GPUS, step=2)
    return check(m0.equal(m2), "step 2 map equals step 0 map (2-step wrap)")


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek unit tests  (64 experts / 8 GPUs → 8 per GPU, 27 MoE layers)
# ─────────────────────────────────────────────────────────────────────────────
# deepseek-moe-16b: 28 total layers, first_k_dense_replace=1
# → 27 MoE layers, 64 routed experts, top-6 routing

DS_JSON   = os.path.join(SCRIPT_DIR, "deepseek_EP_test.json")
DS_LAYERS, DS_EXPERTS, DS_GPUS, DS_SLOTS = 27, 64, 8, 8  # 64/8 = 8 per GPU


def test_deepseek_step0_reversal():
    """Step 0: full reversal — GPU k holds experts [(7-k)*8 .. (7-k)*8+7]."""
    print("\n[Unit / DeepSeek] Step 0 — full block reversal")
    m = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=0)
    ok = True
    for g in range(DS_GPUS):
        mirrored = (DS_GPUS - 1 - g) * DS_SLOTS
        expected = list(range(mirrored, mirrored + DS_SLOTS))
        ok &= check(experts_on_gpu(m, g, DS_SLOTS) == expected,
                    f"GPU {g} holds experts {expected[0]}-{expected[-1]}",
                    f"got {experts_on_gpu(m, g, DS_SLOTS)}")
    return ok


def test_deepseek_step0_layer10_rotation():
    """Step 0, layer 10: GPU k gets block (k+1 mod 8)."""
    print("\n[Unit / DeepSeek] Step 0 — layer 10 rotation (+1 block)")
    m = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=0)
    ok = True
    # From JSON layer_configs["10"]: experts 0-7 → GPU 1, 8-15 → GPU 2, ...,
    # 48-55 → GPU 7, 56-63 → GPU 0
    for g in range(DS_GPUS):
        block = (g - 1) % DS_GPUS   # block that lands on GPU g after +1 rotation
        expected = list(range(block * DS_SLOTS, (block + 1) * DS_SLOTS))
        got = experts_on_gpu(m, g, DS_SLOTS, layer=10)
        ok &= check(got == expected,
                    f"Layer 10, GPU {g} holds experts {expected[0]}-{expected[-1]}",
                    f"got {got}")
    diff = any(
        experts_on_gpu(m, g, DS_SLOTS, layer=10) != experts_on_gpu(m, g, DS_SLOTS, layer=0)
        for g in range(DS_GPUS)
    )
    ok &= check(diff, "Layer 10 differs from layer 0 (per-layer override active)")
    return ok


def test_deepseek_step0_non_override_layers_use_global():
    """Non-overridden layers use the global reversal config."""
    print("\n[Unit / DeepSeek] Step 0 — non-overridden layers use global config")
    m = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=0)
    ok = True
    for layer in [0, 5, 15, 20, 26]:
        g0 = experts_on_gpu(m, 0, DS_SLOTS, layer=layer)
        g7 = experts_on_gpu(m, 7, DS_SLOTS, layer=layer)
        ok &= check(g0 == list(range(56, 64)),
                    f"Layer {layer}, GPU 0 = [56-63]", f"got {g0}")
        ok &= check(g7 == list(range(0, 8)),
                    f"Layer {layer}, GPU 7 = [0-7]",  f"got {g7}")
    return ok


def test_deepseek_step1_linear():
    """Step 1: GPU k holds experts [8k .. 8k+7]."""
    print("\n[Unit / DeepSeek] Step 1 — default linear layout")
    m = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=1)
    ok = True
    for g in range(DS_GPUS):
        expected = list(range(g * DS_SLOTS, (g + 1) * DS_SLOTS))
        ok &= check(experts_on_gpu(m, g, DS_SLOTS) == expected,
                    f"GPU {g} holds experts {expected[0]}-{expected[-1]}",
                    f"got {experts_on_gpu(m, g, DS_SLOTS)}")
    return ok


def test_deepseek_cycling():
    """Step 2 wraps back to step 0."""
    print("\n[Unit / DeepSeek] Multi-step cycling (step 2 == step 0)")
    m0 = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=0)
    m2 = build_map(DS_JSON, DS_LAYERS, DS_EXPERTS, DS_GPUS, step=2)
    return check(m0.equal(m2), "step 2 map equals step 0 map (2-step wrap)")


# ─────────────────────────────────────────────────────────────────────────────
# Shared edge-case tests
# ─────────────────────────────────────────────────────────────────────────────

def test_safety_net_fills_gaps():
    """Partial config (only 2 experts assigned) must leave no -1 slots."""
    print("\n[Unit] Safety net fills unassigned slots")
    import tempfile
    partial = {"expert_to_gpu": {"0": 0, "5": 1}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(partial, f)
        path = f.name
    try:
        m = StaticPlacementPolicy._build_map_from_config(path, 4, 64, 8, step=0)
        return check(not (m == -1).any().item(), "No -1 slots after safety net fill")
    finally:
        os.unlink(path)


def test_missing_config_fallback():
    """Missing config path → identity map, no crash."""
    print("\n[Unit] Missing config path falls back to identity map")
    m = StaticPlacementPolicy._build_map_from_config(
        "/does/not/exist.json", 4, 8, 8, step=0)
    return check(m[0].equal(torch.arange(8, dtype=torch.int32)),
                 "Fallback map is identity [0..7]",
                 f"got {m[0].tolist()}")


def run_unit_tests() -> int:
    print("=" * 60)
    print("PART 1: Unit tests (no GPU required)")
    print("=" * 60)
    results = [
        test_mixtral_step0_reversal(),
        test_mixtral_step0_layer15_rotation(),
        test_mixtral_step1_linear(),
        test_mixtral_cycling(),
        test_deepseek_step0_reversal(),
        test_deepseek_step0_layer10_rotation(),
        test_deepseek_step0_non_override_layers_use_global(),
        test_deepseek_step1_linear(),
        test_deepseek_cycling(),
        test_safety_net_fills_gaps(),
        test_missing_config_fallback(),
    ]
    passed, total = sum(results), len(results)
    print(f"\nUnit tests: {passed}/{total} passed", end="")
    print("  — all good!" if passed == total else f"  ({total - passed} FAILED)")
    return total - passed


# ─────────────────────────────────────────────────────────────────────────────
# Integration test helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_placement_blocks(output: str, num_gpus: int) -> list[dict[int, list[int]]]:
    """
    Parse "Expert Mapping per GPU" log blocks from combined subprocess output.

    Each DP rank process emits its own blocks with a prefix like:
        (EngineCore_DP3 pid=...) INFO ... Expert Mapping per GPU ...
        (EngineCore_DP3 pid=...) INFO ...   Rank 0: Experts [56, ...]

    We filter to a single PID (the one with the most blocks) to avoid
    interleaving from concurrent processes, then return each block as
    {rank_id: [expert_ids]}.
    """
    dp_re = re.compile(r"\(EngineCore_DP\d+ pid=(\d+)\)")
    rank_lines: dict[str, list[str]] = {}
    for line in output.splitlines():
        m = dp_re.search(line)
        if m:
            rank_lines.setdefault(m.group(1), []).append(line)

    if not rank_lines:
        lines = output.splitlines()
    else:
        best = max(rank_lines,
                   key=lambda p: sum(1 for l in rank_lines[p] if "Expert Mapping per GPU" in l))
        lines = rank_lines[best]

    blocks: list[dict[int, list[int]]] = []
    current: dict[int, list[int]] = {}
    for line in lines:
        if "Expert Mapping per GPU" in line:
            if current:
                blocks.append(current)
            current = {}
        else:
            m2 = re.search(r"Rank\s+(\d+):\s+Experts\s+(\[[^\]]+\])", line)
            if m2:
                current[int(m2.group(1))] = sorted(json.loads(m2.group(2)))
    if current:
        blocks.append(current)
    return blocks


def verify_blocks(blocks, step0_expected, step1_expected, num_gpus, model_name) -> int:
    """Check first block == step 0, second block == step 1, confirm they differ."""
    if not blocks:
        print(f"  {FAIL}  No 'Expert Mapping per GPU' blocks found.")
        print("         Ensure VLLM_LOGGING_LEVEL=INFO is set.")
        return 1

    print(f"\n{INFO} Found {len(blocks)} rebalance call(s) for {model_name}")

    ok = True
    print(f"\n{INFO} Rebalance call #1 (expected: step 0)")
    for g in range(num_gpus):
        actual = blocks[0].get(g)
        exp = step0_expected[g]
        ok &= check(actual == exp,
                    f"GPU {g}: {exp[0] if len(exp) == 1 else f'{exp[0]}-{exp[-1]}'}",
                    f"got {actual}")

    if len(blocks) >= 2:
        print(f"\n{INFO} Rebalance call #2 (expected: step 1)")
        for g in range(num_gpus):
            actual = blocks[1].get(g)
            exp = step1_expected[g]
            ok &= check(actual == exp,
                        f"GPU {g}: {exp[0] if len(exp) == 1 else f'{exp[0]}-{exp[-1]}'}",
                        f"got {actual}")
        ok &= check(blocks[0] != blocks[1],
                    "Steps 0 and 1 differ (multi-step cycling confirmed)")
    else:
        print(f"\n{INFO} Only 1 block captured — increase --output-length to see step 1")

    return 0 if ok else 1


def run_inference(model, dp_size, tp_size, config_file, extra_args=(), timeout=900):
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "expert_placement_example.py"),
        f"--model={model}",
        f"--dp-size={dp_size}",
        f"--tp-size={tp_size}",
        "--trust-remote-code",
        "--enforce-eager",
        f"--expert-placement-config={config_file}",
        "--dataset=simple",
        "--num-prompts-per-rank=1",
        "--output-length=4",
        "--gpu-memory-utilization=0.90",
        *extra_args,
    ]
    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["NCCL_IB_DISABLE"] = "1"
    print(f"\n{INFO} Running: {os.path.basename(cmd[1])} {' '.join(cmd[2:])}\n")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=SCRIPT_DIR, env=env)
    except subprocess.TimeoutExpired:
        print(f"  {FAIL}  Timed out (>{timeout}s)")
        return None, 1
    combined = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        print(f"  {FAIL}  Process exited with code {r.returncode}")
        lines = combined.strip().splitlines()
        print("\n--- last 50 lines ---")
        print("\n".join(lines[-50:]))
        return combined, 1
    return combined, 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — Mixtral-8x7B  (8 experts / 8 GPUs → 1 per GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_mixtral_integration():
    print("\n" + "=" * 60)
    print("PART 2a: Integration — Mixtral-8x7B (8×DP+EP, 1 expert/GPU)")
    print("=" * 60)

    step0 = {g: [7 - g] for g in range(8)}           # full reversal
    step1 = {g: [g] for g in range(8)}                # linear

    combined, rc = run_inference(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        dp_size=8, tp_size=1,
        config_file="mixtral_EP_test.json",
    )
    if rc:
        return rc

    blocks = parse_placement_blocks(combined, num_gpus=8)
    return verify_blocks(blocks, step0, step1, num_gpus=8, model_name="Mixtral-8x7B")


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — DeepSeek-MoE-16B  (64 experts / 8 GPUs → 8 per GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_deepseek_integration():
    print("\n" + "=" * 60)
    print("PART 2b: Integration — DeepSeek-MoE-16B-Chat (8×DP+EP, 8 experts/GPU)")
    print("=" * 60)

    SLOTS = 8
    step0 = {g: list(range((7 - g) * SLOTS, (7 - g + 1) * SLOTS)) for g in range(8)}  # reversal
    step1 = {g: list(range(g * SLOTS, (g + 1) * SLOTS)) for g in range(8)}             # linear

    combined, rc = run_inference(
        model="deepseek-ai/deepseek-moe-16b-chat",
        dp_size=8, tp_size=1,
        config_file="deepseek_EP_test.json",
    )
    if rc:
        return rc

    blocks = parse_placement_blocks(combined, num_gpus=8)
    return verify_blocks(blocks, step0, step1, num_gpus=8, model_name="DeepSeek-MoE-16B")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_mixtral  = "--full" in sys.argv or "--mixtral"  in sys.argv
    run_deepseek = "--full" in sys.argv or "--deepseek" in sys.argv

    failures = 0
    try:
        failures += run_unit_tests()
    except Exception:
        traceback.print_exc()
        failures += 1

    if run_mixtral:
        try:
            failures += run_mixtral_integration()
        except Exception:
            traceback.print_exc()
            failures += 1

    if run_deepseek:
        try:
            failures += run_deepseek_integration()
        except Exception:
            traceback.print_exc()
            failures += 1

    if not run_mixtral and not run_deepseek:
        print(f"\n{INFO} Skipping integration tests.")
        print(f"       --mixtral   live Mixtral-8x7B inference")
        print(f"       --deepseek  live DeepSeek-MoE-16B inference")
        print(f"       --full      both")

    print("\n" + "=" * 60)
    print(f"{PASS}  All tests passed." if failures == 0 else f"{FAIL}  {failures} suite(s) failed.")
    print("=" * 60)
    sys.exit(failures)
