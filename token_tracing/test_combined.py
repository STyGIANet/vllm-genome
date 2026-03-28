#!/usr/bin/env python3
"""
test_combined.py — unit and integration tests for the merged moe-merged branch.

Tests verify that:
1. StaticPlacementPolicy.set_dynamic_config() overrides JSON reading (unit).
2. _on_routing_step() invokes the compute_placement callback and wires the
   result into StaticPlacementPolicy (unit).
3. The dynamic config slot is consumed after one rebalance_experts() call (unit).
4. End-to-end: tracking + placement both work in a single vLLM session
   (integration — requires GPU, skipped otherwise).

Run (no GPU needed for units):
    source token_inference_tracking/vllm/.venv/bin/activate
    python token_tracing/test_combined.py

Run integration tests (8 GPUs required):
    python token_tracing/test_combined.py --integration
"""

import argparse
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Load custom_policy and layer modules directly to avoid the full vllm import
# chain (which requires compiled C extensions that may not be available when
# running against an unbuilt checkout).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module_direct(rel_path: str, module_name: str):
    """Load a .py file without triggering package-level __init__ imports."""
    full_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_custom_policy():
    """Load StaticPlacementPolicy without pulling in vllm.distributed."""
    import torch  # only real dep

    # Stub out vllm.logger so the policy can call init_logger()
    _stub_logger()

    # Stub abstract base so we can load policy without full vllm package
    abstract_mod = types.ModuleType("vllm.distributed.eplb.policy.abstract")
    class AbstractEplbPolicy:
        pass
    abstract_mod.AbstractEplbPolicy = AbstractEplbPolicy
    sys.modules["vllm.distributed.eplb.policy.abstract"] = abstract_mod

    mod = _load_module_direct(
        "vllm/distributed/eplb/policy/custom_policy.py",
        "vllm.distributed.eplb.policy.custom_policy",
    )
    return mod.StaticPlacementPolicy


def _stub_logger():
    """Inject a no-op vllm.logger if not already present."""
    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        def init_logger(name):
            import logging
            return logging.getLogger(name)
        logger_mod.init_logger = init_logger
        sys.modules["vllm.logger"] = logger_mod


def _load_layer_module():
    """Load layer.py globals without triggering full vllm import."""
    _stub_logger()

    # Stub heavy deps that layer.py may import at module level
    for stub in [
        "vllm.model_executor.layers.fused_moe.config",
        "vllm.envs",
    ]:
        if stub not in sys.modules:
            sys.modules[stub] = types.ModuleType(stub)

    # VLLM_TRACK_ROUTING env var controls _is_tracking_enabled()
    mod = _load_module_direct(
        "vllm/model_executor/layers/fused_moe/layer.py",
        "vllm.model_executor.layers.fused_moe.layer",
    )
    return mod


# ---------------------------------------------------------------------------
# Unit tests — StaticPlacementPolicy dynamic config
# ---------------------------------------------------------------------------

class TestDynamicConfigPolicy(unittest.TestCase):
    """Tests for StaticPlacementPolicy.set_dynamic_config() / _build_map_from_config()."""

    @classmethod
    def setUpClass(cls):
        cls.Policy = _load_custom_policy()

    def setUp(self):
        self.Policy._dynamic_config = None
        self.Policy._step = 0

    def test_dynamic_config_overrides_json(self):
        """set_dynamic_config() causes _build_map_from_config to use the dict."""
        import torch

        num_layers, num_experts, num_gpus = 2, 4, 2
        # Reversed: experts 2,3 → GPU 0; experts 0,1 → GPU 1
        mapping = {"expert_to_gpu": {"0": 1, "1": 1, "2": 0, "3": 0}}
        self.Policy.set_dynamic_config(mapping)

        result = self.Policy._build_map_from_config(
            config_path=None,
            num_layers=num_layers,
            num_physical_experts=num_experts,
            num_gpus=num_gpus,
        )
        # GPU 0 → physical slots [0, 1]; should hold experts 2 and 3.
        gpu0_experts = sorted(result[0, :2].tolist())
        self.assertEqual(gpu0_experts, [2, 3])

    def test_dynamic_config_consumed_after_one_call(self):
        """The dynamic config slot is cleared after one _build_map_from_config call."""
        self.Policy.set_dynamic_config({"expert_to_gpu": {"0": 0, "1": 1}})
        self.assertIsNotNone(self.Policy._dynamic_config)

        self.Policy._build_map_from_config(
            config_path=None, num_layers=1, num_physical_experts=2, num_gpus=2
        )
        self.assertIsNone(self.Policy._dynamic_config)

    def test_fallback_to_identity_when_no_config_and_no_dynamic(self):
        """Without a config file or dynamic override, returns sequential mapping."""
        result = self.Policy._build_map_from_config(
            config_path=None, num_layers=1, num_physical_experts=4, num_gpus=2
        )
        self.assertEqual(result.tolist(), [[0, 1, 2, 3]])

    def test_set_dynamic_config_roundtrip(self):
        """set_dynamic_config then rebalance_experts returns correct maps."""
        import torch

        num_experts, num_gpus, num_layers = 4, 2, 1
        # Identity placement
        mapping = {"expert_to_gpu": {"0": 0, "1": 0, "2": 1, "3": 1}}
        self.Policy.set_dynamic_config(mapping)

        dummy_load = torch.zeros(num_layers, num_experts)
        p2l, l2p, replica_count = self.Policy.rebalance_experts(
            global_expert_load=dummy_load,
            num_replicas=num_experts,
            num_groups=1,
            num_nodes=1,
            num_gpus=num_gpus,
        )
        # Experts 0,1 on GPU 0 → physical slots 0,1; experts 2,3 on GPU 1 → slots 2,3
        self.assertEqual(sorted(p2l[0, :2].tolist()), [0, 1])
        self.assertEqual(sorted(p2l[0, 2:].tolist()), [2, 3])


# ---------------------------------------------------------------------------
# Unit tests — _on_routing_step callback wiring
# ---------------------------------------------------------------------------

class TestOnRoutingStepCallback(unittest.TestCase):
    """Tests for gpu_model_runner._on_routing_step() callback integration."""

    @classmethod
    def setUpClass(cls):
        cls.Policy = _load_custom_policy()
        # Use a MagicMock for the layer module — tests patch it anyway.
        cls.layer_mod = MagicMock()

    def setUp(self):
        self.Policy._dynamic_config = None

    def _make_runner(self):
        runner = MagicMock()
        runner._compute_placement_callback = None
        runner.parallel_config.enable_eplb = False
        runner.parallel_config.eplb_config.policy = "default"
        return runner

    def _call_on_routing_step(self, runner, fake_routing):
        """Call _on_routing_step logic directly without importing gpu_model_runner."""
        layer_mod = self.layer_mod
        Policy = self.Policy

        # Replicate the _on_routing_step logic inline so we can test it
        # without importing gpu_model_runner (which requires C extensions).
        if layer_mod._is_tracking_enabled():
            routing_snapshot = layer_mod.get_routing_data()
            layer_mod.push_step_snapshot()
            layer_mod.clear_routing_data()

            callback = runner._compute_placement_callback
            if (
                callback is not None
                and runner.parallel_config.enable_eplb
                and runner.parallel_config.eplb_config.policy == "custom"
            ):
                try:
                    placement = callback(routing_snapshot)
                    if placement:
                        Policy.set_dynamic_config(placement)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "compute_placement callback raised an exception: %s", exc
                    )

    def test_callback_not_called_when_eplb_disabled(self):
        """Callback is NOT called when enable_eplb=False."""
        calls = []
        def cb(routing):
            calls.append(routing)
            return {"expert_to_gpu": {}}

        runner = self._make_runner()
        runner._compute_placement_callback = cb
        runner.parallel_config.enable_eplb = False
        runner.parallel_config.eplb_config.policy = "custom"

        fake_routing = {0: [{"topk_ids": MagicMock(), "topk_weights": MagicMock(), "num_tokens": 4}]}

        with patch.object(self.layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(self.layer_mod, "get_routing_data", return_value=fake_routing), \
             patch.object(self.layer_mod, "push_step_snapshot"), \
             patch.object(self.layer_mod, "clear_routing_data"):
            self._call_on_routing_step(runner, fake_routing)

        self.assertEqual(calls, [], "callback must not fire when eplb disabled")

    def test_callback_called_when_eplb_and_custom_policy(self):
        """Callback IS called and result stored in StaticPlacementPolicy."""
        placement_result = {"expert_to_gpu": {"0": 1, "1": 0}}
        calls = []
        def cb(routing):
            calls.append(routing)
            return placement_result

        runner = self._make_runner()
        runner._compute_placement_callback = cb
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        fake_routing = {0: [{"topk_ids": MagicMock(), "topk_weights": MagicMock(), "num_tokens": 4}]}

        with patch.object(self.layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(self.layer_mod, "get_routing_data", return_value=fake_routing), \
             patch.object(self.layer_mod, "push_step_snapshot"), \
             patch.object(self.layer_mod, "clear_routing_data"):
            self._call_on_routing_step(runner, fake_routing)

        self.assertEqual(len(calls), 1, "callback should be called once")
        self.assertEqual(self.Policy._dynamic_config, placement_result)

    def test_callback_exception_does_not_crash(self):
        """An exception inside the callback is caught and logged, not re-raised."""
        def bad_cb(routing):
            raise RuntimeError("intentional test error")

        runner = self._make_runner()
        runner._compute_placement_callback = bad_cb
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        with patch.object(self.layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(self.layer_mod, "get_routing_data", return_value={}), \
             patch.object(self.layer_mod, "push_step_snapshot"), \
             patch.object(self.layer_mod, "clear_routing_data"):
            self._call_on_routing_step(runner, {})

        self.assertIsNone(self.Policy._dynamic_config)

    def test_empty_callback_result_skips_policy_update(self):
        """An empty dict returned by the callback does not update the policy."""
        runner = self._make_runner()
        runner._compute_placement_callback = lambda r: {}
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        with patch.object(self.layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(self.layer_mod, "get_routing_data", return_value={}), \
             patch.object(self.layer_mod, "push_step_snapshot"), \
             patch.object(self.layer_mod, "clear_routing_data"):
            self._call_on_routing_step(runner, {})

        self.assertIsNone(self.Policy._dynamic_config)


# ---------------------------------------------------------------------------
# Integration test — real inference (skipped without GPU)
# ---------------------------------------------------------------------------

class TestCombinedIntegration(unittest.TestCase):
    """Live inference test: token tracking + expert placement in one session."""

    @unittest.skipUnless(
        os.environ.get("RUN_COMBINED_INTEGRATION") == "1",
        "Set RUN_COMBINED_INTEGRATION=1 to run (requires >=1 GPU)",
    )
    def test_tracking_and_placement_no_crash(self):
        """Both features active together should produce routing data and not crash."""
        import json
        import tempfile

        os.environ["VLLM_TRACK_ROUTING"] = "1"
        placement = {"expert_to_gpu": {str(i): 0 for i in range(8)}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(placement, f)
            json_path = f.name
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = json_path

        try:
            from vllm import LLM, SamplingParams

            llm = LLM(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                tensor_parallel_size=1,
                enable_expert_parallel=False,
                enable_eplb=False,
                enforce_eager=True,
                max_model_len=128,
            )
            params = SamplingParams(temperature=0.0, max_tokens=8)
            llm.generate(["Hello"], params)

            snapshots = llm.drain_step_snapshots()
            self.assertGreater(len(snapshots), 0, "should have ≥1 routing step")
            for snap in snapshots:
                self.assertIn("step_idx", snap)
                self.assertIn("routing", snap)
        finally:
            os.unlink(json_path)


# ---------------------------------------------------------------------------
# Unit tests — compute_placement() load-balancing algorithm
# ---------------------------------------------------------------------------

class TestComputePlacement(unittest.TestCase):
    """Tests for the greedy bin-packing compute_placement() in combined_launch.py."""

    @classmethod
    def setUpClass(cls):
        import importlib.util, sys, types
        # Load combined_launch.py directly to avoid importing vllm
        spec = importlib.util.spec_from_file_location(
            "_combined_launch",
            str(_REPO_ROOT / "token_tracing/combined_launch.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Stub out heavy imports that combined_launch pulls in at module level
        for name in ["vllm", "vllm.config", "datasets"]:
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        cls.compute_placement = staticmethod(mod.compute_placement)

    def setUp(self):
        # Set dp env vars that compute_placement reads
        os.environ["VLLM_DP_SIZE"] = "4"
        os.environ["_COMBINED_TP_SIZE"] = "1"
        os.environ["VLLM_DP_RANK"] = "1"  # non-0 so logging is suppressed

    def _make_routing(self, expert_hits: list[int]) -> dict:
        """Build a fake routing dict where expert i is hit expert_hits[i] times."""
        import torch
        # All tokens route to exactly one expert (top_k=1 for simplicity)
        token_list = []
        for expert_id, count in enumerate(expert_hits):
            token_list.extend([expert_id] * count)
        topk_ids = torch.tensor(token_list, dtype=torch.int32).unsqueeze(1)  # [T, 1]
        topk_wts = torch.ones_like(topk_ids, dtype=torch.float32)
        return {0: [{"topk_ids": topk_ids, "topk_weights": topk_wts, "num_tokens": len(token_list)}]}

    def test_empty_routing_returns_empty(self):
        """Empty routing dict → return {}."""
        self.assertEqual(self.compute_placement({}), {})

    def test_all_experts_assigned(self):
        """Every expert in the routing data is assigned to some GPU."""
        routing = self._make_routing([10, 5, 8, 3])  # 4 experts
        result = self.compute_placement(routing)
        self.assertIn("expert_to_gpu", result)
        assigned = set(result["expert_to_gpu"].keys())
        self.assertEqual(assigned, {"0", "1", "2", "3"})

    def test_gpu_assignments_in_range(self):
        """All assigned GPUs are valid (0 ≤ gpu_id < num_gpus)."""
        num_gpus = 4  # matches VLLM_DP_SIZE * _COMBINED_TP_SIZE
        routing = self._make_routing([20, 1, 15, 7, 3, 9, 2, 14])  # 8 experts
        result = self.compute_placement(routing)
        for gpu_id in result["expert_to_gpu"].values():
            self.assertGreaterEqual(gpu_id, 0)
            self.assertLess(gpu_id, num_gpus)

    def test_balanced_load(self):
        """With equal-load experts, all GPUs should receive the same number of experts."""
        # 8 experts × 4 GPUs = 2 experts per GPU (perfect balance)
        routing = self._make_routing([10] * 8)
        result = self.compute_placement(routing)
        from collections import Counter
        counts = Counter(result["expert_to_gpu"].values())
        for gpu_id in range(4):
            self.assertEqual(counts[gpu_id], 2,
                             f"GPU {gpu_id} should get 2 experts but got {counts[gpu_id]}")

    def test_heavy_expert_gets_own_gpu(self):
        """A single very heavy expert should be placed alone on one GPU."""
        # Expert 0 has 1000 hits; experts 1-3 each have 1 hit.
        # Greedy: expert 0 → GPU A; remaining 3 experts spread across GPUs.
        routing = self._make_routing([1000, 1, 1, 1])
        result = self.compute_placement(routing)
        # GPU assigned to expert 0 should appear only once in the mapping
        from collections import Counter
        heavy_gpu = result["expert_to_gpu"]["0"]
        counts = Counter(result["expert_to_gpu"].values())
        self.assertEqual(counts[heavy_gpu], 1,
                         "Heavy expert should be alone on its GPU")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integration", action="store_true",
        help="Also run integration tests (requires GPU; sets RUN_COMBINED_INTEGRATION=1)"
    )
    args, remaining = parser.parse_known_args()
    if args.integration:
        os.environ["RUN_COMBINED_INTEGRATION"] = "1"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicConfigPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestOnRoutingStepCallback))
    suite.addTests(loader.loadTestsFromTestCase(TestComputePlacement))
    if args.integration:
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
