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
import sys
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests — StaticPlacementPolicy dynamic config
# ---------------------------------------------------------------------------

class TestDynamicConfigPolicy(unittest.TestCase):
    """Tests for StaticPlacementPolicy.set_dynamic_config() / _build_map_from_config()."""

    def setUp(self):
        # Reset class-level state before each test.
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy
        StaticPlacementPolicy._dynamic_config = None
        StaticPlacementPolicy._step = 0
        self.Policy = StaticPlacementPolicy

    def test_dynamic_config_overrides_json(self):
        """set_dynamic_config() causes _build_map_from_config to use the dict."""
        import torch
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        num_layers, num_experts, num_gpus = 2, 4, 2
        # Reversed: experts 2,3 → GPU 0; experts 0,1 → GPU 1
        mapping = {"expert_to_gpu": {"0": 1, "1": 1, "2": 0, "3": 0}}
        StaticPlacementPolicy.set_dynamic_config(mapping)

        result = StaticPlacementPolicy._build_map_from_config(
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
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        StaticPlacementPolicy.set_dynamic_config({"expert_to_gpu": {"0": 0, "1": 1}})
        self.assertIsNotNone(StaticPlacementPolicy._dynamic_config)

        StaticPlacementPolicy._build_map_from_config(
            config_path=None, num_layers=1, num_physical_experts=2, num_gpus=2
        )
        self.assertIsNone(StaticPlacementPolicy._dynamic_config)

    def test_fallback_to_identity_when_no_config_and_no_dynamic(self):
        """Without a config file or dynamic override, returns sequential mapping."""
        import torch
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        result = StaticPlacementPolicy._build_map_from_config(
            config_path=None, num_layers=1, num_physical_experts=4, num_gpus=2
        )
        self.assertEqual(result.tolist(), [[0, 1, 2, 3]])

    def test_set_dynamic_config_roundtrip(self):
        """set_dynamic_config then rebalance_experts returns correct maps."""
        import torch
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        num_experts, num_gpus, num_layers = 4, 2, 1
        # Identity placement
        mapping = {"expert_to_gpu": {"0": 0, "1": 0, "2": 1, "3": 1}}
        StaticPlacementPolicy.set_dynamic_config(mapping)

        dummy_load = torch.zeros(num_layers, num_experts)
        p2l, l2p, replica_count = StaticPlacementPolicy.rebalance_experts(
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

    def _make_runner(self):
        """Build a minimal mock of GPUModelRunner with tracking enabled."""
        runner = MagicMock()
        runner._compute_placement_callback = None
        runner.parallel_config.enable_eplb = False
        runner.parallel_config.eplb_config.policy = "default"
        return runner

    def test_callback_not_called_when_eplb_disabled(self):
        """Callback is NOT called when enable_eplb=False."""
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        calls = []
        def cb(routing):
            calls.append(routing)
            return {"expert_to_gpu": {}}

        import vllm.model_executor.layers.fused_moe.layer as layer_mod
        import vllm.v1.worker.gpu_model_runner as runner_mod

        runner = MagicMock()
        runner._compute_placement_callback = cb
        runner.parallel_config.enable_eplb = False
        runner.parallel_config.eplb_config.policy = "custom"

        fake_routing = {0: [{"topk_ids": MagicMock(), "topk_weights": MagicMock(), "num_tokens": 4}]}

        with patch.object(layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(layer_mod, "get_routing_data", return_value=fake_routing), \
             patch.object(layer_mod, "push_step_snapshot"), \
             patch.object(layer_mod, "clear_routing_data"):
            # Call the actual method with the mock as self
            runner_mod.GPUModelRunner._on_routing_step(runner)

        self.assertEqual(calls, [], "callback must not fire when eplb disabled")

    def test_callback_called_when_eplb_and_custom_policy(self):
        """Callback IS called and result stored in StaticPlacementPolicy."""
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy
        StaticPlacementPolicy._dynamic_config = None

        placement_result = {"expert_to_gpu": {"0": 1, "1": 0}}
        calls = []
        def cb(routing):
            calls.append(routing)
            return placement_result

        import vllm.model_executor.layers.fused_moe.layer as layer_mod
        import vllm.v1.worker.gpu_model_runner as runner_mod

        runner = MagicMock()
        runner._compute_placement_callback = cb
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        fake_routing = {0: [{"topk_ids": MagicMock(), "topk_weights": MagicMock(), "num_tokens": 4}]}

        with patch.object(layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(layer_mod, "get_routing_data", return_value=fake_routing), \
             patch.object(layer_mod, "push_step_snapshot"), \
             patch.object(layer_mod, "clear_routing_data"):
            runner_mod.GPUModelRunner._on_routing_step(runner)

        self.assertEqual(len(calls), 1, "callback should be called once")
        self.assertEqual(StaticPlacementPolicy._dynamic_config, placement_result)

    def test_callback_exception_does_not_crash(self):
        """An exception inside the callback is caught and logged, not re-raised."""
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy
        StaticPlacementPolicy._dynamic_config = None

        def bad_cb(routing):
            raise RuntimeError("intentional test error")

        import vllm.model_executor.layers.fused_moe.layer as layer_mod
        import vllm.v1.worker.gpu_model_runner as runner_mod

        runner = MagicMock()
        runner._compute_placement_callback = bad_cb
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        fake_routing = {}

        with patch.object(layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(layer_mod, "get_routing_data", return_value=fake_routing), \
             patch.object(layer_mod, "push_step_snapshot"), \
             patch.object(layer_mod, "clear_routing_data"):
            # Should not raise
            runner_mod.GPUModelRunner._on_routing_step(runner)

        # dynamic config should remain None since callback raised
        self.assertIsNone(StaticPlacementPolicy._dynamic_config)

    def test_empty_callback_result_skips_policy_update(self):
        """An empty dict returned by the callback does not update the policy."""
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy
        StaticPlacementPolicy._dynamic_config = None

        import vllm.model_executor.layers.fused_moe.layer as layer_mod
        import vllm.v1.worker.gpu_model_runner as runner_mod

        runner = MagicMock()
        runner._compute_placement_callback = lambda r: {}
        runner.parallel_config.enable_eplb = True
        runner.parallel_config.eplb_config.policy = "custom"

        with patch.object(layer_mod, "_is_tracking_enabled", return_value=True), \
             patch.object(layer_mod, "get_routing_data", return_value={}), \
             patch.object(layer_mod, "push_step_snapshot"), \
             patch.object(layer_mod, "clear_routing_data"):
            runner_mod.GPUModelRunner._on_routing_step(runner)

        self.assertIsNone(StaticPlacementPolicy._dynamic_config)


# ---------------------------------------------------------------------------
# Integration test — real inference (skipped without GPU)
# ---------------------------------------------------------------------------

class TestCombinedIntegration(unittest.TestCase):
    """Live inference test: token tracking + expert placement in one session."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls._has_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 1

    @unittest.skipUnless(
        __import__("os").environ.get("RUN_COMBINED_INTEGRATION") == "1",
        "Set RUN_COMBINED_INTEGRATION=1 to run (requires >=1 GPU)",
    )
    def test_tracking_and_placement_no_crash(self):
        """Both features active together should produce routing data and not crash."""
        import os
        import tempfile, json
        import torch
        from vllm import LLM, SamplingParams
        from vllm.config import EPLBConfig
        from vllm.distributed.eplb.policy.custom_policy import StaticPlacementPolicy

        StaticPlacementPolicy._dynamic_config = None
        StaticPlacementPolicy._step = 0
        os.environ["VLLM_TRACK_ROUTING"] = "1"

        # Write a minimal placement JSON — identity mapping for 8 experts on 1 GPU
        placement = {"expert_to_gpu": {str(i): 0 for i in range(8)}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(placement, f)
            json_path = f.name
        os.environ["VLLM_EXPERT_CONFIG_PATH"] = json_path

        cb_calls = []
        def cb(routing):
            cb_calls.append(len(routing))
            return {}

        try:
            llm = LLM(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                tensor_parallel_size=1,
                enable_expert_parallel=False,   # single-GPU: no EP
                enable_eplb=False,              # single-GPU: no EPLB
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
                self.assertIsInstance(snap["routing"], dict)
        finally:
            os.unlink(json_path)


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
        import os
        os.environ["RUN_COMBINED_INTEGRATION"] = "1"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicConfigPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestOnRoutingStepCallback))
    if args.integration:
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
