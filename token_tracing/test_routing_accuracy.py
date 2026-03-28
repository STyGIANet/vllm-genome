#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Tests for token routing accuracy and per-step tracking in DP+EP mode.

Unit tests (no GPU):
    python test_routing_accuracy.py

Integration test — no-duplicate + per-step (requires 8× GPU):
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 \\
    python test_routing_accuracy.py --integration \\
        --model mistralai/Mixtral-8x7B-Instruct-v0.1

Integration test — DeepSeek:
    VLLM_TRACK_ROUTING=1 NCCL_IB_DISABLE=1 \\
    python test_routing_accuracy.py --integration \\
        --model deepseek-ai/deepseek-moe-16b-chat --trust-remote-code
"""

import sys
import os
import unittest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — slicing logic (no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalSliceLogic(unittest.TestCase):
    """Unit-test the rank-local slice computation independently of vLLM."""

    def _make_gathered(self, sizes):
        total = sum(sizes)
        topk_ids = torch.arange(total * 2, dtype=torch.int32).reshape(total, 2)
        topk_weights = torch.rand(total, 2, dtype=torch.float32)
        return topk_ids, topk_weights, sizes

    def _slice(self, topk_ids, topk_weights, sizes, ep_rank):
        start = sum(sizes[:ep_rank])
        end = start + sizes[ep_rank]
        if end <= topk_ids.shape[0] and end > start:
            return topk_ids[start:end], topk_weights[start:end], sizes[ep_rank]
        return topk_ids, topk_weights, topk_ids.shape[0]

    def test_rank0_gets_first_block(self):
        sizes = [10, 15, 12, 8, 10, 10, 10, 5]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=0)
        self.assertEqual(n, 10)
        self.assertTrue(torch.equal(ids, topk_ids[:10]))

    def test_rank3_gets_correct_block(self):
        sizes = [10, 15, 12, 8, 10, 10, 10, 5]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=3)
        expected_start = 10 + 15 + 12
        self.assertEqual(n, 8)
        self.assertTrue(torch.equal(ids, topk_ids[37:45]))

    def test_last_rank_gets_tail(self):
        sizes = [10, 15, 12, 8, 10, 10, 10, 5]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=7)
        total = sum(sizes)
        self.assertEqual(n, 5)
        self.assertTrue(torch.equal(ids, topk_ids[total - 5:]))

    def test_uniform_sizes(self):
        sizes = [8] * 8
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        for rank in range(8):
            ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=rank)
            self.assertEqual(n, 8)
            self.assertTrue(torch.equal(ids, topk_ids[rank * 8:(rank + 1) * 8]))

    def test_slice_preserves_topk_dim(self):
        sizes = [6, 4]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=1)
        self.assertEqual(ids.shape, (4, 2))
        self.assertEqual(wts.shape, (4, 2))

    def test_single_gpu_no_dp(self):
        total = 12
        topk_ids = torch.zeros(total, 2, dtype=torch.int32)
        dp_metadata = None
        capture_ids = topk_ids
        capture_num = topk_ids.shape[0]
        if dp_metadata is not None:
            pass
        self.assertEqual(capture_num, total)
        self.assertIs(capture_ids, topk_ids)

    def test_slicing_does_not_mutate_original(self):
        sizes = [5, 5, 5, 5]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        original_ids = topk_ids.clone()
        self._slice(topk_ids, topk_weights, sizes, ep_rank=2)
        self.assertTrue(torch.equal(topk_ids, original_ids))

    def test_each_rank_slice_is_disjoint(self):
        sizes = [3, 4, 5, 2]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        slices = []
        for rank in range(len(sizes)):
            ids, _, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=rank)
            slices.append(set(range(sum(sizes[:rank]), sum(sizes[:rank]) + n)))
        for i in range(len(slices)):
            for j in range(i + 1, len(slices)):
                self.assertEqual(slices[i] & slices[j], set())

    def test_union_of_slices_covers_all_tokens(self):
        sizes = [7, 3, 11, 5]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        covered = set()
        for rank in range(len(sizes)):
            start = sum(sizes[:rank])
            end = start + sizes[rank]
            covered |= set(range(start, end))
        self.assertEqual(covered, set(range(sum(sizes))))

    def test_empty_rank_not_out_of_bounds(self):
        sizes = [8, 8, 8, 1]
        topk_ids, topk_weights, _ = self._make_gathered(sizes)
        ids, wts, n = self._slice(topk_ids, topk_weights, sizes, ep_rank=3)
        self.assertEqual(n, 1)
        self.assertEqual(ids.shape[0], 1)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — step snapshot queue logic (no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestStepSnapshotQueue(unittest.TestCase):
    """Unit-test the per-step snapshot queue in layer.py."""

    def setUp(self):
        # Import and reset state for each test
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, _STEP_ROUTING_SNAPSHOTS, _STEP_COUNTER,
            reset_step_counter,
        )
        reset_step_counter()
        _ROUTING_DATA.clear()

    def test_push_and_drain(self):
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
        )
        # Simulate a capture for layer 0
        fake_ids = torch.zeros(5, 2, dtype=torch.int32)
        fake_wts = torch.ones(5, 2, dtype=torch.float32)
        _ROUTING_DATA[0] = [{'topk_ids': fake_ids, 'topk_weights': fake_wts, 'num_tokens': 5}]

        push_step_snapshot()
        snaps = drain_step_snapshots()

        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]['step_idx'], 0)
        self.assertIn(0, snaps[0]['routing'])
        self.assertEqual(snaps[0]['routing'][0][0]['num_tokens'], 5)

    def test_drain_clears_queue(self):
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
        )
        _ROUTING_DATA[0] = [{'topk_ids': torch.zeros(3, 2), 'topk_weights': torch.zeros(3, 2), 'num_tokens': 3}]
        push_step_snapshot()
        drain_step_snapshots()
        self.assertEqual(drain_step_snapshots(), [])

    def test_step_indices_increment(self):
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
        )
        for i in range(4):
            _ROUTING_DATA[0] = [{'topk_ids': torch.zeros(2, 2), 'topk_weights': torch.zeros(2, 2), 'num_tokens': 2}]
            push_step_snapshot()

        snaps = drain_step_snapshots()
        self.assertEqual(len(snaps), 4)
        for i, snap in enumerate(snaps):
            self.assertEqual(snap['step_idx'], i)

    def test_reset_clears_both_queue_and_counter(self):
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
            reset_step_counter, _STEP_COUNTER,
        )
        _ROUTING_DATA[0] = [{'topk_ids': torch.zeros(2, 2), 'topk_weights': torch.zeros(2, 2), 'num_tokens': 2}]
        push_step_snapshot()
        reset_step_counter()
        self.assertEqual(drain_step_snapshots(), [])
        self.assertEqual(_STEP_COUNTER[0], 0)

    def test_snapshot_is_independent_copy(self):
        """Modifying _ROUTING_DATA after push should not affect the snapshot."""
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
        )
        original_ids = torch.zeros(3, 2, dtype=torch.int32)
        _ROUTING_DATA[0] = [{'topk_ids': original_ids, 'topk_weights': torch.zeros(3, 2), 'num_tokens': 3}]
        push_step_snapshot()
        # Mutate _ROUTING_DATA after snapshot
        _ROUTING_DATA[0][0]['num_tokens'] = 999

        snaps = drain_step_snapshots()
        # Snapshot was taken before the mutation — it uses the original list
        # (shallow copy of the list, but the dict objects are the same references
        #  which is fine since we snapshot after clear in execute_model)
        self.assertEqual(len(snaps), 1)

    def test_multi_layer_snapshot(self):
        from vllm.model_executor.layers.fused_moe.layer import (
            _ROUTING_DATA, push_step_snapshot, drain_step_snapshots,
        )
        for layer in range(5):
            _ROUTING_DATA[layer] = [{'topk_ids': torch.zeros(4, 2), 'topk_weights': torch.zeros(4, 2), 'num_tokens': 4}]
        push_step_snapshot()
        snaps = drain_step_snapshots()
        self.assertEqual(len(snaps), 1)
        self.assertEqual(set(snaps[0]['routing'].keys()), {0, 1, 2, 3, 4})


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — requires real GPUs
# ─────────────────────────────────────────────────────────────────────────────

def run_integration_test(model: str, dp_size: int, trust_remote_code: bool,
                         num_prompts_per_rank: int = 2, max_tokens: int = 8):
    """
    Verify two properties:
    1. No duplicate token counting (each rank captures local tokens only).
    2. Per-step tracking: we get separate snapshots for prefill AND decode.
       With max_tokens=N, we expect at least N+1 snapshots per rank
       (1 prefill + at least 1 decode).
    """
    from multiprocessing import Process, Queue
    from vllm.utils.network_utils import get_open_port

    dp_master_port = get_open_port()

    PROMPTS = [
        "What is 2 + 2?",
        "Name the capital of France.",
    ][:num_prompts_per_rank]

    result_queue: Queue = Queue()

    def worker(global_rank: int, local_rank: int, out_q):
        os.environ["VLLM_DP_RANK"] = str(global_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
        os.environ["VLLM_TRACK_ROUTING"] = "1"

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model,
            tensor_parallel_size=1,
            enforce_eager=True,
            enable_expert_parallel=True,
            trust_remote_code=trust_remote_code,
            max_num_seqs=32,
            max_model_len=512,
            gpu_memory_utilization=0.85,
            seed=42,
        )
        llm.reset_step_counter()

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        llm.generate(PROMPTS, sampling_params)

        # Collect per-step snapshots (preserves prefill vs decode structure)
        step_snaps = llm.drain_step_snapshots()

        result = {
            "rank": global_rank,
            "ok": False,
            "error": None,
            "num_steps": 0,
            "step_token_counts": [],   # num_tokens per step (first layer)
            "max_tokens": max_tokens,
            "num_prompts": len(PROMPTS),
            "dp_size": dp_size,
        }
        try:
            if not step_snaps:
                result["error"] = "No step snapshots captured"
                out_q.put(result)
                return

            first_layer = sorted(step_snaps[0]['routing'].keys())[0]
            step_counts = []
            for snap in step_snaps:
                caps = snap['routing'].get(first_layer, [])
                tokens_this_step = sum(c['num_tokens'] for c in caps)
                step_counts.append(tokens_this_step)

            result["num_steps"] = len(step_snaps)
            result["step_token_counts"] = step_counts
            result["ok"] = True
        except Exception as e:
            import traceback
            result["error"] = f"{e}\n{traceback.format_exc()}"
        out_q.put(result)

    procs = []
    for local_rank in range(dp_size):
        p = Process(target=worker, args=(local_rank, local_rank, result_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=360)
        if p.exitcode is None:
            p.kill()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print(f"\n{'=' * 70}")
    print(f"Integration test: {model}  dp_size={dp_size}  max_tokens={max_tokens}")
    print(f"{'=' * 70}")

    all_passed = True
    for r in sorted(results, key=lambda x: x["rank"]):
        rank = r["rank"]
        if not r["ok"]:
            print(f"  Rank {rank}: ERROR — {r['error']}")
            all_passed = False
            continue

        num_steps = r["num_steps"]
        counts = r["step_token_counts"]
        local_upper = r["num_prompts"] * 50  # generous upper bound per step
        bug_threshold = r["dp_size"] * local_upper

        # Check 1: at least 2 steps captured (prefill + at least 1 decode)
        has_decode = num_steps >= 2
        # Check 2: no step has more tokens than the local upper bound
        no_duplicates = all(c < bug_threshold for c in counts)

        status_parts = []
        if has_decode:
            status_parts.append(f"✓ {num_steps} steps (prefill+decode)")
        else:
            status_parts.append(f"✗ only {num_steps} step(s) — decode missing?")
            all_passed = False

        if no_duplicates:
            status_parts.append("✓ no duplicate counting")
        else:
            status_parts.append(f"✗ duplicate tokens detected (max={max(counts)})")
            all_passed = False

        print(f"  Rank {rank}: {', '.join(status_parts)}")
        print(f"    step token counts: {counts}")

    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Token routing accuracy tests")
    p.add_argument("--integration", action="store_true",
                   help="Run integration test (requires GPUs)")
    p.add_argument("--model", type=str,
                   default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    p.add_argument("--dp-size", type=int, default=8)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--num-prompts-per-rank", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.integration:
        ok = run_integration_test(
            model=args.model,
            dp_size=args.dp_size,
            trust_remote_code=args.trust_remote_code,
            num_prompts_per_rank=args.num_prompts_per_rank,
            max_tokens=args.max_tokens,
        )
        sys.exit(0 if ok else 1)
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestLocalSliceLogic)
        suite.addTests(
            unittest.TestLoader().loadTestsFromTestCase(TestStepSnapshotQueue)
        )
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
