# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

# Global singleton instances
_global_experts_capturer: RoutedExpertsCapturer | None = None


class RoutedExpertsCapturer:
    """
    Capturer for routed experts with device and optional shared memory buffer.

    This class captures expert routing decisions during model forward passes
    and optionally stores them in shared memory for cross-process access.
    """

    _instance: RoutedExpertsCapturer | None = None

    def __init__(self) -> None:
        self._device_buffer: torch.Tensor | None = None

    @classmethod
    def create(cls) -> RoutedExpertsCapturer:
        """Create a global singleton instance."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        _global_experts_capturer = cls()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> RoutedExpertsCapturer | None:
        """Get the global singleton instance."""
        return _global_experts_capturer

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        """
        Initialize the device buffer.

        Args:
            max_num_batched_tokens: Maximum number of tokens in a batch.
            max_num_kv_tokens: Unused legacy argument kept for compatibility.
            vllm_config: vllm configuration containing layer and expert info.
        """

        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok

        # Initialize device buffer
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        ctx = get_forward_context()
        if ctx.additional_kwargs.get("skip_routed_experts_capture", False):
            return
        prefill_ranges = list(
            ctx.additional_kwargs.get("routing_capture_prefill_ranges", [])
        )

        if layer_id >= self._device_buffer.shape[1]:
            return

        if not prefill_ranges:
            return

        for start, end in prefill_ranges:
            if end <= start:
                continue
            self._device_buffer[start:end, layer_id, :] = topk_ids[start:end]

    def get_captured_experts(self, num_tokens: int) -> np.ndarray | None:
        """
        Materialize the current step's captured experts on CPU.

        Args:
            num_tokens: Number of scheduled tokens in the current step.
        """
        if get_tensor_model_parallel_rank() != 0:
            return None
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")
        if num_tokens <= 0:
            return None

        return self._device_buffer[:num_tokens, :, :].cpu().numpy()

    def get_captured_experts_for_ranges(
        self, ranges: list[tuple[int, int]]
    ) -> np.ndarray | None:
        """Materialize only the requested token ranges on CPU."""
        if get_tensor_model_parallel_rank() != 0:
            return None
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")
        if not ranges:
            return None

        parts: list[np.ndarray] = []
        for start, end in ranges:
            if end <= start:
                continue
            parts.append(self._device_buffer[start:end, :, :].cpu().numpy())

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    def get_unique_layer_expert_pairs_for_ranges(
        self, ranges: list[tuple[int, int]]
    ) -> list[np.ndarray] | None:
        if get_tensor_model_parallel_rank() != 0:
            return None
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")
        if not ranges:
            return None

        num_layers = self._device_buffer.shape[1]
        top_k = self._device_buffer.shape[2]
        layer_ids = torch.arange(
            num_layers,
            dtype=torch.int32,
            device=self._device_buffer.device,
        ).view(1, num_layers, 1).expand(1, num_layers, top_k)

        results: list[np.ndarray] = []
        for start, end in ranges:
            if end <= start:
                results.append(np.empty((0, 2), dtype=np.int32))
                continue

            token_slice = self._device_buffer[start:end]
            pair_tensor = torch.stack(
                (
                    layer_ids.expand(token_slice.shape[0], -1, -1),
                    token_slice,
                ),
                dim=-1,
            ).reshape(-1, 2)
            pair_tensor = pair_tensor[pair_tensor[:, 1] >= 0]
            if pair_tensor.numel() == 0:
                results.append(np.empty((0, 2), dtype=np.int32))
                continue
            results.append(torch.unique(pair_tensor, dim=0).cpu().numpy())

        return results

    def cleanup(self) -> None:
        """Explicitly clean up resources."""
        self._device_buffer = None

    def __del__(self) -> None:
        """Clean up resources on destruction."""
        self.cleanup()
