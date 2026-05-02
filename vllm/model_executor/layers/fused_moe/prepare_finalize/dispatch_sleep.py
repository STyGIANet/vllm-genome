# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

_FUSED_MOE_SLEEP_BEFORE_DISPATCH_S = (
    float(os.getenv("VLLM_FUSED_MOE_SLEEP_BEFORE_DISPATCH_MS", "0.0"))
    / 1000.0
)


def maybe_sleep_before_dispatch() -> None:
    if _FUSED_MOE_SLEEP_BEFORE_DISPATCH_S > 0.0:
        time.sleep(_FUSED_MOE_SLEEP_BEFORE_DISPATCH_S)
