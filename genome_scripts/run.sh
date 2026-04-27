#!/bin/bash

MODEL=${1:-deepseek-ai/deepseek-moe-16b-chat}

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLACEMENT_PATH="${SCRIPT_DIR}/expert-placement/placement_fns.py"
export VLLM_SERVER_DEV_MODE=1

# export NVSHMEM_DIR=/usr/local/nvshmem
# export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
# export DEEP_EP_CPU_TIMEOUT_SECS=${DEEP_EP_CPU_TIMEOUT_SECS:-600}
# export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}
# export VLLM_SIMPLE_COMPILE_BACKEND=${VLLM_SIMPLE_COMPILE_BACKEND:-eager}
# export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# export VLLM_PREFIX_LEARNING_DUMMY_OWNER=1

vllm serve $MODEL \
		--tensor-parallel-size 1 \
		--data-parallel-size 8 \
		--enable-expert-parallel \
		--all2all-backend deepep_high_throughput \
		--trust_remote_code \
		--max_num_batched_tokens 2048 \
		--api-server-count=1 \
		--expert-affinity-routing-weight 1 \
		--kv-block-prefix-routing-weight 0.5 \
		--load-score-routing-weight 0.5 \
		--enable-eplb \
		--max-pending-requests-per-engine 256 \
		--enable-load-score-routing \
		--enable-kv-block-prefix-routing \
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement \
		--enable-prefix-affinity-routing \
		--prefix-affinity-only-prefill \
		--prefix-learning-algorithm prefixtrie \
		#--load-balancer-debug \


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done


      # --max-pending-requests-per-engine set this to a small number for now 
      # until the load balancer is stable and works as expected.
      # Any sudden severe imbalances can cause async issues and deepep will throw errors.

      # --max-pending-requests controls the total number of requests pending across all engines.

      # --enable-prefix-affinity-routing this enables expert-aware load balancing
      # --prefix-affinity-only-prefill use this in conjunction with the above, as our current lb is prefix-based

      # --enable-kv-block-prefix-routing this directly takes vllm reported kv blocks and normalizes to a score
      
      # --enable-load-score-routing this is waiting *4 + running for capturing load (same as how vllm does)
      
      # These three are weights in the convex combination of the individual scores
      # --expert-affinity-routing-weight 
      # --kv-block-prefix-routing-weight
      # --load-score-routing-weight


	  # No longer needed. setting EPLB custom policy is enough
      # --enable-return-routed-experts \
      # 
	  # --disable-custom-all-reduce \
	  # --max-model-len 200 \
