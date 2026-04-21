#!/bin/bash

MODEL=${1:-Qwen/Qwen1.5-MoE-A2.7B}

RANK=${RANK:-${OMPI_COMM_WORLD_RANK}}
WORLD_SIZE=${WORLD_SIZE:-${OMPI_COMM_WORLD_SIZE}}
HEAD_RANK=${HEAD_RANK:-0}

export NVSHMEM_DIR=/usr/local/nvshmem
export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
export DEEP_EP_CPU_TIMEOUT_SECS=${DEEP_EP_CPU_TIMEOUT_SECS:-600}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}
export VLLM_SIMPLE_COMPILE_BACKEND=${VLLM_SIMPLE_COMPILE_BACKEND:-eager}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

if [ "$RANK" -eq "$HEAD_RANK" ]; then
	EXTRA_ARGS="--api-server-count=1"
else
	EXTRA_ARGS="--headless"
fi

echo "$RANK, $HEAD_RANK"
vllm serve $MODEL \
	  --tensor-parallel-size 1 \
	  --data-parallel-size ${WORLD_SIZE} \
	  --data-parallel-size-local 1 \
	  --data-parallel-start-rank ${RANK} \
	  --enable-expert-parallel \
	  --all2all-backend deepep_high_throughput \
	  --trust_remote_code \
	  --data-parallel-address ${HEAD_IP} \
	  --data-parallel-rpc-port 18000 \
	  ${EXTRA_ARGS} \
	  ${VLLM_EXTRA_ARGS} \
	  --disable-custom-all-reduce \
      --expert-affinity-routing-weight 0.1 \
      --prefix-affinity-only-prefill \
      --kv-block-prefix-routing-weight 0.1 \
      --load-score-routing-weight 0.1 \
      --enable-eplb \
      --eplb-config '{"policy":"custom","use_async":true,"step_interval":3000,"window_size":1000}' \
	  --placement-callback-path ${PLACEMENT_PATH} \
	  --placement-callback-func compute_placement \
	  --load-balancer-debug \
      --enable-return-routed-experts \
      --enable-load-score-routing \
      --enable-prefix-affinity-routing \
      --enable-kv-block-prefix-routing \

	  # --max-model-len 200 \
