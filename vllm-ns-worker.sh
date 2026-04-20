#!/bin/bash

MODEL=${1:-Qwen/Qwen1.5-MoE-A2.7B}

export NVSHMEM_DIR=/usr/local/nvshmem
export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}

if [ "$RANK" -eq "$HEAD_RANK" ]; then
	echo "using api-server-count for $RANK"
  EXTRA_ARGS="--api-server-count=1"
else
	echo "using --headless for $RANK"
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
	  --max-model-len 200 \
	  --data-parallel-address ${HEAD_IP} \
	  --data-parallel-rpc-port 18000 \
	  ${EXTRA_ARGS} \
	  --disable-custom-all-reduce
