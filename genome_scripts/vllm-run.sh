#!/bin/bash


HEAD_RANK=0
HEAD_IP=10.10.0.2

MODEL=${1:-deepseek-ai/deepseek-moe-16b-chat}

# nswrap is a custom script on our server with NCCL, NVSHMEM envs and passes the rank and world size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$SCRIPT_DIR/DeepEP-SM8x:$PYTHONPATH
mpirun -np 8 -H \
10.201.0.2,\
10.201.1.2,\
10.201.2.2,\
10.201.3.2,\
10.201.4.2,\
10.201.5.2,\
10.201.6.2,\
10.201.7.2,\
	-x RANK=$RANK \
	-x WORLD_SIZE=$WORLD_SIZE \
	-x HEAD_RANK=$HEAD_RANK \
	-x HEAD_IP=$HEAD_IP \
	-x VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" \
	-x VLLM_SKIP_DEEPEP_PROFILE="${VLLM_SKIP_DEEPEP_PROFILE}" \
	-x VLLM_SKIP_DEEPEP_WARMUP="${VLLM_SKIP_DEEPEP_WARMUP}" \
	-x VLLM_SKIP_DEEPEP_DUMMY_BATCH="${VLLM_SKIP_DEEPEP_DUMMY_BATCH}" \
	-x CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING}" \
	--map-by ppr:1:node \
	-x PLACEMENT_PATH="${SCRIPT_DIR}/expert-placement/placement_fns.py" \
	nswrap --python ${SCRIPT_DIR}/.venv ${SCRIPT_DIR}/vllm-ns-worker.sh $MODEL
