#!/bin/bash


HEAD_RANK=0
HEAD_IP=10.10.0.2

MODEL=${1:-Qwen/Qwen1.5-MoE-A2.7B}

# nswrap is a custom script on our server with NCCL, NVSHMEM envs and passes the rank and world size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mpirun -np 8 -H \
10.201.0.2,\
10.201.1.2,\
10.201.2.2,\
10.201.3.2,\
10.201.4.2,\
10.201.5.2,\
10.201.6.2,\
10.201.7.2 \
	-x RANK=$RANK \
	-x WORLD_SIZE=$WORLD_SIZE \
	-x HEAD_RANK=$HEAD_RANK \
	-x HEAD_IP=$HEAD_IP \
	--map-by ppr:1:node \
	nswrap --python ${SCRIPT_DIR}/.venv ${SCRIPT_DIR}/vllm-ns-worker.sh $MODEL