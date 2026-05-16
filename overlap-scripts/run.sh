#!/bin/bash

#MODEL=${1:-deepseek-ai/deepseek-moe-16b-chat}
#MODEL=${1:-mistralai/Mixtral-8x22B-v0.1}
#MODEL=${1:-mistralai/Mixtral-8x7B-v0.1}
MODEL=${1:-mistralai/Mixtral-8x7B-Instruct-v0.1}
#MODEL=${1:-microsoft/Phi-3.5-MoE-instruct}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# intra-node only for now
export NCCL_IB_DISABLE=1
# for eplb step interval runtime update and kv cache clearout
export VLLM_SERVER_DEV_MODE=1

#### NCCL ####
export NCCL_IB_GID_INDEX=3

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0

export NCCL_PXN_DISABLE=1
export NCCL_NET_GDR_LEVEL=SYS

#export NCCL_CHECKS_DISABLE=1
#export NCCL_RUNTIME_CONNECT=0

########

vllm serve $MODEL \
		--tensor-parallel-size 1 \
		--data-parallel-size 8 \
		--enable-expert-parallel \
		--trust_remote_code \
		--max_num_batched_tokens ${MAX_NUM_BATCHED_TOKENS} \
		--api-server-count=1 \
		--all2all-backend nccl_alltoall \
		--moe-backend triton \
		--enable-overlap \
		--overlap-decomposition-reorder none \
		# --enable-eplb \
		# --eplb-config '{"use_async":true,"step_interval":256,"window_size":1000,"num_redundant_experts":0}' \
		#--chat-template ${SCRIPT_DIR}/../examples/template_chatml.jinja \
		# --enable-overlap \
		# --overlap-decomposition-reorder johnson \
		# --overlap-johnson-estimate paper \
		# --overlap-comm-alpha 5.0 \
		# --overlap-comm-beta 3.3e-11 \
		# --overlap-comp-mfu 0.35 \
		# --overlap-comp-tflops 300


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done
