#!/bin/bash

MODEL=${1:-deepseek-ai/deepseek-moe-16b-chat}

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLACEMENT_PATH="${SCRIPT_DIR}/expert-placement/placement_fns.py"

# for runtime updates
export VLLM_SERVER_DEV_MODE=1

export TRACE_DIR=${SCRIPT_DIR}/traces/
export TRAFFIC_DIR=${SCRIPT_DIR}/traffic/
# mkdir -p ${TRACE_DIR}
# mkdir -p ${TRAFFIC_DIR}
source ${SCRIPT_DIR}/../.venv/bin/activate

##############################################################
echo "Running Moor 0 0 0 experiment"
mkdir -p ${SCRIPT_DIR}/summary-moor-0-0-0

(vllm serve $MODEL \
		--tensor-parallel-size 8 \
		--data-parallel-size 1 \
		--enable-expert-parallel \
		--all2all-backend deepep_high_throughput \
		--trust_remote_code \
		--max_num_batched_tokens 16384 \
		--api-server-count=1 \
		--expert-affinity-routing-weight 1 \
		--kv-block-prefix-routing-weight 0.5 \
		--load-score-routing-weight 0.5 \
		--enable-eplb \
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement  > ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt) &

while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done

cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 0 0 0 128 ${SCRIPT_DIR}/traces-moor-0-0-0/ ${SCRIPT_DIR}/traffic-moor-0-0-0/ ${SCRIPT_DIR}/summary-moor-0-0-0/ \
	> ${SCRIPT_DIR}/summary-moor-0-0-0/all-results.txt 2> ${SCRIPT_DIR}/summary-moor-0-0-0/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done

##############################################################
echo "Running MoorVertex 0 0 0 experiment"
mkdir -p ${SCRIPT_DIR}/summary-moorvertex-0-0-0

(vllm serve $MODEL \
		--tensor-parallel-size 8 \
		--data-parallel-size 1 \
		--enable-expert-parallel \
		--all2all-backend deepep_high_throughput \
		--trust_remote_code \
		--max_num_batched_tokens 16384 \
		--api-server-count=1 \
		--expert-affinity-routing-weight 1 \
		--kv-block-prefix-routing-weight 0.5 \
		--load-score-routing-weight 0.5 \
		--enable-eplb \
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path "${SCRIPT_DIR}/expert-placement/placement_fns_vweights.py" \
		--placement-callback-func compute_placement  > ${SCRIPT_DIR}/summary-moorvertex-0-0-0/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moorvertex-0-0-0/vllm-log.txt) &

while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done

cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 0 0 0 128 ${SCRIPT_DIR}/traces-moorvertex-0-0-0/ ${SCRIPT_DIR}/traffic-moorvertex-0-0-0/ ${SCRIPT_DIR}/summary-moorvertex-0-0-0/ \
	> ${SCRIPT_DIR}/summary-moorvertex-0-0-0/all-results.txt 2> ${SCRIPT_DIR}/summary-moorvertex-0-0-0/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done

##############################################################
echo "Running EPLB experiment"
mkdir -p ${SCRIPT_DIR}/summary-eplb

(vllm serve $MODEL \
		--tensor-parallel-size 8 \
		--data-parallel-size 1 \
		--enable-expert-parallel \
		--all2all-backend deepep_high_throughput \
		--trust_remote_code \
		--max_num_batched_tokens 16384 \
		--api-server-count=1 \
		--expert-affinity-routing-weight 1 \
		--kv-block-prefix-routing-weight 0.5 \
		--load-score-routing-weight 0.5 \
		--enable-eplb \
		--eplb-config '{"use_async":false,"step_interval":30,"window_size":1000,"num_redundant_experts":0}' > ${SCRIPT_DIR}/summary-eplb/vllm-log.txt 2> ${SCRIPT_DIR}/summary-eplb/vllm-log.txt )&
		# Crashing with async

while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 0 0 0 128 ${SCRIPT_DIR}/traces-eplb/ ${SCRIPT_DIR}/traffic-eplb/ ${SCRIPT_DIR}/summary-eplb/ \
	> ${SCRIPT_DIR}/summary-eplb/all-results.txt 2> ${SCRIPT_DIR}/summary-eplb/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done

##############################################################
echo "Running vLLM experiment"
mkdir -p ${SCRIPT_DIR}/summary-vllm

(vllm serve $MODEL \
		--tensor-parallel-size 8 \
		--data-parallel-size 1 \
		--enable-expert-parallel \
		--all2all-backend deepep_high_throughput \
		--trust_remote_code \
		--max_num_batched_tokens 16384 \
		--api-server-count=1 \
		--expert-affinity-routing-weight 1 \
		--kv-block-prefix-routing-weight 0.5 \
		--load-score-routing-weight 0.5 \
		 > ${SCRIPT_DIR}/summary-vllm/vllm-log.txt 2> ${SCRIPT_DIR}/summary-vllm/vllm-log.txt )&

while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 0 0 0 128 ${SCRIPT_DIR}/traces-vllm/ ${SCRIPT_DIR}/traffic-vllm/ ${SCRIPT_DIR}/summary-vllm/ \
	> ${SCRIPT_DIR}/summary-vllm/all-results.txt 2> ${SCRIPT_DIR}/summary-vllm/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done

##################### Exiting here ##########################
exit

##############################################################
echo "Running Moor 1 2 3 experiment"
mkdir -p ${SCRIPT_DIR}/summary-moor-1-2-3

(vllm serve $MODEL \
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
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement \
		--max-pending-requests-per-engine 256 \
		--enable-load-score-routing \
		--enable-kv-block-prefix-routing \
		--enable-prefix-affinity-routing \
		--prefix-affinity-only-prefill \
		--prefix-learning-algorithm prefixtrie > ${SCRIPT_DIR}/summary-moor-1-2-3/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moor-1-2-3/vllm-log.txt )&


while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 1 2 3 128 ${SCRIPT_DIR}/traces-moor-1-2-3/ ${SCRIPT_DIR}/traffic-moor-1-2-3/ ${SCRIPT_DIR}/summary-moor-1-2-3/ \
	> ${SCRIPT_DIR}/summary-moor-1-2-3/all-results.txt 2> ${SCRIPT_DIR}/summary-moor-1-2-3/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done


##############################################################
echo "Running Moor 3 2 1 experiment"
mkdir -p ${SCRIPT_DIR}/summary-moor-3-2-1

(vllm serve $MODEL \
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
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement \
		--max-pending-requests-per-engine 256 \
		--enable-load-score-routing \
		--enable-kv-block-prefix-routing \
		--enable-prefix-affinity-routing \
		--prefix-affinity-only-prefill \
		--prefix-learning-algorithm prefixtrie > ${SCRIPT_DIR}/summary-moor-3-2-1/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moor-3-2-1/vllm-log.txt )&


while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 3 2 1 128 ${SCRIPT_DIR}/traces-moor-3-2-1/ ${SCRIPT_DIR}/traffic-moor-3-2-1/ ${SCRIPT_DIR}/summary-moor-3-2-1/ \
	> ${SCRIPT_DIR}/summary-moor-3-2-1/all-results.txt 2> ${SCRIPT_DIR}/summary-moor-3-2-1/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done


##############################################################
echo "Running Moor 2 3 1 experiment"
mkdir -p ${SCRIPT_DIR}/summary-moor-2-3-1

(vllm serve $MODEL \
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
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement \
		--max-pending-requests-per-engine 256 \
		--enable-load-score-routing \
		--enable-kv-block-prefix-routing \
		--enable-prefix-affinity-routing \
		--prefix-affinity-only-prefill \
		--prefix-learning-algorithm prefixtrie > ${SCRIPT_DIR}/summary-moor-2-3-1/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moor-2-3-1/vllm-log.txt ) &


while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 2 3 1 128 ${SCRIPT_DIR}/traces-moor-2-3-1/ ${SCRIPT_DIR}/traffic-moor-2-3-1/ ${SCRIPT_DIR}/summary-moor-2-3-1/ \
	> ${SCRIPT_DIR}/summary-moor-2-3-1/all-results.txt 2> ${SCRIPT_DIR}/summary-moor-2-3-1/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done



##############################################################
echo "Running tracing experiment"
mkdir -p ${SCRIPT_DIR}/summary-moor-tracing

export TRACE_DIR=${SCRIPT_DIR}/traces-moor-tracing/
export TRAFFIC_DIR=${SCRIPT_DIR}/traffic-moor-tracing/

(vllm serve $MODEL \
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
		--eplb-config '{"policy":"custom","use_async":true,"step_interval":30,"window_size":1000,"num_redundant_experts":0,"graph_ema_alpha":0.5}' \
		--placement-callback-path ${PLACEMENT_PATH} \
		--placement-callback-func compute_placement \
		--moe-dispatch-traffic-dump-dir ${TRAFFIC_DIR} \
		--placement-routing-dump-dir ${TRACE_DIR} > ${SCRIPT_DIR}/summary-moor-tracing/vllm-log.txt 2> ${SCRIPT_DIR}/summary-moor-tracing/vllm-log.txt ) &


while [[ $(cat ${SCRIPT_DIR}/summary-moor-0-0-0/vllm-log.txt | grep 'completions') == "" ]];do
	echo "Waiting for vLLM to start..."
	sleep 1
done


cd ${SCRIPT_DIR}/online-inference/
python3 send-prompts.py 0 0 0 64 ${TRACE_DIR} ${TRAFFIC_DIR} ${SCRIPT_DIR}/summary-moor-tracing/ \
	> ${SCRIPT_DIR}/summary-moor-tracing/all-results.txt 2> ${SCRIPT_DIR}/summary-moor-tracing/all-results.txt


# cleanup
for i in $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | awk '{print $1}' | awk -F ',' '{print $1}');do kill -9 $i;done