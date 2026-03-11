import os
import multiprocessing
from vllm import LLM, SamplingParams

# Configuration
MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TP_SIZE = 1        # Tensor Parallel size (keep as 1 for DP attention)
DP_SIZE = 8        # Data Parallel size (use all 8 GPUs)
GPU_MEMORY = 0.90  # Adjust as needed for L4 (24GB is tight but sufficient)

def run_inference(rank, prompts):
    # Set environment variables for Distributed / Expert Parallelism
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP_SIZE)
    
    # Initialize vLLM with Expert Parallelism enabled
    # enable_expert_parallel=True combined with DP_SIZE > 1 activates the "DP+EP" mode
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=True, 
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY,
        enforce_eager=False, # CUDA graphs are generally supported now
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
    outputs = llm.generate(prompts, sampling_params)

    # Print or save results
    for output in outputs:
        print(f"[Rank {rank}] Prompt: {output.prompt!r} | Generated: {output.outputs[0].text!r}")

if __name__ == "__main__":
    # Example prompts - split them among ranks manually
    all_prompts = [
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a poem about GPUs.",
        "How do MoE models work?",
        # ... add more prompts ...
    ] * 4

    # Divide prompts evenly among the 8 GPUs
    prompts_per_rank = [all_prompts[i::DP_SIZE] for i in range(DP_SIZE)]

    processes = []
    for rank in range(DP_SIZE):
        p = multiprocessing.Process(
            target=run_inference,
            args=(rank, prompts_per_rank[rank])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()