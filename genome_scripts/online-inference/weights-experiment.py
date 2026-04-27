import asyncio
import aiohttp
import time
import random
import csv
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import requests
from datetime import datetime

# --------------------------
# CONFIG
# --------------------------
HOST = "http://0.0.0.0:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

LOG_DIR = "logs"
CSV_FILE = "results.csv"

SYSTEM_PROMPT = "You are a precise assistant. Answer clearly and concisely."

tokenizer = AutoTokenizer.from_pretrained(MODEL)

os.makedirs(LOG_DIR, exist_ok=True)

#%%
def now_ns():
    return time.perf_counter_ns()

def poisson_interarrival(rate):
    return random.expovariate(rate)

def count_tokens(text):
    return len(tokenizer.encode(text))

#%%
def set_lb_weights(expert, kv, load, step_interval):
    requests.post(
        f"{HOST}/load_balancer/weights",
        json={
            "expert_affinity_routing_weight": expert,
            "kv_block_prefix_routing_weight": kv,
            "load_score_routing_weight": load,
            "step_interval": step_interval,
        },
        timeout=10,
    )

#%%
async def ask(session, sem, idx, ex):
    async with sem:
        arrival_ns = now_ns()

        full_prompt = SYSTEM_PROMPT + "\n" + ex["question"]
        input_tokens = count_tokens(full_prompt)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]}
            ],
            "max_tokens": 1,
            "temperature": 0.0
        }

        async with session.post(f"{HOST}/v1/chat/completions", json=payload) as resp:
            first_ns = now_ns()
            try:
                data = await resp.json()
                pred = data["choices"][0]["message"]["content"].strip()
            except:
                pred = "ERR"

        done_ns = now_ns()

        return {
            "idx": idx,
            "arrival_ns": arrival_ns,
            "first_ns": first_ns,
            "done_ns": done_ns,
            "ttft": (first_ns - arrival_ns) / 1e9,
            "latency": (done_ns - arrival_ns) / 1e9,
            "queue_delay": (first_ns - arrival_ns) / 1e9,
            "pred": pred,
            "gt": chr(65 + ex["answer"]),
            "input_tokens": input_tokens,
        }

#%%
async def run_dataset(name, subset, log_path):
    base_ds = load_dataset(name, subset, split="test")

    random.seed(1234)
    np.random.seed(1234)

    warmup_epochs = 3
    measured_epochs = 5
    rate = 100  # arrival rate

    ds = concatenate_datasets([base_ds] * (warmup_epochs + measured_epochs))
    warmup_count = len(base_ds) * warmup_epochs

    connector = aiohttp.TCPConnector(limit=200)
    sem = asyncio.Semaphore(50)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        next_time = now_ns()

        for i, ex in enumerate(ds):
            next_time += int(poisson_interarrival(rate) * 1e9)

            while True:
                now = now_ns()
                if now >= next_time:
                    break

                delta = next_time - now
                if delta > 50_000:
                    await asyncio.sleep(delta / 1e9 / 2)

            tasks.append(asyncio.create_task(ask(session, sem, i, ex)))

        results = await asyncio.gather(*tasks)


    results.sort(key=lambda x: x["arrival_ns"])

    with open(log_path, "w") as f:
        f.write("idx,arrival_ns,first_ns,done_ns,ttft,latency,input_tokens,pred\n")
        for r in results:
            f.write(
                f"{r['idx']},{r['arrival_ns']},{r['first_ns']},{r['done_ns']},"
                f"{r['ttft']},{r['latency']},{r['input_tokens']},{r['pred']}\n"
            )

    measured = results[warmup_count:]

    ttfts = np.array([r["ttft"] for r in measured])

    start_ns = measured[0]["arrival_ns"]
    end_ns = measured[-1]["done_ns"]

    effective_time = (end_ns - start_ns) / 1e9

    total_input_tokens = sum(r["input_tokens"] for r in measured)

    prefill_throughput = total_input_tokens / effective_time

    avg_ttft = ttfts.mean()
    p50 = np.percentile(ttfts, 50)
    p95 = np.percentile(ttfts, 95)
    p99 = np.percentile(ttfts, 99)

    return prefill_throughput, avg_ttft, p50, p95, p99

#%%
async def sweep():
    num_steps = 5
    step_interval = 64

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "expert", "kv", "load",
            "prefill_throughput",
            "avg_ttft", "p50_ttft", "p95_ttft", "p99_ttft"
        ])

        for i in range(num_steps + 1):
            for j in range(num_steps + 1 - i):
                k = num_steps - i - j

                expert = i / num_steps
                kv = j / num_steps
                load = k / num_steps

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = f"{LOG_DIR}/exp_{expert:.2f}_{kv:.2f}_{load:.2f}_{ts}.log"

                print(f"\n### Running: expert={expert:.2f}, kv={kv:.2f}, load={load:.2f} ###")

                set_lb_weights(expert, kv, load, step_interval)

                prefill_tp, avg_ttft, p50, p95, p99 = await run_dataset(
                    "cais/mmlu", "abstract_algebra", log_path
                )

                writer.writerow([
                    expert, kv, load,
                    prefill_tp,
                    avg_ttft, p50, p95, p99
                ])
                f.flush()

#%%
if __name__ == "__main__":
    print("This was a bad idea to iterate over all experiments in a single launch!")
    print("This can silently benefit KV cache hit rate since many runs may reuse the cache and distort results..!")
    print("Note: Generally, it is best to relaunch vLLM, or find a way to clear out kv caches and any other state across iterations.")
    print("Exiting....! Use this script for quick tests, but do not rely on results")

    asyncio.run(sweep())