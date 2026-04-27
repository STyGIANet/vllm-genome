import asyncio
import aiohttp
import time
import random
import csv
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import requests
from datetime import datetime

from prompt_datasets import WEIGHTS_EXPERIMENT_DATASETS

# --------------------------
# CONFIG
# --------------------------
HOST = "http://0.0.0.0:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

LOG_DIR = "logs"
CSV_FILE = "results.csv"

SYSTEM_PROMPT = "You are a precise assistant. Answer clearly and concisely."
POISSON_RATE_MODE = "tok_per_sec"
POISSON_REQUESTS_PER_SEC = 1
# POISSON_INPUT_TOKENS_PER_SEC = None
POISSON_INPUT_TOKENS_PER_SEC = 1000

tokenizer = AutoTokenizer.from_pretrained(MODEL)

os.makedirs(LOG_DIR, exist_ok=True)

#%%
def now_ns():
    return time.perf_counter_ns()

def poisson_interarrival_req_sec(requests_per_sec):
    if requests_per_sec <= 0:
        raise ValueError("requests_per_sec must be > 0")
    return random.expovariate(requests_per_sec)


def poisson_interarrival_tok_sec(tokens_per_sec, input_tokens):
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be > 0")
    return random.expovariate(tokens_per_sec / max(input_tokens, 1))


def count_message_tokens(messages):
    try:
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    except Exception:
        return sum(len(tokenizer.encode(msg["content"])) for msg in messages)


def resolve_poisson_rate(rate_mode, mean_input_tokens):
    if rate_mode == "req_per_sec":
        return POISSON_REQUESTS_PER_SEC
    if rate_mode == "tok_per_sec":
        if POISSON_INPUT_TOKENS_PER_SEC is not None:
            return POISSON_INPUT_TOKENS_PER_SEC
        return POISSON_REQUESTS_PER_SEC * mean_input_tokens
    raise ValueError(f"Unsupported rate mode: {rate_mode}")


def sample_poisson_interarrival(rate, rate_mode, input_tokens):
    if rate_mode == "req_per_sec":
        return poisson_interarrival_req_sec(rate)
    if rate_mode == "tok_per_sec":
        return poisson_interarrival_tok_sec(rate, input_tokens)
    raise ValueError(f"Unsupported rate mode: {rate_mode}")


async def sleep_until_ns(target_ns):
    while True:
        remaining_ns = target_ns - now_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            await asyncio.sleep((remaining_ns / 1e9) * 0.5)
        elif remaining_ns > 50_000:
            await asyncio.sleep(remaining_ns / 1e9)
        else:
            await asyncio.sleep(0)


def build_messages(ex):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["question"]},
    ]


DATASETS = WEIGHTS_EXPERIMENT_DATASETS


def prepare_example(ex):
    messages = build_messages(ex)
    return {
        **ex,
        "messages": messages,
        "input_tokens": count_message_tokens(messages),
    }

#%%
def set_vllm_config(expert, kv, load, step_interval):
    resp = requests.post(
      f"{HOST}/load_balancer/weights",
      json={
          "expert_affinity_routing_weight": expert,
          "kv_block_prefix_routing_weight": kv,
          "load_score_routing_weight": load,
          "eplb_step_interval": step_interval, # this is a copy in the frontend
      },
      timeout=10,
    )
    resp.raise_for_status()
    print("POST /load_balancer/weights ->", resp.json())


    # eplb frequency of expert placement updates
    # number of engine steps / forward passes
    resp = requests.post(
      f"{HOST}/eplb/step_interval",
      json={"step_interval": step_interval},
      timeout=10,
    )
    resp.raise_for_status()
    print("POST /eplb/step_interval ->", resp.json())

    # Clearing out kv so that the next experiment does not reuse previous ones
    resp = requests.post(
      f"{HOST}/reset_prefix_cache",
      params={
          "reset_running_requests": True,
          "reset_external": True,
      },
      timeout=10,
    )
    resp.raise_for_status()
    print("POST /reset_prefix_cache -> 200 OK")

    
    # Checking if the updates are applied
    resp = requests.get(
      f"{HOST}/load_balancer/weights",
      timeout=10,
    )
    resp.raise_for_status()
    print("GET /load_balancer/weights ->", resp.json())

    resp = requests.get(
      f"{HOST}/eplb/step_interval",
      timeout=10,
    )
    resp.raise_for_status()
    print("GET /eplb/step_interval ->", resp.json())

#%%
async def ask(session, sem, idx, ex, scheduled_arrival_ns):
    async with sem:
        payload = {
            "model": MODEL,
            "messages": ex["messages"],
            "max_tokens": 1,
            "temperature": 0.0,
        }

        dispatch_ns = now_ns()

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
            "scheduled_arrival_ns": scheduled_arrival_ns,
            "dispatch_ns": dispatch_ns,
            "first_ns": first_ns,
            "done_ns": done_ns,
            "ttft": (first_ns - scheduled_arrival_ns) / 1e9,
            "latency": (done_ns - scheduled_arrival_ns) / 1e9,
            "queue_delay": (dispatch_ns - scheduled_arrival_ns) / 1e9,
            "service_ttft": (first_ns - dispatch_ns) / 1e9,
            "service_latency": (done_ns - dispatch_ns) / 1e9,
            "pred": pred,
            "gt": chr(65 + ex["answer"]),
            "input_tokens": ex["input_tokens"],
        }

#%%
async def poisson_driver(session, sem, examples, rate, rate_mode):
    tasks = []
    next_arrival_ns = now_ns()

    for i, ex in enumerate(examples):
        next_arrival_ns += int(
            sample_poisson_interarrival(rate, rate_mode, ex["input_tokens"]) * 1e9
        )
        await sleep_until_ns(next_arrival_ns)
        tasks.append(asyncio.create_task(ask(session, sem, i, ex, next_arrival_ns)))

    return await asyncio.gather(*tasks)

#%%
async def run_dataset(name, subset, formatter, split, log_path):
    base_ds = load_dataset(name, subset, split=split)
    base_ds = [formatter(ex) for ex in base_ds]

    random.seed(1234)
    np.random.seed(1234)

    warmup_epochs = 3
    measured_epochs = 5
    prepared_base = [prepare_example(ex) for ex in base_ds if ex is not None]
    warmup_count = len(prepared_base) * warmup_epochs
    examples = prepared_base * (warmup_epochs + measured_epochs)
    mean_input_tokens = (
        sum(ex["input_tokens"] for ex in prepared_base) / len(prepared_base)
        if prepared_base else 0.0
    )
    rate = resolve_poisson_rate(POISSON_RATE_MODE, mean_input_tokens)
    rate_units = "tok/s" if POISSON_RATE_MODE == "tok_per_sec" else "req/s"
    print(
        f"Poisson arrivals: mode={POISSON_RATE_MODE} rate={rate:.2f} {rate_units} "
        f"(mean_input_tokens={mean_input_tokens:.2f})"
    )

    connector = aiohttp.TCPConnector(limit=200)
    sem = asyncio.Semaphore(50)

    async with aiohttp.ClientSession(connector=connector) as session:
        results = await poisson_driver(
            session,
            sem,
            examples,
            rate,
            POISSON_RATE_MODE,
        )


    results.sort(key=lambda x: x["scheduled_arrival_ns"])

    with open(log_path, "w") as f:
        f.write(
            "idx,scheduled_arrival_ns,dispatch_ns,first_ns,done_ns,"
            "queue_delay,ttft,service_ttft,latency,service_latency,input_tokens,pred\n"
        )
        for r in results:
            f.write(
                f"{r['idx']},{r['scheduled_arrival_ns']},{r['dispatch_ns']},"
                f"{r['first_ns']},{r['done_ns']},{r['queue_delay']},"
                f"{r['ttft']},{r['service_ttft']},{r['latency']},"
                f"{r['service_latency']},{r['input_tokens']},{r['pred']}\n"
            )

    measured = results[warmup_count:]

    ttfts = np.array([r["ttft"] for r in measured])

    start_ns = measured[0]["scheduled_arrival_ns"]
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
            "dataset", "subset",
            "expert", "kv", "load",
            "prefill_throughput",
            "avg_ttft", "p50_ttft", "p95_ttft", "p99_ttft"
        ])

        for dataset_name, dataset_subset, dataset_formatter, dataset_split in DATASETS:
            dataset_label = dataset_name.replace("/", "_")
            subset_label = (dataset_subset or "none").replace("/", "_")

            for i in range(num_steps + 1):
                for j in range(num_steps + 1 - i):
                    k = num_steps - i - j

                    expert = i / num_steps
                    kv = j / num_steps
                    load = k / num_steps

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_path = (
                        f"{LOG_DIR}/{dataset_label}_{subset_label}_"
                        f"{expert:.2f}_{kv:.2f}_{load:.2f}_{ts}.log"
                    )

                    print(
                        f"\n### Running: dataset={dataset_name}/{dataset_subset} "
                        f"expert={expert:.2f}, kv={kv:.2f}, load={load:.2f} ###"
                    )

                    set_vllm_config(expert, kv, load, step_interval)

                    prefill_tp, avg_ttft, p50, p95, p99 = await run_dataset(
                        dataset_name,
                        dataset_subset,
                        dataset_formatter,
                        dataset_split,
                        log_path,
                    )

                    writer.writerow([
                        dataset_name,
                        dataset_subset,
                        expert, kv, load,
                        prefill_tp,
                        avg_ttft, p50, p95, p99
                    ])
                    f.flush()

#%%
if __name__ == "__main__":
    asyncio.run(sweep())
