import asyncio
import aiohttp
import time
import random
import requests
import sys
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from prompt_datasets import SEND_PROMPTS_DATASETS, SYSTEM_PROMPTS
# Change which prompt datasets to test in prompt_datasets.py at the end of the file

HOST = "http://0.0.0.0:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

POISSON_RATE_MODE = "tok_per_sec"
POISSON_REQUESTS_PER_SEC = 1.0
# POISSON_INPUT_TOKENS_PER_SEC = None
POISSON_INPUT_TOKENS_PER_SEC = 16000

tokenizer = AutoTokenizer.from_pretrained(MODEL)

rng = random.Random(10)

def pick_sysprompt_random(prompts):
    idx = rng.randrange(len(prompts))
    return prompts[idx]


def extract_choice(text, num_choices):
    text = text.strip().upper()
    valid = [chr(65 + i) for i in range(num_choices)]
    if text in valid:
        return text

    match = re.search(r"[A-Z]", text)
    if match:
        letter = match.group(0)
        if letter in valid:
            return letter

    return "INVALID"


def poisson_interarrival_req_sec(requests_per_sec):
    if requests_per_sec <= 0:
        raise ValueError("requests_per_sec must be > 0")
    return rng.expovariate(requests_per_sec)


def poisson_interarrival_tok_sec(tokens_per_sec, input_tokens):
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be > 0")
    return rng.expovariate(tokens_per_sec / max(input_tokens, 1))


def now_ns():
    return time.perf_counter_ns()


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


def build_user_prompt(ex):
    choices = ex["choices"]
    letters = [chr(65 + i) for i in range(len(choices))]
    options = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])

    if len(choices) <= 4:
        return f"{ex['question']}\n{options}\nAnswer with ONLY one letter: {', '.join(letters)}."
    return f"{ex['question']}\nAnswer concisely."


def prepare_example(ex):
    user_prompt = build_user_prompt(ex)
    messages = [
        {"role": "system", "content": pick_sysprompt_random(SYSTEM_PROMPTS)},
        {"role": "user", "content": user_prompt},
    ]
    return {
        **ex,
        "user_prompt": user_prompt,
        "messages": messages,
        "input_tokens": count_message_tokens(messages),
    }


async def ask(session, sem, idx, ex, scheduled_arrival_ns):
    async with sem:
        choices = ex["choices"]
        answer = ex["answer"]
        payload = {
            "model": MODEL,
            "messages": ex["messages"],
            "max_tokens": 1,
            "temperature": 0.0,
        }

        dispatch_ns = now_ns()

        async with session.post(f"{HOST}/v1/chat/completions", json=payload) as resp:
            first_byte_ns = now_ns()

            try:
                data = await resp.json()
                raw_pred = data["choices"][0]["message"]["content"]
                pred = extract_choice(raw_pred, len(choices))
                # pred = data["choices"][0]["message"]["content"].strip()
            except Exception:
                pred = "ERROR"

        completion_ns = now_ns()
        # Queueing here just the async io's queueing, not on the inference side
        queue_delay = (dispatch_ns - scheduled_arrival_ns) / 1e9
        ttft = (first_byte_ns - scheduled_arrival_ns) / 1e9
        total_latency = (completion_ns - scheduled_arrival_ns) / 1e9
        service_ttft = (first_byte_ns - dispatch_ns) / 1e9
        service_latency = (completion_ns - dispatch_ns) / 1e9

        input_tokens = ex["input_tokens"]
        output_tokens = 1

        print(
            f"[{idx}] QD={queue_delay:.4f}s TTFT={ttft:.4f}s "
            f"LAT={total_latency:.4f}s SVC_TTFT={service_ttft:.4f}s "
            f"SVC_LAT={service_latency:.4f}s Pred={pred}, Tokens={input_tokens + output_tokens}",
            flush=True
        )

        return {
            "pred": pred,
            "gt": chr(65 + answer),
            "latency": total_latency,
            "ttft": ttft,
            "service_ttft": service_ttft,
            "service_latency": service_latency,
            "queue_delay": queue_delay,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "scheduled_arrival_ns": scheduled_arrival_ns,
            "dispatch_ns": dispatch_ns,
            "first_byte_ns": first_byte_ns,
            "completion_ns": completion_ns,
        }


async def poisson_driver(examples, rate, rate_mode, warmup_count):
    connector = aiohttp.TCPConnector(limit=100)
    sem = asyncio.Semaphore(40)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        next_arrival_ns = now_ns()

        for i, ex in enumerate(examples):
            next_arrival_ns += int(
                sample_poisson_interarrival(rate, rate_mode, ex["input_tokens"]) * 1e9
            )
            await sleep_until_ns(next_arrival_ns)
            tasks.append(asyncio.create_task(ask(session, sem, i, ex, next_arrival_ns)))

        results = await asyncio.gather(*tasks)
    measured = results[warmup_count:]
    if not measured:
        return measured, 0.0

    effective_time = (
        measured[-1]["completion_ns"] - measured[0]["scheduled_arrival_ns"]
    ) / 1e9

    return measured, effective_time


async def run_dataset(name, subset, formatter, split):
    print(f"\n### Running: {name} / {subset} ###")

    if subset is None:
        ds = load_dataset(name)
    else:
        ds = load_dataset(name, subset)

    # split = "test" if "test" in ds else "validation"
    base_ds = ds[split]
    # Setting 200 for now just for quick experiments. TODO: Remove it.
    base_ds = base_ds.select(range(400))
    print("Dataset split size:", len(base_ds))

    base_ds = base_ds.map(formatter)
    examples = [prepare_example(ex) for ex in base_ds if ex is not None]

    print("Prompts:", len(examples))

    warmup_epochs = 0
    measured_epochs = 1

    num_repeats = warmup_epochs + measured_epochs
    examples_full = examples * num_repeats
    warmup_count = len(examples) * warmup_epochs

    mean_input_tokens = (
        sum(ex["input_tokens"] for ex in examples) / len(examples) if examples else 0.0
    )
    rate = resolve_poisson_rate(POISSON_RATE_MODE, mean_input_tokens)
    rate_units = "tok/s" if POISSON_RATE_MODE == "tok_per_sec" else "req/s"
    print(
        f"Poisson arrivals: mode={POISSON_RATE_MODE} rate={rate:.2f} {rate_units} "
        f"(mean_input_tokens={mean_input_tokens:.2f})"
    )

    measured, total_time = await poisson_driver(
        examples_full,
        rate,
        POISSON_RATE_MODE,
        warmup_count,
    )

    total_input = sum(r["input_tokens"] for r in measured)
    total_output = sum(r["output_tokens"] for r in measured)

    avg_ttft = sum(r["ttft"] for r in measured) / len(measured)
    service_ttft = sum(r["service_ttft"] for r in measured) / len(measured)
    throughput = (total_input + total_output) / total_time
    # correct = sum(1 for r in measured if r["pred"].startswith(r["gt"]))
    correct = sum(1 for r in measured if r["pred"] == r["gt"])

    stats = {
        "dataset": name,
        "subset": subset,
        "requests": len(measured),
        "avg_ttft": avg_ttft,
        "service_ttft": service_ttft,
        "throughput": throughput,
        "accuracy": correct / len(measured),
    }

    print(f"TTFT={avg_ttft:.4f}s | SVC_TTFT={service_ttft:.4f} | Throughput={throughput:.1f} tok/s | Acc={stats['accuracy']:.3f}")

    return stats

def set_vllm_config(expert, kv, load, step_interval):
    resp = requests.post(
      f"{HOST}/load_balancer/weights",
      json={
          "kv_block_prefix_routing_weight": kv,
          "load_score_routing_weight": load,
          "eplb_step_interval": step_interval, # this is a copy in the frontend
          "expert_affinity_routing_weight": expert,
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



DATASETS = SEND_PROMPTS_DATASETS


async def main():
    results = []

    for name, subset, formatter, split in DATASETS:
        stats = await run_dataset(name, subset, formatter, split)
        results.append(stats)
        print(
            f"{stats['dataset']} / {stats['subset']} | "
            f"TTFT={stats['avg_ttft']:.3f}s | "
            f"SVC_TTFT={stats['service_ttft']:.3f} |"
            f"Throughput={stats['throughput']:.1f} tok/s | "
            f"Acc={stats['accuracy']:.3f}"
        )

    print("\n### FINAL SUMMARY ###")
    for r in results:
        print(
            f"{r['dataset']} / {r['subset']} | "
            f"TTFT={r['avg_ttft']:.3f}s | "
            f"SVC_TTFT={r['service_ttft']:.3f} |"
            f"Throughput={r['throughput']:.1f} tok/s | "
            f"Acc={r['accuracy']:.3f}"
        )

expert_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
kv_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.33
load_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.34
step_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 30

print(f"Setting LB weights: expert={expert_weight} kv={kv_weight} load={load_weight}")
set_vllm_config(expert_weight, kv_weight, load_weight, step_interval)

asyncio.run(main())
