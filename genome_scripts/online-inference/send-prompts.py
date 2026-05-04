import asyncio
import aiohttp
import time
import random
import requests
import sys
import re
import os
import math

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from prompt_datasets import (
    ALL_DATASETS,
    MIXED_DATASET_SHUFFLE_SEED,
    MIXED_DATASET_STRIDE,
    MIXED_HOTPOT_BOOLQ_NAME,
    SEND_PROMPTS_DATASETS,
    SYSTEM_PROMPTS,
)
# Change which prompt datasets to test in prompt_datasets.py at the end of the file

HOST = "http://0.0.0.0:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

POISSON_RATE_MODE = "tok_per_sec"
POISSON_REQUESTS_PER_SEC = 1.0
# POISSON_INPUT_TOKENS_PER_SEC = None
POISSON_INPUT_TOKENS_PER_SEC = 16000

tokenizer = AutoTokenizer.from_pretrained(MODEL)

rng = random.Random(10)

PARQUET_DATASET_OVERRIDES = {
    "gimmaru/piqa": {
        "repo_id": "gimmaru/piqa",
        "files_by_split": {
            "validation": [
                "data/validation-00000-of-00001-26538eb75c618d24.parquet",
            ],
        },
    },
    # Keep a compatibility alias in case an older dataset registry still
    # points to the original script-backed repo id.
    "ybisk/piqa": {
        "repo_id": "gimmaru/piqa",
        "files_by_split": {
            "validation": [
                "data/validation-00000-of-00001-26538eb75c618d24.parquet",
            ],
        },
    },
}

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


def percentile(values, q):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")

    sorted_values = sorted(float(v) for v in values)
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]

    frac = pos - lo
    return (
        sorted_values[lo] * (1.0 - frac)
        + sorted_values[hi] * frac
    )


def load_dataset_compat(name, subset):
    if name == MIXED_HOTPOT_BOOLQ_NAME:
        hotpot_name, hotpot_subset, hotpot_formatter, hotpot_split = (
            ALL_DATASETS["hotpot_qa_fullwiki"]
        )
        boolq_name, boolq_subset, boolq_formatter, boolq_split = ALL_DATASETS["boolq"]

        hotpot_ds = load_dataset_compat(hotpot_name, hotpot_subset)[hotpot_split]
        boolq_ds = load_dataset_compat(boolq_name, boolq_subset)[boolq_split]

        hotpot_ds = hotpot_ds.select(range(0, len(hotpot_ds), MIXED_DATASET_STRIDE))
        boolq_ds = boolq_ds.select(range(0, len(boolq_ds), MIXED_DATASET_STRIDE))

        mixed_examples = [
            formatted
            for formatted in (
                hotpot_formatter(ex) for ex in hotpot_ds
            )
            if formatted is not None
        ]
        mixed_examples.extend(
            formatted
            for formatted in (
                boolq_formatter(ex) for ex in boolq_ds
            )
            if formatted is not None
        )

        mixed_rng = random.Random(MIXED_DATASET_SHUFFLE_SEED)
        mixed_rng.shuffle(mixed_examples)
        return DatasetDict({"validation": Dataset.from_list(mixed_examples)})

    override = PARQUET_DATASET_OVERRIDES.get(name)
    if override is not None:
        data_files = {
            split_name: [
                hf_hub_download(
                    repo_id=override["repo_id"],
                    filename=filename,
                    repo_type="dataset",
                )
                for filename in filenames
            ]
            for split_name, filenames in override["files_by_split"].items()
        }
        return load_dataset("parquet", data_files=data_files)

    if subset is None:
        return load_dataset(name)
    return load_dataset(name, subset)


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
        {"role": "system", "content": "Always answer in English."}, # pick_sysprompt_random(SYSTEM_PROMPTS)
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

    ds = load_dataset_compat(name, subset)

    # split = "test" if "test" in ds else "validation"
    base_ds = ds[split]
    # Setting 200 for now just for quick experiments. TODO: Remove it.
    # base_ds = base_ds.select(range(400))
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
    service_ttft_values = [r["service_ttft"] for r in measured]
    service_ttft_p50 = percentile(service_ttft_values, 0.50)
    service_ttft_p95 = percentile(service_ttft_values, 0.95)
    service_ttft_p99 = percentile(service_ttft_values, 0.99)
    throughput = (total_input + total_output) / total_time
    # correct = sum(1 for r in measured if r["pred"].startswith(r["gt"]))
    correct = sum(1 for r in measured if r["pred"] == r["gt"])

    stats = {
        "dataset": name,
        "subset": subset,
        "requests": len(measured),
        "avg_ttft": avg_ttft,
        "service_ttft": service_ttft,
        "service_ttft_p50": service_ttft_p50,
        "service_ttft_p95": service_ttft_p95,
        "service_ttft_p99": service_ttft_p99,
        "throughput": throughput,
        "accuracy": correct / len(measured),
    }

    print(
        f"TTFT={avg_ttft:.4f}s | "
        f"SVC_TTFT={service_ttft:.4f}s | "
        f"SVC_TTFT_P50={service_ttft_p50:.4f}s | "
        f"SVC_TTFT_P95={service_ttft_p95:.4f}s | "
        f"SVC_TTFT_P99={service_ttft_p99:.4f}s | "
        f"Throughput={throughput:.1f} tok/s | "
        f"Acc={stats['accuracy']:.3f}"
    )

    return stats

def set_vllm_config(expert, kv, load, step_interval, expert_dump_dir, traffic_dump_dir):
    if expert > 0 or kv > 0 or load > 0:
        resp = requests.post(
          f"{HOST}/load_balancer/weights",
          json={
              "kv_block_prefix_routing_weight": kv,
              "load_score_routing_weight": load,
              "eplb_step_interval": step_interval, # this is a copy in the frontend
              "expert_affinity_routing_weight": expert,
          },
          timeout=100,
        )
        resp.raise_for_status()
        print("POST /load_balancer/weights ->", resp.json())


    # eplb frequency of expert placement updates
    # number of engine steps / forward passes
    resp = requests.post(
      f"{HOST}/eplb/step_interval",
      json={"step_interval": step_interval},
      timeout=100,
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
      timeout=100,
    )
    resp.raise_for_status()
    print("POST /reset_prefix_cache -> 200 OK")


    resp = requests.post(
      f"{HOST}/eplb/placement_routing_dump",
      json={"dump_dir": expert_dump_dir},
      timeout=100,
    )
    resp.raise_for_status()

    resp = requests.post(
      f"{HOST}/eplb/moe_dispatch_traffic_dump",
      json={"dump_dir": traffic_dump_dir},
      timeout=100,
    )
    resp.raise_for_status()

    
    # Checking if the updates are applied
    if expert > 0 or kv > 0 or load > 0:
        resp = requests.get(
          f"{HOST}/load_balancer/weights",
          timeout=100,
        )
        resp.raise_for_status()
        print("GET /load_balancer/weights ->", resp.json())

    resp = requests.get(
      f"{HOST}/eplb/step_interval",
      timeout=100,
    )
    resp.raise_for_status()
    print("GET /eplb/step_interval ->", resp.json())

    resp = requests.get(f"{HOST}/eplb/placement_routing_dump", timeout=100)
    resp.raise_for_status()
    print("GET /eplb/placement_routing_dump ->", resp.json())


    resp = requests.get(f"{HOST}/eplb/moe_dispatch_traffic_dump", timeout=100)
    resp.raise_for_status()
    print("GET /eplb/moe_dispatch_traffic_dump ->", resp.json())

DATASETS = SEND_PROMPTS_DATASETS


def build_dump_dir(root_dump_dir, dataset_name, subset, split):
    model_label = MODEL.rsplit("/", 1)[-1]
    dataset_label = dataset_name.rsplit("/", 1)[-1]
    subset_label = (subset or "none").replace("/", "_")
    split_label = (split or "none").replace("/", "_")
    root = root_dump_dir.rstrip("/")
    return (
        f"{root}/"
        f"{model_label}-{dataset_label}-{subset_label}-{split_label}"
    )


def build_summary_path(root_dump_dir):
    model_label = MODEL.rsplit("/", 1)[-1]
    root = root_dump_dir.rstrip("/") or "."
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(root, f"{model_label}_final_summary.csv")


def format_summary_lines(results):
    lines = ["dataset,ttft,svcttft,svcttft50,svcttft95,svcttft99,throughput,accuracy"]
    for r in results:
        lines.append(
            f"{r['dataset']}/{r['subset']},"
            f"{r['avg_ttft']:.3f},"
            f"{r['service_ttft']:.3f},"
            f"{r['service_ttft_p50']:.3f},"
            f"{r['service_ttft_p95']:.3f},"
            f"{r['service_ttft_p99']:.3f},"
            f"{r['throughput']:.1f},"
            f"{r['accuracy']:.3f}"
        )
    return lines


async def main():
    expert_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
    kv_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.33
    load_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.34
    step_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    expert_dump_dir = str(sys.argv[5]) if len(sys.argv) > 5 else "traces/"
    traffic_dump_dir = str(sys.argv[6]) if len(sys.argv) > 6 else "traffic/"
    summary_dir = str(sys.argv[7]) if len(sys.argv) > 7 else "summary/"

    print(f"Setting LB weights: expert={expert_weight} kv={kv_weight} load={load_weight}")
    results = []

    for name, subset, formatter, split in DATASETS:
        expert_trace = build_dump_dir(expert_dump_dir, name, subset, split)
        os.makedirs(expert_trace, exist_ok=True)
        traffic_trace = build_dump_dir(traffic_dump_dir, name, subset, split)
        os.makedirs(traffic_trace, exist_ok=True)
        set_vllm_config(
            expert_weight,
            kv_weight,
            load_weight,
            step_interval,
            expert_trace,
            traffic_trace
        )
        stats = await run_dataset(name, subset, formatter, split)
        results.append(stats)
        print(
            f"{stats['dataset']} / {stats['subset']} | "
            f"TTFT={stats['avg_ttft']:.3f}s | "
            f"SVC_TTFT={stats['service_ttft']:.3f}s | "
            f"SVC_TTFT_P50={stats['service_ttft_p50']:.3f}s | "
            f"SVC_TTFT_P95={stats['service_ttft_p95']:.3f}s | "
            f"SVC_TTFT_P99={stats['service_ttft_p99']:.3f}s | "
            f"Throughput={stats['throughput']:.1f} tok/s | "
            f"Acc={stats['accuracy']:.3f}"
        )

    summary_lines = format_summary_lines(results)
    print()
    for line in summary_lines:
        print(line)

    os.makedirs(summary_dir, exist_ok=True)
    summary_path = build_summary_path(summary_dir)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Wrote final summary to {summary_path}")

asyncio.run(main())
