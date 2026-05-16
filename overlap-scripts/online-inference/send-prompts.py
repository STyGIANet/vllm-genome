import asyncio
import aiohttp
import time
import random
import requests
import sys
import re
import os
import math

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from prompt_datasets import (
    ALL_DATASETS,
    BBH_ALL_NAME,
    MIXED_DATASET_SHUFFLE_SEED,
    MIXED_DATASET_STRIDE,
    MIXED_HOTPOT_BOOLQ_NAME,
    SEND_PROMPTS_DATASETS,
    SYSTEM_PROMPTS,
)
# Change which prompt datasets to test in prompt_datasets.py at the end of the file

HOST = os.getenv("VLLM_HOST", "http://127.0.0.1:8000")
#MODEL = os.getenv("VLLM_MODEL", "deepseek-ai/deepseek-moe-16b-chat")
#MODEL = os.getenv("VLLM_MODEL", "microsoft/Phi-3.5-MoE-instruct")
MODEL = os.getenv("VLLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
#MODEL = os.getenv("VLLM_MODEL", "mistralai/Mixtral-8x22B-v0.1")

POISSON_RATE_MODE = os.getenv("POISSON_RATE_MODE", "tok_per_sec")
POISSON_REQUESTS_PER_SEC = float(os.getenv("POISSON_REQUESTS_PER_SEC", "1.0"))
_poisson_input_tokens_per_sec = os.getenv("POISSON_INPUT_TOKENS_PER_SEC", "90000")
POISSON_INPUT_TOKENS_PER_SEC = (
    None
    if _poisson_input_tokens_per_sec.lower() == "none"
    else float(_poisson_input_tokens_per_sec)
)
CLIENT_CONN_LIMIT = int(os.getenv("CLIENT_CONN_LIMIT", "1024"))
CLIENT_CONCURRENCY = int(os.getenv("CLIENT_CONCURRENCY", "128"))
REQUEST_MAX_TOKENS = int(os.getenv("REQUEST_MAX_TOKENS", "1"))
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("MODEL_MAX_CONTEXT_TOKENS", "32768"))
MAX_INPUT_TOKENS = int(
    os.getenv(
        "MAX_INPUT_TOKENS",
        str(max(1, MODEL_MAX_CONTEXT_TOKENS - REQUEST_MAX_TOKENS)),
    )
)
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "0"))
MEASURED_EPOCHS = int(os.getenv("MEASURED_EPOCHS", "1"))
PRE_RUN_WARMUP_PROMPTS = int(os.getenv("PRE_RUN_WARMUP_PROMPTS", "2000"))
PRE_RUN_WARMUP_INPUT_TOKENS_PER_SEC = float(
    os.getenv("PRE_RUN_WARMUP_INPUT_TOKENS_PER_SEC", "20000")
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

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

    if name == BBH_ALL_NAME:
        bbh_configs = sorted(get_dataset_config_names("lukaemon/bbh"))
        bbh_examples: list[dict[str, str]] = []

        for config_name in bbh_configs:
            subset_ds = load_dataset("lukaemon/bbh", config_name)["test"]
            bbh_examples.extend(
                {
                    "question": ex["input"],
                    "task": config_name,
                }
                for ex in subset_ds
            )

        return DatasetDict({"test": Dataset.from_list(bbh_examples)})

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
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = tokenized.get("input_ids") if hasattr(tokenized, "get") else tokenized
        if hasattr(input_ids, "shape"):
            shape = tuple(int(dim) for dim in input_ids.shape)
            if not shape:
                return int(input_ids.item())
            return int(shape[-1])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            return len(input_ids)
        return len(input_ids)
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
    choices = ex.get("choices")
    if not choices:
        return f"{ex['question']}\nAnswer concisely."

    letters = [chr(65 + i) for i in range(len(choices))]
    options = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])

    if len(choices) <= 4:
        return f"{ex['question']}\n{options}\nAnswer with ONLY one letter: {', '.join(letters)}."
    return f"{ex['question']}\nAnswer concisely."


def truncate_messages_to_input_budget(messages, max_input_tokens):
    input_tokens = count_message_tokens(messages)
    if input_tokens <= max_input_tokens:
        return messages, input_tokens, False

    truncated_messages = [dict(msg) for msg in messages]
    if not truncated_messages:
        return truncated_messages, input_tokens, False

    target_idx = len(truncated_messages) - 1
    content = truncated_messages[target_idx].get("content", "")
    content_token_ids = tokenizer.encode(content, add_special_tokens=False)
    base_messages = [dict(msg) for msg in truncated_messages]
    base_messages[target_idx]["content"] = ""
    base_tokens = count_message_tokens(base_messages)
    allowed_content_tokens = max(0, max_input_tokens - base_tokens)

    truncated_token_ids = content_token_ids[:allowed_content_tokens]
    truncated_messages[target_idx]["content"] = tokenizer.decode(
        truncated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    truncated_input_tokens = count_message_tokens(truncated_messages)

    while truncated_input_tokens > max_input_tokens and truncated_token_ids:
        overflow = truncated_input_tokens - max_input_tokens
        truncated_token_ids = truncated_token_ids[:-max(overflow, 1)]
        truncated_messages[target_idx]["content"] = tokenizer.decode(
            truncated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        truncated_input_tokens = count_message_tokens(truncated_messages)

    return truncated_messages, truncated_input_tokens, True


def prepare_example(ex):
    user_prompt = build_user_prompt(ex)
    messages = [
        {"role": "system", "content": "Always answer in English."}, # pick_sysprompt_random(SYSTEM_PROMPTS)
        {"role": "user", "content": user_prompt},
    ]
    messages, input_tokens, was_truncated = truncate_messages_to_input_budget(
        messages,
        MAX_INPUT_TOKENS,
    )
    return {
        **ex,
        "user_prompt": user_prompt,
        "messages": messages,
        "input_tokens": input_tokens,
        "truncated_to_fit_context": was_truncated,
    }


def load_prepared_examples(name, subset, formatter, split):
    ds = load_dataset_compat(name, subset)
    base_ds = ds[split]

    formatted_examples = []
    for ex in base_ds:
        formatted = formatter(ex)
        if formatted is None:
            continue
        formatted_examples.append(formatted)

    return [prepare_example(ex) for ex in formatted_examples], len(base_ds)


async def ask(session, sem, idx, ex, scheduled_arrival_ns):
    async with sem:
        choices = ex.get("choices")
        payload = {
            "model": MODEL,
            "messages": ex["messages"],
            "max_tokens": REQUEST_MAX_TOKENS,
            "temperature": 0.0,
        }

        dispatch_ns = now_ns()

        async with session.post(f"{HOST}/v1/chat/completions", json=payload) as resp:
            first_byte_ns = now_ns()

            try:
                data = await resp.json()
                raw_pred = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                if choices:
                    pred = extract_choice(raw_pred, len(choices))
                else:
                    pred = raw_pred.strip()
                output_tokens = int(
                    usage.get(
                        "completion_tokens",
                        len(tokenizer.encode(raw_pred)) if raw_pred else 0,
                    )
                )
            except Exception:
                pred = "ERROR"
                output_tokens = 0

        completion_ns = now_ns()
        # Queueing here just the async io's queueing, not on the inference side
        queue_delay = (dispatch_ns - scheduled_arrival_ns) / 1e9
        ttft = (first_byte_ns - scheduled_arrival_ns) / 1e9
        total_latency = (completion_ns - scheduled_arrival_ns) / 1e9
        service_ttft = (first_byte_ns - dispatch_ns) / 1e9
        service_latency = (completion_ns - dispatch_ns) / 1e9

        input_tokens = ex["input_tokens"]
        print(
            f"[{idx}] QD={queue_delay:.4f}s TTFT={ttft:.4f}s "
            f"LAT={total_latency:.4f}s SVC_TTFT={service_ttft:.4f}s "
            f"SVC_LAT={service_latency:.4f}s Pred={pred}, Tokens={input_tokens + output_tokens}",
            flush=True
        )

        return {
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
    connector = aiohttp.TCPConnector(limit=CLIENT_CONN_LIMIT)
    sem = asyncio.Semaphore(CLIENT_CONCURRENCY)

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

    examples, split_size = load_prepared_examples(name, subset, formatter, split)
    print("Dataset split size:", split_size)

    print("Prompts:", len(examples))

    warmup_epochs = WARMUP_EPOCHS
    measured_epochs = MEASURED_EPOCHS

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
    }

    print(
        f"TTFT={avg_ttft:.4f}s | "
        f"SVC_TTFT={service_ttft:.4f}s | "
        f"SVC_TTFT_P50={service_ttft_p50:.4f}s | "
        f"SVC_TTFT_P95={service_ttft_p95:.4f}s | "
        f"SVC_TTFT_P99={service_ttft_p99:.4f}s | "
        f"Throughput={throughput:.1f} tok/s"
    )

    return stats


async def run_pre_run_warmup():
    if PRE_RUN_WARMUP_PROMPTS <= 0:
        return

    name, subset, formatter, split = ALL_DATASETS["mmlu_all"]
    examples, split_size = load_prepared_examples(name, subset, formatter, split)
    if not examples:
        print("Skipping pre-run warmup: no Humaneval prompts available.")
        return

    warmup_examples = examples[:min(PRE_RUN_WARMUP_PROMPTS, len(examples))]
    mean_input_tokens = (
        sum(ex["input_tokens"] for ex in warmup_examples) / len(warmup_examples)
    )

    print(
        "\n### Pre-run warmup: humaneval ###\n"
        f"Dataset split size: {split_size}\n"
        f"Warmup prompts: {len(warmup_examples)}\n"
        "Poisson arrivals: mode=tok_per_sec "
        f"rate={PRE_RUN_WARMUP_INPUT_TOKENS_PER_SEC:.2f} tok/s "
        f"(mean_input_tokens={mean_input_tokens:.2f})"
    )

    await poisson_driver(
        warmup_examples,
        PRE_RUN_WARMUP_INPUT_TOKENS_PER_SEC,
        "tok_per_sec",
        0,
    )
    print("Completed pre-run warmup.")

def set_vllm_config():
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
    lines = ["dataset,ttft,svcttft,svcttft50,svcttft95,svcttft99,throughput"]
    for r in results:
        lines.append(
            f"{r['dataset']}/{r['subset']},"
            f"{r['avg_ttft']:.3f},"
            f"{r['service_ttft']:.3f},"
            f"{r['service_ttft_p50']:.3f},"
            f"{r['service_ttft_p95']:.3f},"
            f"{r['service_ttft_p99']:.3f},"
            f"{r['throughput']:.1f}"
        )
    return lines


async def main():
    summary_dir = str(sys.argv[1]) if len(sys.argv) > 1 else "summary/"
    results = []
    await run_pre_run_warmup()

    for name, subset, formatter, split in DATASETS:
        set_vllm_config()
        stats = await run_dataset(name, subset, formatter, split)
        results.append(stats)
        print(
            f"{stats['dataset']} / {stats['subset']} | "
            f"TTFT={stats['avg_ttft']:.3f}s | "
            f"SVC_TTFT={stats['service_ttft']:.3f}s | "
            f"SVC_TTFT_P50={stats['service_ttft_p50']:.3f}s | "
            f"SVC_TTFT_P95={stats['service_ttft_p95']:.3f}s | "
            f"SVC_TTFT_P99={stats['service_ttft_p99']:.3f}s | "
            f"Throughput={stats['throughput']:.1f} tok/s"
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
