import asyncio
import aiohttp
import time
import random
import requests
import sys
import re

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

HOST = "http://0.0.0.0:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

SYSTEM_PROMPT = "You are a precise assistant. Answer clearly and concisely. Always answer in English."

tokenizer = AutoTokenizer.from_pretrained(MODEL)


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

def format_hotpotqa(ex):
    try:
        sentences = ex.get("context", {}).get("sentences", [])
        flat = []

        for para in sentences:
            flat.extend(para)

        context = " ".join(flat[:50])
        question = ex.get("question", "")

        return {
            "question": f"{context}\n\nQuestion: {question}",
            "choices": ["Yes", "No"],
            "answer": 0,
        }
    except Exception:
        return None

def format_pubmedqa(ex):
    try:
        contexts = ex.get("context", {}).get("contexts", [])
        context = " ".join(contexts[:5])

        question = ex.get("question", "")

        choices = ["yes", "no", "maybe"]
        answer_map = {"yes": 0, "no": 1, "maybe": 2}

        answer = answer_map.get(ex.get("final_decision", ""), 0)

        return {
            "question": f"{context}\n\nQuestion: {question}",
            "choices": choices,
            "answer": answer,
        }
    except Exception:
        return None

def format_arc(ex):
    return {
        "question": ex["question"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_openbookqa(ex):
    return {
        "question": ex["question_stem"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_csqa(ex):
    return {
        "question": ex["question"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_boolq(ex):
    return {
        "question": ex["question"],
        "choices": ["True", "False"],
        "answer": 0 if ex["answer"] else 1,
    }


def format_piqa(ex):
    return {
        "question": ex["goal"],
        "choices": [ex["sol1"], ex["sol2"]],
        "answer": ex["label"],
    }


def poisson_interarrival(rate):
    return random.expovariate(rate)


def now_ns():
    return time.perf_counter_ns()


def count_tokens(text):
    return len(tokenizer.encode(text))


async def ask(session, sem, idx, ex):
    async with sem:
        q = ex["question"]
        choices = ex["choices"]
        answer = ex["answer"]

        letters = [chr(65 + i) for i in range(len(choices))]
        options = "\n".join([f"{letters[i]}. {c}" for i, c in enumerate(choices)])

        # user_prompt = f"{q}\n{options}\nAnswer with ONLY one Alphabet: {', '.join(letters)}."
        if len(choices) <= 4:
            user_prompt = f"{q}\n{options}\nAnswer with ONLY one letter: {', '.join(letters)}."
        else:
            user_prompt = f"{q}\nAnswer concisely."

        # print(user_prompt, flush=True)
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1,
            "temperature": 0.0
        }

        arrival_ns = now_ns()

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

        ttft = (first_byte_ns - arrival_ns) / 1e9
        total_latency = (completion_ns - arrival_ns) / 1e9

        input_tokens = count_tokens(user_prompt)
        output_tokens = 1

        print(
            f"[{idx}] TTFT={ttft:.4f}s LAT={total_latency:.4f}s Pred={pred}, Tokens={input_tokens + output_tokens}", 
            flush=True
        )

        return {
            "pred": pred,
            "gt": chr(65 + answer),
            "latency": total_latency,
            "ttft": ttft,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }


async def poisson_driver(ds, rate, warmup_count):
    connector = aiohttp.TCPConnector(limit=100)
    sem = asyncio.Semaphore(40)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        start_ns = time.perf_counter_ns()

        async def arrival_loop():
            for i, ex in enumerate(ds):
                await asyncio.sleep(poisson_interarrival(rate))
                tasks.append(asyncio.create_task(ask(session, sem, i, ex)))

        await arrival_loop()
        results = await asyncio.gather(*tasks)

        end_ns = time.perf_counter_ns()

    total_time = (end_ns - start_ns) / 1e9
    measured = results[warmup_count:]

    return measured, total_time


async def run_dataset(name, subset, formatter, split):
    print(f"\n### Running: {name} / {subset} ###")

    if subset is None:
        ds = load_dataset(name)
    else:
        ds = load_dataset(name, subset)

    # split = "test" if "test" in ds else "validation"
    base_ds = ds[split]
    base_ds = base_ds.select(range(200))
    print("Dataset split size:", len(base_ds))

    base_ds = base_ds.map(formatter)

    print("Prompts:", len(base_ds))

    warmup_epochs = 0
    measured_epochs = 1

    num_repeats = warmup_epochs + measured_epochs
    ds_full = concatenate_datasets([base_ds] * num_repeats)
    warmup_count = len(base_ds) * warmup_epochs

    rate = 1

    measured, total_time = await poisson_driver(ds_full, rate, warmup_count)

    total_input = sum(r["input_tokens"] for r in measured)
    total_output = sum(r["output_tokens"] for r in measured)

    avg_ttft = sum(r["ttft"] for r in measured) / len(measured)
    throughput = (total_input + total_output) / total_time
    # correct = sum(1 for r in measured if r["pred"].startswith(r["gt"]))
    correct = sum(1 for r in measured if r["pred"] == r["gt"])

    stats = {
        "dataset": name,
        "subset": subset,
        "requests": len(measured),
        "avg_ttft": avg_ttft,
        "throughput": throughput,
        "accuracy": correct / len(measured),
    }

    print(f"TTFT={avg_ttft:.4f}s | Throughput={throughput:.1f} tok/s | Acc={stats['accuracy']:.3f}")

    return stats



def set_lb_weights(expert_weight, kv_weight, load_weight, step_interval):
    resp = requests.post(
        f"{HOST}/load_balancer/weights",
        json={
            "expert_affinity_routing_weight": expert_weight,
            "kv_block_prefix_routing_weight": kv_weight,
            "load_score_routing_weight": load_weight,
            "step_interval": step_interval,
        },
        timeout=10,
    )
    resp.raise_for_status()



DATASETS = [
    ("hotpot_qa", "fullwiki", format_hotpotqa, "validation"),
    ("pubmed_qa", "pqa_labeled", format_pubmedqa, "train"),
    ("ai2_arc", "ARC-Challenge", format_arc, "validation"),
    # ("ai2_arc", "ARC-Easy", format_arc, "validation"),
    # ("openbookqa", "main", format_openbookqa, "validation"),
    # ("commonsense_qa", None, format_csqa, "validation"),
    # ("boolq", None, format_boolq, "validation"),
    # ("piqa", None, format_piqa, "validation"),
]


async def main():
    results = []

    for name, subset, formatter, split in DATASETS:
        stats = await run_dataset(name, subset, formatter, split)
        results.append(stats)
        print(
            f"{stats['dataset']} / {stats['subset']} | "
            f"TTFT={stats['avg_ttft']:.3f}s | "
            f"Throughput={stats['throughput']:.1f} tok/s | "
            f"Acc={stats['accuracy']:.3f}"
        )

    print("\n### FINAL SUMMARY ###")
    for r in results:
        print(
            f"{r['dataset']} / {r['subset']} | "
            f"TTFT={r['avg_ttft']:.3f}s | "
            f"Throughput={r['throughput']:.1f} tok/s | "
            f"Acc={r['accuracy']:.3f}"
        )

expert_weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
kv_weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.33
load_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.34
step_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 30

print(f"Setting LB weights: expert={expert_weight} kv={kv_weight} load={load_weight}")
set_lb_weights(expert_weight, kv_weight, load_weight, step_interval)

asyncio.run(main())
