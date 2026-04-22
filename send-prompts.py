import asyncio
import aiohttp
import time
import random
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

HOST = "http://10.201.0.2:8000"
MODEL = "deepseek-ai/deepseek-moe-16b-chat"

SYSTEM_PROMPT = "You are a precise assistant. Answer clearly and concisely. Always answer in English."

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def poisson_interarrival(rate):
    return random.expovariate(rate)


def count_tokens(text):
    return len(tokenizer.encode(text))


async def ask(session, sem, idx, ex):
    async with sem:
        q = ex["question"]
        choices = ex["choices"]
        answer = ex["answer"]

        options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        user_prompt = f"{q}\n{options}\nAnswer with ONLY one letter: A, B, C, or D."

        full_prompt = SYSTEM_PROMPT + "\n" + user_prompt

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
                pred = data["choices"][0]["message"]["content"].strip()
            except Exception:
                pred = "ERROR"

        completion_ns = now_ns()

        # convert to seconds
        ttft = (first_byte_ns - arrival_ns) / 1e9
        total_latency = (completion_ns - arrival_ns) / 1e9

        input_tokens = count_tokens(full_prompt)
        output_tokens = 1

        print(
            f"[{idx}] arr_ns={arrival_ns} first_ns={first_byte_ns} done_ns={completion_ns} "
            f"TTFT={ttft:.6f}s LAT={total_latency:.6f}s Pred={pred}",
            flush=True
        )

        return {
            "idx": idx,
            "pred": pred,
            "gt": chr(65 + answer),
            "latency": total_latency,
            "ttft": ttft,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
def now_ns():
    return time.perf_counter_ns()

async def poisson_driver(ds, rate, warmup_count):
    connector = aiohttp.TCPConnector(limit=200)
    sem = asyncio.Semaphore(50)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        start_ns = time.perf_counter_ns()

        async def arrival_loop():
            for i, ex in enumerate(ds):
                await asyncio.sleep(poisson_interarrival(rate))
                task = asyncio.create_task(ask(session, sem, i, ex))
                tasks.append(task)

        arrival_task = asyncio.create_task(arrival_loop())

        await arrival_task
        results = await asyncio.gather(*tasks)

        end_ns = time.perf_counter_ns()

    total_time = (end_ns - start_ns) / 1e9

    # split warmup and measured
    measured = results[warmup_count:]

    return results, measured, total_time

async def run_dataset(name, subset):
    print(f"\nRunning dataset: {name} / {subset}")

    base_ds = load_dataset(name, subset, split="test")

    warmup_epochs = 1
    measured_epochs = 1

    num_repeats = warmup_epochs + measured_epochs
    ds = concatenate_datasets([base_ds] * num_repeats)

    warmup_count = len(base_ds) * warmup_epochs

    rate = 1

    results, measured, total_time = await poisson_driver(ds, rate, warmup_count)

    total_input = sum(r["input_tokens"] for r in measured)
    total_output = sum(r["output_tokens"] for r in measured)

    avg_ttft = sum(r["latency"] for r in measured) / len(measured)
    throughput = (total_input + total_output) / total_time

    correct = sum(1 for r in measured if r["pred"].startswith(r["gt"]))

    print(f"\n----- Average performance for {name} / {subset} -----")
    print(f"Requests (measured): {len(measured)}")
    print(f"Warmup requests skipped: {warmup_count}")
    print(f"Avg TTFT: {avg_ttft:.4f} sec")
    print(f"Throughput (tokens/sec): {throughput:.2f}")
    print(f"Accuracy: {correct}/{len(measured)}")
    print("----- End -----\n")

async def main():
    await run_dataset("cais/mmlu", "abstract_algebra")


asyncio.run(main())