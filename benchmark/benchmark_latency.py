"""
Standalone latency benchmark for Phase II / Phase VI.

Sends a fixed set of representative prompts straight to the local Ollama
model (bypassing the FastAPI layer, to isolate pure inference latency) and
reports time-to-first-token and tokens/sec for each, plus averages.

Usage:
    python benchmark/benchmark_latency.py
    python benchmark/benchmark_latency.py --model qwen2.5:1.5b-instruct-q4_K_M --runs 3
"""
import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import BASE_SYSTEM_PROMPT
from app.llm_engine import OllamaEngine

PROMPTS = [
    "Where is my order? It's order number BC-77120.",
    "Can I return a pair of headphones I bought 10 days ago?",
    "Does the KeyCraft K87 keyboard work on macOS?",
    "What's your shipping policy for express delivery?",
    "I want to cancel an order I placed 5 minutes ago, is that possible?",
]


async def run_benchmark(model: str, runs: int):
    engine = OllamaEngine(model=model)
    reachable = await engine.health_check()
    if not reachable:
        print(f"ERROR: could not reach Ollama. Is `ollama serve` running and "
              f"`ollama pull {model}` done?")
        return

    print(f"Benchmarking model: {model}\n{'-'*60}")

    all_ttft, all_tps, all_total = [], [], []

    for run_idx in range(1, runs + 1):
        print(f"\nRun {run_idx}/{runs}")
        for prompt in PROMPTS:
            messages = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            final_metrics = None
            async for _, metrics in engine.stream_chat(messages):
                final_metrics = metrics

            if final_metrics:
                all_ttft.append(final_metrics.time_to_first_token or 0)
                all_total.append(final_metrics.total_time or 0)
                if final_metrics.tokens_per_second:
                    all_tps.append(final_metrics.tokens_per_second)
                print(
                    f"  '{prompt[:40]}...' -> "
                    f"TTFT={final_metrics.time_to_first_token}s  "
                    f"total={final_metrics.total_time}s  "
                    f"tok/s={final_metrics.tokens_per_second}"
                )

    print(f"\n{'='*60}\nSUMMARY over {len(all_ttft)} generations")
    print(f"  Avg time-to-first-token : {statistics.mean(all_ttft):.3f}s")
    print(f"  Avg total response time : {statistics.mean(all_total):.3f}s")
    if all_tps:
        print(f"  Avg tokens/sec          : {statistics.mean(all_tps):.2f}")
    print(f"  Min/Max TTFT            : {min(all_ttft):.3f}s / {max(all_ttft):.3f}s")
    print("\nCopy these numbers into README.md's Latency Benchmarks section.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b-instruct-q4_K_M")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.model, args.runs))
