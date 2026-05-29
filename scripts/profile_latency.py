#!/usr/bin/env python3
"""
scripts/profile_latency.py
--------------------------
Profiles end-to-end latency of the AuraLSP completion pipeline.

Measures each stage independently:
    - FIM prompt construction time
    - Ollama API call latency
    - Time-to-first-token (TTFT)
    - Total generation time

Output: prints a table + saves to eval/results/latency_profile.csv

Run: python scripts/profile_latency.py --runs 10
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

sys.path.insert(0, ".")

from src.config import load_config
from src.fim_builder import build_fim_prompt
from src.ollama_client import OllamaClient

# A realistic test prompt — representative of real usage
TEST_PREFIX = """import numpy as np
from typing import List, Optional

class PortfolioAnalyzer:
    def __init__(self, returns: List[float]):
        self.returns = np.array(returns)

    def sharpe_ratio(self, risk_free: float = 0.02) -> float:
        excess = self.returns - risk_free / 252
        if np.std(excess) == 0:
            return 0.0
        """

TEST_SUFFIX = """
    def max_drawdown(self) -> float:
        cumulative = np.cumprod(1 + self.returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        return float(np.min(drawdowns))
"""


async def measure_single_completion(client: OllamaClient) -> dict:
    """Run one completion and return timing breakdown."""

    # --- Stage 1: FIM prompt construction ---
    t0 = time.perf_counter()
    payload = build_fim_prompt(
        prefix=TEST_PREFIX,
        suffix=TEST_SUFFIX,
        language_id="python",
    )
    prompt_build_ms = (time.perf_counter() - t0) * 1000

    # --- Stage 2: Ollama streaming ---
    first_token_ms = None
    total_tokens = 0
    completion_chars = 0

    t_start = time.perf_counter()
    async for token in client.stream_completion(payload):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t_start) * 1000
        total_tokens += 1
        completion_chars += len(token)

    total_ms = (time.perf_counter() - t_start) * 1000

    return {
        "prompt_build_ms": round(prompt_build_ms, 2),
        "ttft_ms": round(first_token_ms or 0, 2),
        "total_ms": round(total_ms, 2),
        "tokens_generated": total_tokens,
        "completion_chars": completion_chars,
        "token_estimate": payload.token_estimate,
    }


def print_summary(results: list[dict]) -> None:
    if not results:
        return

    def col(key):
        return [r[key] for r in results]

    print("\n" + "=" * 65)
    print(f"  AuraLSP Latency Profile — {len(results)} runs")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Mean':>8} {'Median':>8} {'Stdev':>8}")
    print("-" * 65)

    metrics = [
        ("Prompt build (ms)", "prompt_build_ms"),
        ("Time-to-first-token (ms)", "ttft_ms"),
        ("Total roundtrip (ms)", "total_ms"),
        ("Completion chars", "completion_chars"),
    ]

    for label, key in metrics:
        vals = col(key)
        print(
            f"  {label:<30} "
            f"{mean(vals):>8.1f} "
            f"{median(vals):>8.1f} "
            f"{stdev(vals) if len(vals) > 1 else 0:>8.1f}"
        )

    print("=" * 65)

    # Performance check
    avg_total = mean(col("total_ms"))
    status = "✓ Under 300ms target" if avg_total < 300 else "✗ Over 300ms target"
    print(f"\n  {status} (avg: {avg_total:.0f}ms)\n")


async def main(runs: int):
    config = load_config()
    client = OllamaClient(config.ollama)

    print(f"Profiling {runs} completions with model: {config.ollama.model}")
    print("This requires Ollama to be running.\n")

    # Warmup: first call is always slow (model loading into RAM)
    print("Warmup run (not counted)...")
    try:
        await measure_single_completion(client)
        print("Warmup done.\n")
    except Exception as e:
        print(f"Error during warmup: {e}")
        print("Is Ollama running? Run: ollama serve")
        await client.close()
        sys.exit(1)

    # Timed runs
    results = []
    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        try:
            result = await measure_single_completion(client)
            results.append(result)
            print(f"total={result['total_ms']:.0f}ms | ttft={result['ttft_ms']:.0f}ms")
        except Exception as e:
            print(f"Error: {e}")

    await client.close()

    if results:
        print_summary(results)

        # Save to CSV
        output_dir = Path("eval/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "latency_profile.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    args = parser.parse_args()
    asyncio.run(main(runs=args.runs))
