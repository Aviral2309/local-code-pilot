#!/usr/bin/env python3
"""
scripts/test_connection.py
--------------------------
Manual integration test: verify Ollama is running and the FIM pipeline works.

Run this BEFORE starting the LSP server to confirm your environment is ready.

Usage:
    python scripts/test_connection.py
    python scripts/test_connection.py --model deepseek-coder:1.3b

This script is NOT a unit test. It makes real HTTP calls to Ollama.
"""

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")  # Run from project root

from src.config import load_config
from src.fim_builder import build_fim_prompt
from src.ollama_client import OllamaClient, OllamaConnectionError


async def test_health(client: OllamaClient) -> bool:
    print("1. Testing Ollama connectivity...")
    try:
        await client.health_check()
        print("   ✓ Ollama is running and model is available.\n")
        return True
    except OllamaConnectionError as e:
        print(f"   ✗ Failed: {e}\n")
        return False


async def test_fim_completion(client: OllamaClient) -> bool:
    print("2. Testing FIM completion pipeline...")

    # A realistic Python completion scenario
    prefix = """import numpy as np
from typing import List

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    \"\"\"Calculate the Sharpe Ratio of a portfolio.\"\"\"
    excess = np.array(returns) - risk_free_rate / 252
    """

    suffix = """
    return float(sharpe)

def main():
    returns = [0.001, -0.002, 0.003, 0.001, -0.001]
    print(calculate_sharpe_ratio(returns))
"""

    payload = build_fim_prompt(
        prefix=prefix,
        suffix=suffix,
        language_id="python",
        injected_context="",
    )

    print(f"   Prompt token estimate: ~{payload.token_estimate}")
    print(f"   Streaming response: ", end="", flush=True)

    t_start = time.perf_counter()
    full_response = ""
    first_token = True

    try:
        async for token in client.stream_completion(payload):
            if first_token:
                ttft = (time.perf_counter() - t_start) * 1000
                print(f"\n   First token in {ttft:.0f}ms")
                print(f"   Completion: ", end="")
                first_token = False
            print(token, end="", flush=True)
            full_response += token

        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"\n   ✓ Complete. Total: {total_ms:.0f}ms | Length: {len(full_response)} chars\n")
        return True

    except Exception as e:
        print(f"\n   ✗ Error: {e}\n")
        return False


async def test_fim_with_context(client: OllamaClient) -> bool:
    print("3. Testing FIM with injected workspace context...")

    context = """
def rolling_std(data: list, window: int) -> list:
    \"\"\"Compute rolling standard deviation.\"\"\"
    import numpy as np
    arr = np.array(data)
    return [float(np.std(arr[max(0, i-window):i+1])) for i in range(len(arr))]
"""

    prefix = """def calculate_volatility(returns: list, window: int = 20) -> float:
    \"\"\"Annualized volatility using rolling std.\"\"\"
    daily_vol = """

    suffix = """
    return daily_vol * (252 ** 0.5)
"""

    payload = build_fim_prompt(
        prefix=prefix,
        suffix=suffix,
        language_id="python",
        injected_context=context,
    )

    print(f"   Context injected: {len(context)} chars")
    print(f"   Token estimate: ~{payload.token_estimate}")

    t_start = time.perf_counter()
    full = ""
    try:
        async for token in client.stream_completion(payload):
            full += token
        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"   Response: {full.strip()}")
        print(f"   ✓ Complete. Total: {total_ms:.0f}ms\n")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return False


async def main(model: str = None):
    print("=" * 60)
    print("  AuraLSP — Connection & Pipeline Test")
    print("=" * 60 + "\n")

    config = load_config()
    if model:
        config.ollama.model = model
        print(f"Using model: {model}\n")

    client = OllamaClient(config.ollama)

    results = []
    results.append(await test_health(client))
    if results[0]:  # Only test FIM if Ollama is reachable
        results.append(await test_fim_completion(client))
        results.append(await test_fim_with_context(client))

    await client.close()

    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"  Results: {passed}/{total} tests passed")

    if passed == total:
        print("  ✓ All tests passed. AuraLSP is ready.")
        print("  Next: open a Python file in VS Code/Neovim and type some code.")
    else:
        print("  ✗ Some tests failed. Check that:")
        print("    1. Ollama is running: `ollama serve`")
        print(f"   2. Model is pulled: `ollama pull {config.ollama.model}`")

    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AuraLSP connection and FIM pipeline")
    parser.add_argument("--model", help="Override model name", default=None)
    args = parser.parse_args()

    exit_code = asyncio.run(main(model=args.model))
    sys.exit(exit_code)
