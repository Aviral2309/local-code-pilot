"""
tests/test_debounce.py
----------------------
Tests for the async debounce mechanism.

Key behaviors to verify:
1. Single call fires after delay
2. Rapid calls result in only ONE execution (the last one)
3. Manual cancellation prevents execution
4. The correct arguments are passed through
"""

import asyncio
import pytest
from src.debounce import AsyncDebouncer


@pytest.mark.asyncio
class TestAsyncDebouncer:

    async def test_single_call_fires(self):
        """A single call fires once after the delay."""
        call_count = 0

        async def handler():
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebouncer(delay_ms=50)
        await debouncer.call(handler)
        await asyncio.sleep(0.1)  # Wait longer than delay

        assert call_count == 1

    async def test_rapid_calls_fire_once(self):
        """
        Multiple rapid calls result in exactly ONE execution.
        This is the core contract of debouncing.
        """
        call_count = 0

        async def handler():
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebouncer(delay_ms=100)

        # Fire 10 times rapidly
        for _ in range(10):
            await debouncer.call(handler)
            await asyncio.sleep(0.01)  # 10ms between calls < 100ms delay

        # Wait for debounce to settle
        await asyncio.sleep(0.2)

        assert call_count == 1, f"Expected 1 call, got {call_count}"

    async def test_call_after_delay_fires_again(self):
        """
        If enough time passes between calls, each fires independently.
        """
        call_count = 0

        async def handler():
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebouncer(delay_ms=50)

        await debouncer.call(handler)
        await asyncio.sleep(0.15)  # Wait for first to fire

        await debouncer.call(handler)
        await asyncio.sleep(0.15)  # Wait for second to fire

        assert call_count == 2

    async def test_manual_cancel_prevents_execution(self):
        """Calling cancel() prevents the pending execution."""
        call_count = 0

        async def handler():
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebouncer(delay_ms=200)
        await debouncer.call(handler)
        debouncer.cancel()

        await asyncio.sleep(0.3)

        assert call_count == 0

    async def test_arguments_passed_correctly(self):
        """Arguments are forwarded to the handler."""
        received_args = []

        async def handler(x, y, keyword=None):
            received_args.extend([x, y, keyword])

        debouncer = AsyncDebouncer(delay_ms=50)
        await debouncer.call(handler, 1, 2, keyword="test")
        await asyncio.sleep(0.1)

        assert received_args == [1, 2, "test"]

    async def test_cancel_on_no_pending_is_safe(self):
        """Calling cancel() with nothing pending doesn't raise."""
        debouncer = AsyncDebouncer(delay_ms=50)
        debouncer.cancel()  # Should not raise

    async def test_handler_exception_does_not_crash_debouncer(self):
        """If the handler raises, the debouncer should still be usable."""
        call_count = 0

        async def bad_handler():
            raise ValueError("intentional error")

        async def good_handler():
            nonlocal call_count
            call_count += 1

        debouncer = AsyncDebouncer(delay_ms=50)

        # This will raise inside the task — asyncio swallows it
        await debouncer.call(bad_handler)
        await asyncio.sleep(0.1)

        # Debouncer should still work
        await debouncer.call(good_handler)
        await asyncio.sleep(0.1)

        assert call_count == 1
