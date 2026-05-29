"""
debounce.py
-----------
Async debounce implementation for the LSP completion trigger.

The Problem:
    A developer types at ~60 WPM = ~5 chars/second.
    Without debouncing, every keystroke triggers an Ollama request.
    That's 5 requests/second × 300ms each = the system is always behind.

The Solution:
    Wait N milliseconds after the LAST keystroke before firing.
    If a new keystroke arrives before N ms pass, reset the timer.
    Only fire when the user pauses.

This is a classic CS problem — it's an application of timer-based
event coalescing. Used in search boxes, window resize handlers,
and anywhere human input arrives faster than processing can handle.

Interview talking point: This is implemented as an asyncio.Task
cancellation pattern. The previous pending task is cancelled before
creating a new one — O(1) cancellation, no busy-wait, no threads.
"""

import asyncio
import logging
from typing import Callable, Coroutine, Optional, Any

logger = logging.getLogger(__name__)


class AsyncDebouncer:
    """
    Debounces async coroutine calls.

    Usage:
        debouncer = AsyncDebouncer(delay_ms=200)

        # In your keystroke handler:
        await debouncer.call(my_async_function, arg1, arg2)

        # Only fires my_async_function if no new call arrives
        # within 200ms of this one.
    """

    def __init__(self, delay_ms: int = 200) -> None:
        """
        Args:
            delay_ms: Wait this many milliseconds after the last call
                      before executing the coroutine.
        """
        self.delay_s = delay_ms / 1000.0
        self._pending_task: Optional[asyncio.Task] = None

    async def call(
        self,
        coro_fn: Callable[..., Coroutine],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Schedule `coro_fn(*args, **kwargs)` to run after the debounce delay.

        If called again before the delay expires, the previous scheduled
        call is cancelled and a new timer starts.

        Args:
            coro_fn: An async function to call.
            *args:   Positional arguments for coro_fn.
            **kwargs: Keyword arguments for coro_fn.
        """
        # Cancel any existing pending task
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass  # Expected — this is the cancellation path

        # Schedule a new delayed execution
        self._pending_task = asyncio.create_task(
            self._delayed_call(coro_fn, *args, **kwargs)
        )

    async def _delayed_call(
        self,
        coro_fn: Callable[..., Coroutine],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Wait for the debounce delay, then call the coroutine."""
        try:
            await asyncio.sleep(self.delay_s)
            await coro_fn(*args, **kwargs)
        except asyncio.CancelledError:
            logger.debug("Debounced call cancelled (new keystroke arrived)")
            raise  # Must re-raise CancelledError

    def cancel(self) -> None:
        """
        Cancel any pending debounced call immediately.
        Call this when the document closes or the cursor moves off-screen.
        """
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            logger.debug("Debouncer manually cancelled")
