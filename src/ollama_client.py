"""
ollama_client.py
----------------
Async HTTP client for the local Ollama inference engine.

Architecture note: Ollama runs as a separate process on localhost:11434.
We communicate with it via HTTP — NOT by importing a Python library.
This is the correct pattern: Ollama manages its own GPU/CPU memory,
model loading, and KV-cache. We are just a client.

CRITICAL concurrency note:
    httpx async calls MUST run inside an asyncio context. When called
    from inside a pygls LSP handler (which IS async), this works directly.
    If called from a sync context, use asyncio.run() or run_in_executor.

Streaming:
    Ollama's /api/generate endpoint supports streaming via NDJSON
    (newline-delimited JSON). Each line is one token chunk. We yield
    these chunks as they arrive — the editor gets the first token
    in ~100ms instead of waiting for the full completion.

Interview talking point: Streaming is non-trivial to implement correctly.
You must handle partial JSON lines, detect the "done" signal, handle
network timeouts gracefully, and yield chunks in the correct format
for the LSP CompletionItem response.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

import httpx

from src.config import OllamaConfig
from src.fim_builder import FIMPayload, FIM_STOP_SEQUENCES, extract_clean_completion

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable at startup or during a request."""
    pass


class OllamaTimeoutError(Exception):
    """Raised when Ollama exceeds the configured timeout."""
    pass


class OllamaClient:
    """
    Async client for Ollama's /api/generate endpoint.

    Usage:
        client = OllamaClient(config)
        await client.health_check()  # verify Ollama is running

        # Streaming (recommended for real-time ghost text)
        async for chunk in client.stream_completion(payload):
            send_to_editor(chunk)

        # Full response (for eval harness)
        full_text = await client.complete(payload)
    """

    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize and reuse a single httpx client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=self.config.timeout_seconds,
                    write=5.0,
                    pool=5.0,
                ),
            )
        return self._client

    async def health_check(self) -> bool:
        """
        Verify Ollama is running and the configured model is available.

        Returns:
            True if healthy.

        Raises:
            OllamaConnectionError if unreachable or model not found.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()

            tags_data = response.json()
            available_models = [m["name"] for m in tags_data.get("models", [])]

            # Check if our model (or a prefix match) is available
            model_base = self.config.model.split(":")[0]
            found = any(
                m == self.config.model or m.startswith(model_base)
                for m in available_models
            )

            if not found:
                logger.warning(
                    f"Model '{self.config.model}' not found in Ollama. "
                    f"Available: {available_models}. "
                    f"Run: ollama pull {self.config.model}"
                )
                # Don't raise — the model might still work (Ollama auto-downloads)
            else:
                logger.info(f"Ollama healthy. Model '{self.config.model}' available.")

            return True

        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise OllamaConnectionError(f"Ollama health check failed: {e}")

    def _build_request_body(
        self,
        payload: FIMPayload,
        stream: bool = True,
    ) -> dict:
        """
        Build the JSON body for Ollama's /api/generate endpoint.

        Key parameters:
            raw: True — we handle the full prompt format ourselves.
                 Do NOT use this with 'messages' API (that's /api/chat).
            stop: Stop sequences — Ollama halts generation at any of these.
                  Critical: prevents the model from generating beyond the
                  completion boundary.
        """
        return {
            "model": self.config.model,
            "prompt": payload.prompt,
            "stream": stream,
            "raw": True,  # We handle FIM tokens ourselves
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repeat_penalty,
                "num_predict": 30,
                "stop": FIM_STOP_SEQUENCES + ["\n\n\n", "```", "# Context"],
            },
        }

    async def stream_completion(
        self,
        payload: FIMPayload,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion tokens from Ollama as they are generated.

        Yields raw token strings (may be partial words — that's normal).
        The caller (LSP server) accumulates and sends to the editor.

        This is an async generator. Use it with `async for`:
            async for token in client.stream_completion(payload):
                buffer += token

        NDJSON streaming format from Ollama:
            {"model":"qwen...","response":" def","done":false}
            {"model":"qwen...","response":" hello","done":false}
            {"model":"qwen...","response":"","done":true,"total_duration":...}

        Args:
            payload: FIMPayload from fim_builder.build_fim_prompt()

        Yields:
            Individual token strings.

        Raises:
            OllamaTimeoutError: If generation exceeds timeout.
            OllamaConnectionError: If connection drops mid-stream.
        """
        client = await self._get_client()
        body = self._build_request_body(payload, stream=True)

        t_start = time.perf_counter()
        first_token = True

        try:
            async with client.stream("POST", "/api/generate", json=body) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama chunk: {line!r}")
                        continue

                    token = chunk.get("response", "")
                    done = chunk.get("done", False)

                    if first_token and token:
                        ttft = (time.perf_counter() - t_start) * 1000
                        logger.debug(f"Time-to-first-token: {ttft:.1f}ms")
                        first_token = False

                    if token:
                        yield token

                    if done:
                        total_ms = (time.perf_counter() - t_start) * 1000
                        logger.debug(
                            f"Generation complete | "
                            f"total={total_ms:.1f}ms | "
                            f"eval_count={chunk.get('eval_count', '?')} tokens"
                        )
                        break

        except httpx.TimeoutException:
            raise OllamaTimeoutError(
                f"Ollama timed out after {self.config.timeout_seconds}s. "
                f"Try a smaller model or increase timeout in config."
            )
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Lost connection to Ollama: {e}")

    async def complete(self, payload: FIMPayload) -> str:
        """
        Non-streaming completion — accumulates full output and returns it.

        Used by the eval harness where we need the complete text before
        running CodeBLEU scoring.

        Args:
            payload: FIMPayload from fim_builder.

        Returns:
            Clean completion string (FIM stop tokens stripped).
        """
        buffer = ""
        async for token in self.stream_completion(payload):
            buffer += token

        clean = extract_clean_completion(buffer)
        logger.debug(f"Complete() returned {len(clean)} chars")
        return clean

    async def close(self) -> None:
        """Clean up the httpx client. Call on server shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("Ollama client closed.")
