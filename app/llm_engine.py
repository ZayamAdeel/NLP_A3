"""
Thin async wrapper around Ollama's local REST API.

We talk to Ollama's native /api/chat endpoint (not the LangChain/tool-calling
layer) so it's crystal clear no tools/agents/RAG are involved -- just a
messages list in, a token stream out.
"""
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

from app.config import (GENERATION_OPTIONS, OLLAMA_HOST, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL,
                        OLLAMA_TIMEOUT)


class LLMEngineError(Exception):
    """Raised when the local model backend fails or is unreachable."""


@dataclass
class GenerationMetrics:
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    output_tokens_estimate: int = 0
    eval_seconds: Optional[float] = None  # Ollama's pure decode time, when reported

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Decode speed: tokens / decode time (excludes prompt processing)."""
        if self.eval_seconds:
            dur = self.eval_seconds
        elif self.total_time and self.time_to_first_token is not None:
            dur = self.total_time - self.time_to_first_token
        else:
            return None
        return round(self.output_tokens_estimate / dur, 2) if dur > 0 else None


class OllamaEngine:
    """
    Streaming chat-completion client for a local Ollama server.

    Usage:
        engine = OllamaEngine()
        async for token, metrics in engine.stream_chat(messages):
            ...
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def stream_chat(
        self, messages: list[dict], options: dict | None = None, format: dict | None = None
    ) -> AsyncIterator[tuple[str, GenerationMetrics]]:
        """
        Streams response tokens one at a time.

        Yields (token_text, metrics) tuples. `metrics` is progressively filled
        in and the final yielded metrics object has time_to_first_token,
        total_time, and a rough tokens/sec figure -- used both for the
        live UI and for the Phase VI latency benchmarking.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {**GENERATION_OPTIONS, **(options or {})},
        }
        if format is not None:
            payload["format"] = format  # JSON-schema constrained decoding (structured outputs)

        metrics = GenerationMetrics()
        start = time.perf_counter()
        first_token_seen = False
        token_count = 0

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST", f"{self.host}/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMEngineError(
                            f"Ollama returned status {response.status_code}: {body[:300]!r}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("error"):
                            raise LLMEngineError(chunk["error"])

                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            if not first_token_seen:
                                metrics.time_to_first_token = round(
                                    time.perf_counter() - start, 4
                                )
                                first_token_seen = True
                            token_count += 1
                            metrics.output_tokens_estimate = token_count
                            metrics.total_time = round(time.perf_counter() - start, 4)
                            yield content, metrics

                        if chunk.get("done"):
                            metrics.total_time = round(time.perf_counter() - start, 4)
                            # Prefer Ollama's own eval_count when available -- more
                            # accurate than our "one chunk ~= one token" estimate.
                            if chunk.get("eval_count"):
                                metrics.output_tokens_estimate = chunk["eval_count"]
                            if chunk.get("eval_duration"):
                                # total_time stays wall-clock; decode time is kept apart
                                metrics.eval_seconds = chunk["eval_duration"] / 1e9
                            break
        except httpx.ConnectError as e:
            raise LLMEngineError(
                f"Could not connect to Ollama at {self.host}. Is `ollama serve` running "
                f"and is the model pulled (`ollama pull {self.model}`)?"
            ) from e
        except httpx.HTTPError as e:
            raise LLMEngineError(f"HTTP error talking to Ollama: {e}") from e

    async def chat_once(self, messages: list[dict], options: dict | None = None,
                        format: dict | None = None) -> str:
        """Non-streaming convenience wrapper (summaries and tool routing)."""
        text = ""
        async for token, _ in self.stream_chat(messages, options=options, format=format):
            text += token
        return text
