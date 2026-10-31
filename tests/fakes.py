"""
Test doubles for the RAG layer. HashingEmbedder is a deterministic bag-of-words
embedder (hashed into 512 dims, L2-normalised): it lets the indexing/retrieval
PLUMBING be tested without Ollama. It is lexical, so it says nothing about the
semantic quality of the real embedding model -- use scripts/eval_retrieval.py with
the real model for that.
"""
import asyncio
import hashlib
import re

import numpy as np

_STOP = {"a", "an", "the", "is", "are", "do", "does", "i", "my", "me", "can", "to", "of",
         "and", "or", "it", "in", "on", "for", "with", "how", "what", "if", "you", "your", "be"}


class HashingEmbedder:
    model = "fake-hashing-512"
    DIM = 512

    def __init__(self, delay: float = 0.0, fail: Exception | None = None):
        self.delay, self.fail, self.calls = delay, fail, 0

    async def embed(self, texts):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in _STOP:
                    continue
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.DIM] += 1.0 if (h >> 20) & 1 else -1.0
                out[row, (h >> 8) % self.DIM] += 1.0
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return (out / norms).astype(np.float32)

    async def aclose(self):
        pass


class FakeMetrics:
    time_to_first_token = 0.05
    total_time = 0.4
    output_tokens_estimate = 5
    tokens_per_second = 12.5


class RecordingEngine:
    """Fake LLM: records every prompt it receives and streams a canned reply."""

    def __init__(self, reply="Sure, happy to help with that.", fail: Exception | None = None,
                 token_delay: float = 0.0):
        self.reply, self.fail, self.token_delay = reply, fail, token_delay
        self.prompts: list[list[dict]] = []
        self.timeline: list[str] = []

    async def stream_chat(self, messages, options=None):
        self.prompts.append(messages)
        self.timeline.append("stream")
        if self.fail:
            raise self.fail
        for word in self.reply.split(" "):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield word + " ", FakeMetrics()

    async def chat_once(self, messages, options=None, format=None):
        self.timeline.append("summarize")
        return "Customer asked about their order."


class StaticRetriever:
    """Retriever stand-in returning a fixed result (for API-level tests)."""

    def __init__(self, result, n_chunks=50):
        self.result, self.store, self.ready = result, [None] * n_chunks, True

    async def retrieve(self, query):
        return self.result


class RouterEngine:
    """Fake LLM for the tool router: replies with scripted JSON chosen by substring of the customer text."""

    def __init__(self, script: dict | None = None, raw: str | None = None, delay: float = 0.0):
        self.script, self.raw, self.delay, self.calls = script or {}, raw, delay, []

    async def chat_once(self, messages, options=None, format=None):
        self.calls.append((messages, options, format))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raw is not None:
            return self.raw
        text = messages[-1]["content"].lower()
        for needle, reply in self.script.items():
            if needle.lower() in text:
                return reply if isinstance(reply, str) else __import__("json").dumps(reply)
        return '{"tool":"none"}'


# ----------------------------------------------------------------------------- voice fakes
def tone(ms: int = 100, amp: int = 8000) -> bytes:
    """PCM16 16 kHz sine burst (counts as speech for the energy gate and the fake ASR)."""
    n = 16 * ms
    return (amp * np.sin(np.arange(n) * 0.3)).astype("<i2").tobytes()


def silence(ms: int = 100) -> bytes:
    return b"\x00\x00" * (16 * ms)


class FakeASRStream:
    """Emits a partial on the first loud chunk and the scripted final after speech then silence."""

    def __init__(self, emit, script, fail=None):
        self.emit, self.script, self.fail = emit, script, fail
        self.loud = 0
        self.finished = False
        self.closed = False

    def feed(self, pcm):
        if self.fail:
            raise self.fail
        loud = np.abs(np.frombuffer(pcm, dtype="<i2")).max() > 2000
        if loud:
            self.loud += 1
            if self.loud == 1:
                self.emit("partial", " ".join(self.script.split()[:2]))
        elif self.loud >= 2 and not self.finished:
            self.finished = True
            self.emit("final", self.script)

    def close(self):
        self.closed = True


class FakeASR:
    ready = True

    def __init__(self, script="how many days do I have to send something back", fail=None):
        self.script, self.fail, self.streams = script, fail, []

    def new_stream(self, emit):
        self.streams.append(FakeASRStream(emit, self.script, self.fail))
        return self.streams[-1]


class FakeTTS:
    ready = True
    sample_rate = 22050

    def __init__(self, delay: float = 0.0, fail: Exception | None = None):
        self.texts, self.delay, self.fail = [], delay, fail

    async def synthesize(self, text):
        self.texts.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        yield b"\x01\x00" * (self.sample_rate // 10)          # 100 ms of audio per chunk
