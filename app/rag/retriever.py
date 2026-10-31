"""
Async retriever: embed the query, search the vector store, apply the relevance
threshold. Designed to NEVER raise into the conversation loop and NEVER hang.

Result statuses
  ok           >=1 chunk scored >= min_score
  no_match     searched fine, but nothing relevant (best score below min_score)
  unavailable  could not search: index missing, embedder error, timeout, or the
               circuit breaker is open
(`skipped` is produced by the Conversation Manager for greetings/thanks.)
"""
import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from app.config import (BREAKER_COOLDOWN, BREAKER_FAILURES, CACHE_SIZE, INDEX_DIR,
                        MIN_SCORE, RETRIEVAL_TIMEOUT, TOP_K)
from app.rag.store import VectorStore

logger = logging.getLogger("ecommerce-assistant.rag")


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    status: str
    hits: list[Hit] = field(default_factory=list)
    ms: float = 0.0          # total wall time of this call
    embed_ms: float = 0.0    # query embedding only (0 on cache hit)
    search_ms: float = 0.0   # vector search only
    cached: bool = False
    detail: str = ""


class Retriever:
    def __init__(self, embedder, store: VectorStore | None = None, *, k: int = TOP_K,
                 min_score: float = MIN_SCORE, timeout: float = RETRIEVAL_TIMEOUT,
                 cache_size: int = CACHE_SIZE, breaker_failures: int = BREAKER_FAILURES,
                 breaker_cooldown: float = BREAKER_COOLDOWN, clock=time.monotonic):
        self.embedder, self.store = embedder, store
        self.k, self.min_score, self.timeout = k, min_score, timeout
        self.cache_size = cache_size
        self._cache: OrderedDict[str, RetrievalResult] = OrderedDict()
        self._failures, self._open_until = 0, 0.0
        self._breaker_failures, self._breaker_cooldown, self._clock = (
            breaker_failures, breaker_cooldown, clock)
        self.load_error = ""

    # ---- lifecycle --------------------------------------------------------
    def load(self, index_dir: Path = INDEX_DIR) -> bool:
        """Load (or reload) the on-disk index. Never raises; check `ready`."""
        try:
            self.store = VectorStore.load(index_dir)
            self._cache.clear()           # cached results belong to the old index
            self.load_error = ""
            logger.info("Index loaded: %d chunks, backend=%s, version=%s",
                        len(self.store), self.store.backend, self.store.meta.get("version"))
        except Exception as e:  # noqa: BLE001
            self.store, self.load_error = None, str(e)
            logger.warning("Index not loaded: %s", e)
        return self.ready

    @property
    def ready(self) -> bool:
        return self.store is not None and len(self.store) > 0

    async def warm_up(self):
        """Pay the embedding-model load cost at startup instead of on the first user query."""
        try:
            await asyncio.wait_for(self.embedder.embed(["warm up"]), timeout=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("Embedding warm-up failed: %s", e)

    # ---- query path -------------------------------------------------------
    async def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 2)  # noqa: E731

        if not self.ready:
            return RetrievalResult("unavailable", ms=ms(), detail=self.load_error or "index not loaded")

        key = " ".join(query.lower().split())
        if key in self._cache:
            self._cache.move_to_end(key)
            hit = self._cache[key]
            return RetrievalResult(hit.status, hit.hits, ms=ms(), cached=True)

        if self._clock() < self._open_until:
            return RetrievalResult("unavailable", ms=ms(), detail="circuit open (embedder failing)")

        try:
            te = time.perf_counter()
            vec = await asyncio.wait_for(self.embedder.embed([query]), timeout=self.timeout)
            embed_ms = round((time.perf_counter() - te) * 1000, 2)
            ts = time.perf_counter()
            raw = self.store.search(vec[0], self.k)
            search_ms = round((time.perf_counter() - ts) * 1000, 2)
        except asyncio.TimeoutError:
            self._record_failure()
            return RetrievalResult("unavailable", ms=ms(), detail=f"timed out after {self.timeout}s")
        except Exception as e:  # noqa: BLE001  (EmbeddingError, dim mismatch, ...)
            self._record_failure()
            logger.warning("Retrieval failed: %s", e)
            return RetrievalResult("unavailable", ms=ms(), detail=str(e))

        self._failures = 0
        hits = [Hit(c["chunk_id"], c["doc_id"], c["title"], c["category"], c["text"], round(s, 4))
                for c, s in raw if s >= self.min_score]
        result = RetrievalResult("ok" if hits else "no_match", hits, ms=ms(),
                                 embed_ms=embed_ms, search_ms=search_ms)
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _record_failure(self):
        self._failures += 1
        if self._failures >= self._breaker_failures:
            self._open_until = self._clock() + self._breaker_cooldown
            self._failures = 0
            logger.warning("Retrieval circuit opened for %.0fs", self._breaker_cooldown)
