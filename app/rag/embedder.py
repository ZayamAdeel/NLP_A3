"""
Embedding client: Ollama's /api/embed (batch-capable, returns L2-normalised
vectors). Fully async so a query embedding never blocks the event loop, and a
single shared HTTP client keeps the connection alive between queries.
"""
import numpy as np
import httpx

from app.config import EMBED_MODEL, EMBED_TIMEOUT, OLLAMA_HOST, OLLAMA_KEEP_ALIVE


class EmbeddingError(Exception):
    """Embedding backend unreachable, timed out, or returned garbage."""


def _normalize(m: np.ndarray) -> np.ndarray:
    # Defensive: guarantees cosine similarity == dot product regardless of backend.
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_HOST,
                 timeout: float = EMBED_TIMEOUT):
        self.model = model
        self._client = httpx.AsyncClient(base_url=host.rstrip("/"), timeout=timeout)

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix (len(texts), dim), rows L2-normalised."""
        try:
            resp = await self._client.post(
                "/api/embed",
                json={"model": self.model, "input": texts, "keep_alive": OLLAMA_KEEP_ALIVE},
            )
            resp.raise_for_status()
            vecs = resp.json()["embeddings"]
            if len(vecs) != len(texts):
                raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(vecs)}")
            return _normalize(np.asarray(vecs, dtype=np.float32))
        except EmbeddingError:
            raise
        except httpx.TimeoutException as e:
            raise EmbeddingError(f"embedding request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise EmbeddingError(
                f"cannot reach Ollama for embeddings (is it running, and did you "
                f"`ollama pull {self.model}`?)") from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"embedding failed: {e}") from e

    async def aclose(self):
        await self._client.aclose()
