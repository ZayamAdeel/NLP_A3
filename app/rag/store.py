"""
Vector store: exact (brute-force) inner-product search over L2-normalised
vectors == cosine similarity.

On disk (INDEX_DIR):
  vectors.npy  float32 (N, dim)
  chunks.json  [{chunk_id, doc_id, title, category, text, key}, ...]
  meta.json    {embed_model, dim, n_docs, n_chunks, built_at, version}

At load time the matrix goes into a FAISS IndexFlatIP when faiss-cpu is
installed, otherwise a NumPy matmul is used. Both are exact and return the same
results; at <=100 chunks either is far under a millisecond. The files are written
atomically (tmp + rename) so a running server never reads a half-written index.
"""
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # optional dependency
    faiss = None


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[dict], meta: dict):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunks = chunks
        self.meta = meta
        self._index = None
        if faiss is not None and len(chunks):
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    def __len__(self):
        return len(self.chunks)

    @property
    def backend(self) -> str:
        return "faiss-IndexFlatIP" if self._index is not None else "numpy-flat-ip"

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        """Top-k (chunk, cosine score), best first."""
        if not len(self.chunks):
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        if q.shape[1] != self.vectors.shape[1]:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.vectors.shape[1]} "
                             f"(embedding model changed? rebuild the index)")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
        scores = (self.vectors @ q[0])
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence ------------------------------------------------------
    def save(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        version = hashlib.sha256(self.vectors.tobytes()).hexdigest()[:12]
        meta = {**self.meta, "n_chunks": len(self.chunks), "version": version,
                "dim": int(self.vectors.shape[1]) if len(self.chunks) else 0,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _atomic(index_dir / "vectors.npy", lambda f: np.save(f, self.vectors))
        _atomic(index_dir / "chunks.json", lambda f: f.write(json.dumps(self.chunks, indent=1).encode()))
        _atomic(index_dir / "meta.json", lambda f: f.write(json.dumps(meta, indent=1).encode()))
        self.meta = meta

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        try:
            vectors = np.load(index_dir / "vectors.npy")
            chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
            meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No index in {index_dir}. Build it first: python scripts/build_index.py") from e
        if len(chunks) != len(vectors):
            raise ValueError("index is corrupt: chunk/vector count mismatch; rebuild it")
        return cls(vectors, chunks, meta)


def chunk_to_dict(chunk) -> dict:
    return asdict(chunk)


def _atomic(path: Path, writer):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)
