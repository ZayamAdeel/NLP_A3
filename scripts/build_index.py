"""
Offline indexing pipeline: corpus -> chunks -> embeddings -> vector store.

Re-runnable and incremental. Each chunk is identified by sha256(its text); vectors
from the previous index are reused for any chunk whose hash is unchanged, so after
adding / editing / deleting a few documents only the new chunks are embedded.
If the embedding model changes, everything is re-embedded automatically.

    python scripts/build_index.py            # incremental
    python scripts/build_index.py --full     # ignore the old index, re-embed all
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_SENTENCES, CORPUS_DIR, INDEX_DIR
from app.rag.chunking import chunk_document, load_corpus
from app.rag.embedder import OllamaEmbedder
from app.rag.store import VectorStore, chunk_to_dict

BATCH = 16


async def build(corpus_dir: Path, index_dir: Path, embedder, full: bool = False) -> dict:
    docs = load_corpus(corpus_dir)
    if not docs:
        raise SystemExit(f"No .md documents found in {corpus_dir}")
    chunks = [c for doc_id, raw in docs.items() for c in chunk_document(doc_id, raw)]

    # Vectors we can reuse from the previous index (same text AND same model).
    old: dict[str, np.ndarray] = {}
    if not full:
        try:
            prev = VectorStore.load(index_dir)
            if prev.meta.get("embed_model") == embedder.model:
                old = {c["key"]: v for c, v in zip(prev.chunks, prev.vectors)}
        except (FileNotFoundError, ValueError):
            pass

    todo = [c for c in chunks if c.key not in old]
    fresh: dict[str, np.ndarray] = {}
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vecs = await embedder.embed([c.text for c in batch])
        fresh.update({c.key: v for c, v in zip(batch, vecs)})

    vectors = np.stack([old.get(c.key, fresh.get(c.key)) for c in chunks]).astype(np.float32)
    store = VectorStore(
        vectors, [chunk_to_dict(c) for c in chunks],
        {"embed_model": embedder.model, "n_docs": len(docs),
         "chunk_max_chars": CHUNK_MAX_CHARS, "chunk_overlap_sentences": CHUNK_OVERLAP_SENTENCES},
    )
    store.save(index_dir)
    kept = {c.key for c in chunks}
    return {"docs": len(docs), "chunks": len(chunks), "reused": len(chunks) - len(todo),
            "embedded": len(todo), "dropped": len([k for k in old if k not in kept]),
            "version": store.meta["version"]}


async def _main(args):
    embedder = OllamaEmbedder(timeout=120)
    try:
        stats = await build(Path(args.corpus), Path(args.index), embedder, full=args.full)
    finally:
        await embedder.aclose()
    print(f"Indexed {stats['docs']} docs -> {stats['chunks']} chunks "
          f"({stats['embedded']} embedded, {stats['reused']} reused, {stats['dropped']} dropped) "
          f"[version {stats['version']}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--full", action="store_true", help="re-embed everything")
    asyncio.run(_main(ap.parse_args()))
