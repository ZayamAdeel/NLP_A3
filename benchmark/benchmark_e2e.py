"""
End-to-end latency, RAG vs no-RAG, through the real ConversationManager + Ollama.

For each prompt it reports what the USER feels: time from "message received" to the
first streamed token (retrieval + prompt processing + first decode), plus total time
and decode speed. Run it to show retrieval does not make the assistant feel slower.

    python benchmark/benchmark_e2e.py --runs 3
"""
import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import INDEX_DIR
from app.conversation_manager import ConversationManager
from app.llm_engine import OllamaEngine
from app.rag.embedder import OllamaEmbedder
from app.rag.retriever import Retriever

PROMPTS = [
    "Can I use the WaveSound H200 headphones with my laptop?",
    "How long does express shipping take?",
    "Can I return a gift card?",
    "My payment failed at checkout, what should I do?",
    "Does the LinkHub support 4K over HDMI?",
]


async def one_turn(manager: ConversationManager, text: str) -> dict:
    session = manager.create_session()          # fresh session: no history effects
    t0 = time.perf_counter()
    first = None
    done = {}
    async for kind, payload in manager.handle_user_message(session, text):
        if kind == "token" and first is None:
            first = time.perf_counter() - t0
        elif kind == "done":
            done = payload
    total = time.perf_counter() - t0
    return {"first_token_s": first, "total_s": total, "tps": done.get("tokens_per_second"),
            "retrieval_ms": (done.get("retrieval") or {}).get("ms", 0.0)}


async def bench(label: str, manager: ConversationManager, runs: int) -> dict:
    await one_turn(manager, "hello")            # warm the chat model (first load is slow on CPU)
    rows = [await one_turn(manager, p) for _ in range(runs) for p in PROMPTS]
    f = lambda k: [r[k] for r in rows if r[k] is not None]  # noqa: E731
    out = {"label": label, "n": len(rows),
           "first_token_s": statistics.mean(f("first_token_s")),
           "total_s": statistics.mean(f("total_s")),
           "tps": statistics.mean(f("tps")) if f("tps") else None,
           "retrieval_ms": statistics.mean(f("retrieval_ms"))}
    print(f"{label:<10} first token {out['first_token_s']:.2f} s | total {out['total_s']:.2f} s | "
          f"decode {out['tps'] or 0:.1f} tok/s | retrieval {out['retrieval_ms']:.1f} ms  (n={out['n']})")
    return out


async def main(runs: int):
    engine, embedder = OllamaEngine(), OllamaEmbedder()
    retriever = Retriever(embedder)
    if not retriever.load(INDEX_DIR):
        raise SystemExit(f"Cannot load index: {retriever.load_error}")
    await retriever.warm_up()
    try:
        base = await bench("no RAG", ConversationManager(engine=engine), runs)
        retriever._cache.clear()
        rag = await bench("RAG", ConversationManager(engine=engine, retriever=retriever), runs)
    finally:
        await embedder.aclose()
    print(f"\nRAG adds {rag['first_token_s'] - base['first_token_s']:+.2f} s to time-to-first-token "
          f"({rag['retrieval_ms']:.0f} ms of that is retrieval; the rest is the longer prompt to process).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=2)
    asyncio.run(main(ap.parse_args().runs))
