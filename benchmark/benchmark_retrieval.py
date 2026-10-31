"""
Retrieval latency benchmark (Assignment 2, Phase III) -- retrieval ONLY, no generation.

Measures, per query: query-embedding time, vector-search time, and total retrieve()
time, for (1) cold queries (cache miss), (2) repeated queries (cache hit) and
(3) N concurrent users. Prints a markdown table to paste into the README and checks
the < 1 s target on the p95.

    python benchmark/benchmark_retrieval.py              # real embedder (Ollama + built index)
    python benchmark/benchmark_retrieval.py --json results/retrieval_latency.json
    python benchmark/benchmark_retrieval.py --fake       # dry run, toy embedder (numbers meaningless)
"""
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import CORPUS_DIR, INDEX_DIR
from app.rag.retriever import Retriever

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]  # noqa: E731
    return {"mean": round(statistics.mean(xs), 2), "p50": round(p(50), 2),
            "p95": round(p(95), 2), "max": round(xs[-1], 2)}


async def run(retriever: Retriever, queries: list[str], concurrency: int = 8) -> dict:
    await retriever.warm_up()
    retriever._cache.clear()

    cold = [await retriever.retrieve(q) for q in queries]          # every query is a cache miss
    warm = [await retriever.retrieve(q) for q in queries]          # same queries again

    retriever._cache.clear()
    batch = queries[:concurrency]
    t0 = time.perf_counter()
    conc = await asyncio.gather(*[retriever.retrieve(q) for q in batch])
    conc_wall = (time.perf_counter() - t0) * 1000

    failed = [r for r in cold + warm + list(conc) if r.status == "unavailable"]
    return {
        "n_queries": len(queries),
        "failures": len(failed),
        "cold_total_ms": summarize([r.ms for r in cold]),
        "cold_embed_ms": summarize([r.embed_ms for r in cold]),
        "cold_search_ms": summarize([r.search_ms for r in cold]),
        "cached_total_ms": summarize([r.ms for r in warm]),
        "cache_hit_rate_second_pass": sum(r.cached for r in warm) / len(warm),
        "concurrent": {"users": len(batch), "wall_ms": round(conc_wall, 2),
                       "per_request_ms": summarize([r.ms for r in conc])},
    }


def report(r: dict):
    print(f"\nRetrieval latency over {r['n_queries']} queries (failures: {r['failures']})\n")
    print("| Scenario | mean ms | p50 ms | p95 ms | max ms |")
    print("|---|---|---|---|---|")
    for label, key in [("Cold: query embedding", "cold_embed_ms"), ("Cold: vector search", "cold_search_ms"),
                       ("Cold: total retrieve()", "cold_total_ms"), ("Cached: total retrieve()", "cached_total_ms")]:
        s = r[key]
        print(f"| {label} | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} |")
    c = r["concurrent"]
    s = c["per_request_ms"]
    print(f"| {c['users']} concurrent users, per request | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} |")
    print(f"\n{c['users']} concurrent retrievals finished in {c['wall_ms']} ms wall time "
          f"(sequential would be roughly {c['users']} x cold mean = {round(c['users'] * r['cold_total_ms']['mean'])} ms)")
    verdict = "PASS" if r["cold_total_ms"]["p95"] < 1000 else "FAIL"
    print(f"Target (< 1000 ms cold p95): {verdict} ({r['cold_total_ms']['p95']} ms)")


async def _main(args):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    queries = [i["q"] for i in data["in_corpus"]] + data["out_of_corpus"] + data["hard_negatives"]
    if args.fake:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from fakes import HashingEmbedder
        import build_index
        embedder, index = HashingEmbedder(), Path(tempfile.mkdtemp()) / "idx"
        await build_index.build(CORPUS_DIR, index, embedder)
        print("*** FAKE EMBEDDER: numbers are NOT representative of the real model ***")
    else:
        from app.rag.embedder import OllamaEmbedder
        embedder, index = OllamaEmbedder(timeout=30), Path(args.index)
    retriever = Retriever(embedder, min_score=0.0)
    if not retriever.load(index):
        raise SystemExit(f"Cannot load index: {retriever.load_error}")
    try:
        result = await run(retriever, queries, args.concurrency)
    finally:
        await embedder.aclose()
    print(f"Vector store backend: {retriever.store.backend}, chunks: {len(retriever.store)}")
    report(result)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--fake", action="store_true")
    asyncio.run(_main(ap.parse_args()))
