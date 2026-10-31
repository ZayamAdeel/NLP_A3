"""
Retrieval quality + threshold calibration against data/eval_queries.json.

  * in-corpus queries  -> hit@1 / hit@3 / hit@k and MRR (is an acceptable doc ranked high?)
  * out-of-corpus      -> how high do UNRELATED queries score? (false-match risk)
  * hard negatives     -> store-related questions the corpus cannot answer
Then sweeps MIN_SCORE and recommends a value: highest unrelated-query rejection that
still keeps >= 95% of in-corpus queries answerable.

    python scripts/eval_retrieval.py            # real embedder (needs Ollama + built index)
    python scripts/eval_retrieval.py --json results/retrieval_eval.json
    python scripts/eval_retrieval.py --fake     # dry run with a toy embedder: numbers are NOT meaningful
"""
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import INDEX_DIR, MIN_SCORE, TOP_K
from app.rag.retriever import Retriever

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def evaluate(retriever: Retriever, data: dict, k: int = TOP_K) -> dict:
    retriever.min_score = -1.0  # raw ranking: no threshold while measuring
    retriever.k = max(k, 5)
    retriever._cache.clear()

    ranks, top1_in = [], []
    for item in data["in_corpus"]:
        res = await retriever.retrieve(item["q"])
        ids = [h.doc_id for h in res.hits]
        rank = next((i + 1 for i, d in enumerate(ids) if d in item["expect"]), None)
        ranks.append(rank)
        top1_in.append(res.hits[0].score)
    n = len(ranks)
    hit = lambda m: sum(1 for r in ranks if r and r <= m) / n  # noqa: E731
    metrics = {
        "n_in_corpus": n, "hit@1": hit(1), "hit@3": hit(3), f"hit@{k}": hit(k),
        "mrr": sum(1 / r for r in ranks if r) / n,
        "misses": [data["in_corpus"][i]["q"] for i, r in enumerate(ranks) if not r or r > k],
    }

    async def top1(queries):
        return [(await retriever.retrieve(q)).hits[0].score for q in queries]

    top1_out = await top1(data["out_of_corpus"])
    top1_hard = await top1(data["hard_negatives"])

    sweep = []
    for t in [x / 100 for x in range(10, 80, 5)]:
        sweep.append({
            "threshold": t,
            "in_corpus_answerable": sum(s >= t for s in top1_in) / len(top1_in),
            "unrelated_rejected": sum(s < t for s in top1_out) / len(top1_out),
            "hard_negatives_rejected": sum(s < t for s in top1_hard) / len(top1_hard),
        })
    ok = [r for r in sweep if r["in_corpus_answerable"] >= 0.95]
    best = max(ok, key=lambda r: (r["unrelated_rejected"], -r["threshold"])) if ok else None
    return {"metrics": metrics, "sweep": sweep, "recommended_min_score": best["threshold"] if best else None,
            "score_stats": {
                "in_corpus_top1": {"min": min(top1_in), "p10": pct(top1_in, 10), "median": statistics.median(top1_in)},
                "out_of_corpus_top1": {"median": statistics.median(top1_out), "p90": pct(top1_out, 90), "max": max(top1_out)},
                "hard_negative_top1": {"median": statistics.median(top1_hard), "max": max(top1_hard)}}}


def report(r: dict, k: int):
    m = r["metrics"]
    print(f"\nIn-corpus queries: {m['n_in_corpus']}")
    print(f"  hit@1 {m['hit@1']:.2%} | hit@3 {m['hit@3']:.2%} | hit@{k} {m[f'hit@{k}']:.2%} | MRR {m['mrr']:.3f}")
    if m["misses"]:
        print("  missed (no acceptable doc in top-k):", *[f"\n    - {q}" for q in m["misses"]])
    s = r["score_stats"]
    print("\nTop-1 cosine score distribution")
    print(f"  in-corpus      min {s['in_corpus_top1']['min']:.2f}  p10 {s['in_corpus_top1']['p10']:.2f}  median {s['in_corpus_top1']['median']:.2f}")
    print(f"  unrelated      median {s['out_of_corpus_top1']['median']:.2f}  p90 {s['out_of_corpus_top1']['p90']:.2f}  max {s['out_of_corpus_top1']['max']:.2f}")
    print(f"  hard negatives median {s['hard_negative_top1']['median']:.2f}  max {s['hard_negative_top1']['max']:.2f}")
    print("\n threshold | in-corpus answerable | unrelated rejected | hard negatives rejected")
    for row in r["sweep"]:
        mark = "  <-- recommended" if row["threshold"] == r["recommended_min_score"] else ""
        print(f"    {row['threshold']:.2f}   |        {row['in_corpus_answerable']:6.1%}        |       {row['unrelated_rejected']:6.1%}       |        {row['hard_negatives_rejected']:6.1%}{mark}")
    print(f"\nCurrent MIN_SCORE = {MIN_SCORE}. Recommended: {r['recommended_min_score']}"
          "  (set via the MIN_SCORE env var or app/config.py)")
    print("Note: hard negatives are topically close to real docs, so a score threshold alone cannot reject them;\n"
          "the system prompt's 'say you don't know' rule covers that gap (see README, limitations).")


async def _main(args):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if args.fake:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fakes import HashingEmbedder
        import build_index
        embedder, index = HashingEmbedder(), Path(tempfile.mkdtemp()) / "idx"
        await build_index.build(Path(args.corpus), index, embedder)
        print("*** FAKE EMBEDDER (toy bag-of-words): numbers below are NOT representative of the real model ***")
    else:
        from app.rag.embedder import OllamaEmbedder
        embedder, index = OllamaEmbedder(timeout=30), Path(args.index)
    retriever = Retriever(embedder)
    if not retriever.load(index):
        raise SystemExit(f"Cannot load index: {retriever.load_error}")
    try:
        result = await evaluate(retriever, data, TOP_K)
    finally:
        await embedder.aclose()
    report(result, TOP_K)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    from app.config import CORPUS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--json", help="also write results to this JSON file")
    ap.add_argument("--fake", action="store_true", help="dry run with the toy test embedder")
    asyncio.run(_main(ap.parse_args()))
