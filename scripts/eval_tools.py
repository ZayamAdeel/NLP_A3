"""
Tool-routing accuracy on data/tool_eval.json (real LLM):   python scripts/eval_tools.py

Measures what the assignment's viva will ask: does the LLM pick the right tool (and not call
one when it shouldn't), are the extracted arguments right, and what does routing cost?
Only the FIRST utterance of each flow is routed by the LLM; follow-up answers are regex-filled.
"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestrator import _GATE, ToolOrchestrator  # noqa: E402
from app.tools.normalize import canonical_order_ids, normalize_transcript  # noqa: E402
from app.tools.registry import TOOLS  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "tool_eval.json"


def check_args(expected: dict, got: dict) -> bool:
    for k, v in expected.items():
        g = got.get(k)
        if k in ("query", "item"):
            if not g or v.lower() not in str(g).lower():
                return False
        elif k == "max_price":
            if g is None or float(g) != float(v):
                return False
        elif g != v:
            return False
    return True


async def evaluate(orch: ToolOrchestrator, cases: list[dict]) -> dict:
    rows, lat = [], []
    for c in cases:
        text = canonical_order_ids(normalize_transcript(c["say"]))
        gated = bool(_GATE.search(text))
        tool, args, ms = "none", {}, 0.0
        if gated:
            import time
            t = time.perf_counter()
            prop = await orch._route(text)
            ms = (time.perf_counter() - t) * 1000
            lat.append(ms)
            if prop and prop.get("tool") in TOOLS:
                tool = prop["tool"]
                args = orch._verified_args(TOOLS[tool], prop, text)
                orch._fill_from_text(TOOLS[tool], args, text, [])
        rows.append({"say": c["say"], "want": c["tool"], "got": tool, "gated": gated, "ms": ms,
                     "tool_ok": tool == c["tool"],
                     "args_ok": tool == c["tool"] and check_args(c.get("args", {}), args)})
    by_tool = {}
    for r in rows:
        by_tool.setdefault(r["want"], []).append(r)
    none_rows = by_tool.get("none", [])
    return {
        "n": len(rows),
        "tool_accuracy": sum(r["tool_ok"] for r in rows) / len(rows),
        "arg_accuracy": sum(r["args_ok"] for r in rows if r["want"] != "none") / max(1, sum(r["want"] != "none" for r in rows)),
        "false_positive_rate": sum(r["got"] != "none" for r in none_rows) / max(1, len(none_rows)),
        "gate_skipped_pct": sum(not r["gated"] for r in rows) / len(rows),
        "router_ms": {"mean": statistics.mean(lat) if lat else 0, "p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else 0},
        "per_tool": {k: sum(r["tool_ok"] for r in v) / len(v) for k, v in by_tool.items()},
        "failures": [r for r in rows if not (r["tool_ok"] and (r["args_ok"] or r["want"] == "none"))],
    }


def report(res: dict):
    print(f"\nTool routing over {res['n']} utterances")
    print(f"  tool selection accuracy : {res['tool_accuracy']:.1%}")
    print(f"  argument accuracy       : {res['arg_accuracy']:.1%}  (among utterances that need a tool)")
    print(f"  false-positive rate     : {res['false_positive_rate']:.1%}  (tool called when none is needed)")
    print(f"  skipped by the gate     : {res['gate_skipped_pct']:.1%}  (no LLM call, zero added latency)")
    print(f"  router latency          : mean {res['router_ms']['mean']:.0f} ms, p95 {res['router_ms']['p95']:.0f} ms")
    print("  per expected tool       :", {k: f"{v:.0%}" for k, v in res["per_tool"].items()})
    for f in res["failures"]:
        print(f"  MISS: {f['say']!r} wanted {f['want']}, got {f['got']}")


async def _main(args):
    from app.llm_engine import OllamaEngine
    orch = ToolOrchestrator(OllamaEngine())
    res = await evaluate(orch, json.loads(DATA.read_text(encoding="utf-8")))
    report(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    asyncio.run(_main(ap.parse_args()))
