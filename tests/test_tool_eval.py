"""The tool-eval harness itself, plus a gate/verification check that needs no LLM."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import eval_tools  # noqa: E402
from app.orchestrator import ToolOrchestrator  # noqa: E402
from app.tools.normalize import canonical_order_ids, normalize_transcript  # noqa: E402

CASES = json.loads((ROOT / "data" / "tool_eval.json").read_text(encoding="utf-8"))


class OracleEngine:
    """A perfect router: answers with the labelled tool and the labelled free-text fields."""

    def __init__(self):
        self.by_text = {canonical_order_ids(normalize_transcript(c["say"])): c for c in CASES}

    async def chat_once(self, messages, options=None, format=None):
        text = messages[-1]["content"].removeprefix("Customer: ")
        c = self.by_text[text]
        out = {"tool": c["tool"]}
        out.update({k: v for k, v in c.get("args", {}).items() if k in ("query", "item", "preference", "action")})
        return json.dumps(out)


class NoneEngine:
    async def chat_once(self, messages, options=None, format=None):
        return '{"tool":"none"}'


class EvalHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_scores_perfectly_so_gate_and_verifiers_lose_nothing(self):
        res = await eval_tools.evaluate(ToolOrchestrator(OracleEngine()), CASES)
        self.assertEqual(res["failures"], [], res["failures"])
        self.assertEqual((res["tool_accuracy"], res["arg_accuracy"], res["false_positive_rate"]), (1.0, 1.0, 0.0))

    async def test_useless_router_scores_zero_on_tool_cases_but_never_false_positive(self):
        res = await eval_tools.evaluate(ToolOrchestrator(NoneEngine()), CASES)
        self.assertEqual(res["false_positive_rate"], 0.0)
        self.assertEqual(res["arg_accuracy"], 0.0)
        wanted_tools = sum(c["tool"] != "none" for c in CASES) / len(CASES)
        self.assertAlmostEqual(res["tool_accuracy"], 1 - wanted_tools, places=3)

    def test_labelled_set_is_balanced_and_valid(self):
        from app.tools.registry import TOOLS
        self.assertGreaterEqual(len(CASES), 30)
        for t in TOOLS:
            self.assertGreaterEqual(sum(c["tool"] == t for c in CASES), 5, t)
        self.assertGreaterEqual(sum(c["tool"] == "none" for c in CASES), 8)


if __name__ == "__main__":
    unittest.main()
