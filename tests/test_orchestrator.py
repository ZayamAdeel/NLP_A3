"""Orchestrator tests: gating, routing, 'code verifies', slot filling, failures."""
import asyncio
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.orchestrator import (MAX_ASKS_PER_FIELD, Outcome, ToolOrchestrator, condition_from_text, item_from_text,  # noqa: E402
                              packaging_from_text, price_from_text)
from app.tools.db import ensure_db  # noqa: E402
from app.tools.registry import TOOLS  # noqa: E402
from app.tools.types import ToolResult  # noqa: E402
from fakes import RouterEngine  # noqa: E402

SCRIPT = {
    "wireless headphones": {"tool": "search_products", "query": "wireless headphones", "max_price": 10000},
    "headphones under": {"tool": "search_products", "query": "headphones", "max_price": 99999},
    "happening with order": {"tool": "lookup_order", "order_id": "BC1024"},
    "where is my order": {"tool": "lookup_order"},
    "can i return": {"tool": "check_return_eligibility", "order_id": "BC1025", "item": "headphones",
                     "condition": "unused", "has_original_packaging": True},   # model "guesses" condition/packaging
    "return policy": {"tool": "none"},
}


class OrchCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = ensure_db(self.tmp / "t.db", force=True)
        self.session = SimpleNamespace(verified_orders=set(), failed_verifications=0, pending_tool=None)
        self.engine = RouterEngine(SCRIPT)
        self.orch = ToolOrchestrator(self.engine, db_path=self.db, today=lambda: date(2026, 10, 24))

    async def say(self, text):
        return await self.orch.run(self.session, text)


class ExtractorTests(unittest.TestCase):
    def test_condition(self):
        for text, want in [("it's unused", "unused"), ("I haven't opened it", "unused"), ("not used at all", "unused"),
                           ("it is used", "used"), ("arrived broken", "damaged"), ("hello", None)]:
            self.assertEqual(condition_from_text(text), want, text)

    def test_packaging(self):
        self.assertIs(packaging_from_text("I still have the original box"), True)
        self.assertIs(packaging_from_text("I threw away the box"), False)
        self.assertIs(packaging_from_text("without the packaging"), False)
        self.assertIsNone(packaging_from_text("yes"))
        self.assertIs(packaging_from_text("yes", bare_yes_no=True), True)
        self.assertIs(packaging_from_text("no", bare_yes_no=True), False)

    def test_price_and_item(self):
        self.assertEqual(price_from_text("anything under 10,000 rupees"), 10000.0)
        self.assertEqual(price_from_text("below Rs 2500"), 2500.0)
        self.assertIsNone(price_from_text("I have 5 days"))
        self.assertEqual(item_from_text("the headphones please"), "headphones")


class GateAndRouteTests(OrchCase):
    async def test_chitchat_never_calls_the_llm(self):
        out = await self.say("hello there, how are you")
        self.assertEqual(out.kind, "none")
        self.assertEqual(self.engine.calls, [])

    async def test_policy_question_passes_gate_but_router_says_none(self):
        out = await self.say("what is your return policy")
        self.assertEqual((out.kind, len(self.engine.calls)), ("none", 1))

    async def test_router_uses_schema_and_deterministic_options(self):
        await self.say("what is your return policy")
        _, options, fmt = self.engine.calls[0]
        self.assertEqual(options["temperature"], 0)
        self.assertEqual(fmt["properties"]["tool"]["enum"], [*TOOLS, "none"])

    async def test_search_end_to_end(self):
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(out.kind, "result")
        self.assertIn("WaveSound H200", out.result_text)
        self.assertEqual(out.calls[0].name, "search_products")
        self.assertEqual(out.sources[0]["doc_id"], "products/product_001_wavesound_headphones.md")

    async def test_model_price_not_in_text_is_replaced_by_text(self):
        out = await self.say("headphones under 5000 rupees")        # model said 99999
        self.assertEqual(out.calls[0].args["max_price"], 5000.0)

    async def test_router_cache_avoids_second_llm_call(self):
        await self.say("find wireless headphones under 10000 rupees")
        await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(len(self.engine.calls), 1)


class OrderFlowTests(OrchCase):
    async def test_lookup_asks_for_code_then_answers_without_second_llm_call(self):
        out = await self.say("what's happening with order BC1024")
        self.assertEqual(out.kind, "ask")
        self.assertIn("last four digits", out.ask_text)
        self.assertEqual(self.session.pending_tool["tool"], "lookup_order")

        out = await self.say("4721")                                 # answer filled by regex
        self.assertEqual(out.kind, "result")
        self.assertIn("status shipped", out.result_text)
        self.assertIsNone(self.session.pending_tool)
        self.assertEqual(len(self.engine.calls), 1)                  # router ran once only
        self.assertFalse(out.routed_by_llm)

    async def test_no_order_number_asks_for_it_first(self):
        out = await self.say("where is my order")
        self.assertEqual(out.kind, "ask")
        self.assertIn("order number", out.ask_text)
        out = await self.say("it's BC1024")
        self.assertIn("last four digits", out.ask_text)

    async def test_wrong_code_gets_neutral_result(self):
        await self.say("what's happening with order BC1024")
        out = await self.say("0000")
        self.assertEqual(out.kind, "result")
        self.assertIn("couldn't find an order matching", out.result_text)
        self.assertEqual(self.session.failed_verifications, 1)

    async def test_code_is_never_logged_in_call_record(self):
        await self.say("what's happening with order BC1024")
        out = await self.say("4721")
        self.assertNotIn("verification_code", out.calls[0].args)

    async def test_topic_change_drops_pending_request(self):
        await self.say("what's happening with order BC1024")
        out = await self.say("actually find wireless headphones under 10000 rupees")
        self.assertEqual(out.kind, "result")
        self.assertEqual(out.calls[0].name, "search_products")
        self.assertIsNone(self.session.pending_tool)


class ReturnFlowTests(OrchCase):
    async def test_hallucinated_condition_and_packaging_are_ignored_then_asked(self):
        # The scripted model claims unused + packaging, but the customer said neither.
        out = await self.say("can I return the headphones from order BC1025")
        self.assertEqual(out.kind, "ask")
        self.assertIn("unused, used, or damaged", out.ask_text)
        self.assertNotIn("condition", self.session.pending_tool["args"])

    async def test_full_voice_style_flow(self):
        out = await self.say("can I return the headphones from order BC1025")
        out = await self.say("it's unused")
        self.assertIn("original packaging", out.ask_text)
        out = await self.say("yes")                                  # bare yes -> packaging
        self.assertIn("last four digits", out.ask_text)
        out = await self.say("3308")
        self.assertEqual(out.kind, "result")
        self.assertIn("ELIGIBLE", out.result_text)
        self.assertIn("20 day(s) remain", out.result_text)
        self.assertEqual(len(self.engine.calls), 1)

    async def test_everything_in_one_sentence(self):
        out = await self.say("can I return the headphones from order BC1025, they are unused and I still have the box")
        self.assertIn("last four digits", out.ask_text)              # only the code is missing
        out = await self.say("3308")
        self.assertIn("ELIGIBLE", out.result_text)

    async def test_ambiguous_item_asks_which_one_then_resolves(self):
        self.session.verified_orders.add("BC1025")
        args = dict(order_id="BC1025", item="bc", condition="unused", has_original_packaging=True)  # "bc" is in both product ids
        out = await self.orch._execute_or_ask(self.session, TOOLS["check_return_eligibility"], args, Outcome())
        self.assertEqual(out.kind, "ask")
        self.assertIn("Which one do you mean", out.ask_text)
        out = await self.say("the cable")
        self.assertEqual(out.kind, "result")
        self.assertIn("FlexCharge", out.result_text)
        self.assertIn("ELIGIBLE", out.result_text)


class FailureTests(OrchCase):
    async def test_router_garbage_json_means_no_tool(self):
        self.orch.engine = RouterEngine(raw="not json at all {")
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(out.kind, "none")

    async def test_router_timeout_means_no_tool_and_is_fast(self):
        self.orch.engine = RouterEngine(SCRIPT, delay=2.0)
        self.orch.router_timeout = 0.1
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(out.kind, "none")
        self.assertLess(out.tools_ms, 600)

    async def test_non_object_router_reply_is_ignored(self):
        self.orch.engine = RouterEngine(raw="[1,2,3]")
        self.assertEqual((await self.say("find wireless headphones")).kind, "none")

    async def test_tool_timeout_becomes_friendly_failure(self):
        async def slow(args, ctx):
            await asyncio.sleep(2)
            return ToolResult("ok", "never")
        original = TOOLS["search_products"].handler
        TOOLS["search_products"].handler = slow
        self.addCleanup(setattr, TOOLS["search_products"], "handler", original)
        self.orch.tool_timeout = 0.1
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual((out.kind, out.calls[0].status), ("result", "error"))
        self.assertIn("timed out", out.calls[0].summary)
        self.assertIn("FAILED", out.result_text)

    async def test_tool_crash_becomes_friendly_failure(self):
        async def boom(args, ctx):
            raise RuntimeError("db exploded")
        original = TOOLS["search_products"].handler
        TOOLS["search_products"].handler = boom
        self.addCleanup(setattr, TOOLS["search_products"], "handler", original)
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(out.calls[0].status, "error")
        self.assertIn("FAILED", out.result_text)

    async def test_non_answer_drops_pending_request(self):
        await self.say("what's happening with order BC1024")
        out = await self.say("hmm what about 12")
        self.assertEqual(out.kind, "none")
        self.assertIsNone(self.session.pending_tool)

    async def test_same_question_is_not_asked_forever(self):
        tool = TOOLS["lookup_order"]
        kinds = [self.orch._ask(self.session, tool, {"order_id": "BC1024"}, "verification_code", Outcome()).kind
                 for _ in range(MAX_ASKS_PER_FIELD + 1)]
        self.assertEqual(kinds, ["ask"] * MAX_ASKS_PER_FIELD + ["none"])
        self.assertIsNone(self.session.pending_tool)

    async def test_missing_db_is_a_tool_failure_not_a_crash(self):
        self.orch.db_path = self.tmp / "nope" / "missing.db"
        out = await self.say("find wireless headphones under 10000 rupees")
        self.assertEqual(out.kind, "result")
        self.assertEqual(out.calls[0].status, "error")


if __name__ == "__main__":
    unittest.main()
