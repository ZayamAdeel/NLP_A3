"""Tests for the three tools: catalogue search, order lookup, return eligibility."""
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

from app.tools.db import ensure_db  # noqa: E402
from app.tools.registry import TOOLS, tool_schemas  # noqa: E402
from app.tools.returns import evaluate  # noqa: E402
from app.tools.types import ToolContext  # noqa: E402


class ToolCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = ensure_db(self.tmp / "t.db", force=True)
        self.session = SimpleNamespace(verified_orders=set(), failed_verifications=0)

    def ctx(self):
        return ToolContext(self.session, self.db, date(2026, 10, 24))

    async def call(self, tool, **args):
        return await TOOLS[tool].handler(args, self.ctx())


class SearchTests(ToolCase):
    async def test_user_example_wireless_headphones_under_10000(self):
        r = await self.call("search_products", query="wireless headphones", max_price=10000)
        self.assertEqual(r.status, "ok")
        self.assertEqual([p["id"] for p in r.data["products"]], ["BC-HEAD-001"])
        self.assertIn("9,500 rupees", r.summary)
        self.assertEqual(r.sources[0]["doc_id"], "products/product_001_wavesound_headphones.md")

    async def test_filters_exclude_then_hint_closest(self):
        r = await self.call("search_products", query="wireless headphones", max_price=5000)
        self.assertEqual(r.data["count"], 0)
        self.assertIn("closest match", r.summary)
        self.assertIn("WaveSound", r.summary)

    async def test_in_stock_only_excludes_out_of_stock(self):
        anyway = await self.call("search_products", query="webcam")
        self.assertEqual(anyway.data["count"], 1)
        self.assertIn("out of stock", anyway.summary)         # ClearView has stock 0 in the seed
        only = await self.call("search_products", query="webcam", in_stock_only=True)
        self.assertEqual(only.data["count"], 0)

    async def test_limit_three_and_cheapest_first(self):
        r = await self.call("search_products", category="computer accessories")
        prices = [p["price_pkr"] for p in r.data["products"]]
        self.assertEqual(len(prices), 3)
        self.assertEqual(prices, sorted(prices))
        self.assertGreater(r.data["count"], 3)

    async def test_no_arguments_asks_for_query(self):
        r = await self.call("search_products")
        self.assertEqual((r.status, r.missing), ("needs_info", ["query"]))

    async def test_nothing_matches(self):
        r = await self.call("search_products", query="dishwasher")
        self.assertEqual(r.data["count"], 0)


class OrderTests(ToolCase):
    async def test_verified_lookup(self):
        r = await self.call("lookup_order", order_id="BC1024", verification_code="4721")
        self.assertEqual(r.status, "ok")
        self.assertIn("status shipped", r.summary)
        self.assertIn("TCS-884201", r.summary)
        self.assertIn("Monday 26 October", r.summary)       # 2026-10-24 + 2 days
        self.assertIn("BC1024", self.session.verified_orders)

    async def test_id_variants_are_accepted(self):
        for raw in ("bc 1024", "B C 1024", "bc-1024"):
            r = await self.call("lookup_order", order_id=raw, verification_code="4721")
            self.assertEqual(r.status, "ok", raw)

    async def test_wrong_code_and_unknown_order_look_identical(self):
        wrong = await self.call("lookup_order", order_id="BC1024", verification_code="0000")
        ghost = await self.call("lookup_order", order_id="BC9999", verification_code="4721")
        self.assertEqual((wrong.status, wrong.summary), (ghost.status, ghost.summary))
        self.assertNotIn("BC1024", self.session.verified_orders)

    async def test_lockout_after_three_failures_even_with_right_code(self):
        for _ in range(3):
            await self.call("lookup_order", order_id="BC1024", verification_code="0000")
        r = await self.call("lookup_order", order_id="BC1024", verification_code="4721")
        self.assertEqual(r.status, "locked")

    async def test_delivered_order_reports_days(self):
        r = await self.call("lookup_order", order_id="BC1025", verification_code="3308")
        self.assertIn("delivered 10 day(s) ago", r.summary)


class ReturnTests(ToolCase):
    async def check(self, order, item, cond, pack=True, code="auto"):
        codes = {"BC1024": "4721", "BC1025": "3308", "BC1026": "9156", "BC1028": "6642"}
        args = dict(order_id=order, item=item, condition=cond, has_original_packaging=pack)
        if code == "auto":
            args["verification_code"] = codes[order]
        elif code:
            args["verification_code"] = code
        return await self.call("check_return_eligibility", **args)

    async def test_eligible_with_days_remaining(self):
        r = await self.check("BC1025", "headphones", "unused")
        self.assertIs(r.data["eligible"], True)
        self.assertIn("20 day(s) remain", r.summary)       # 10 days since delivery, 30-day window

    async def test_used_not_eligible(self):
        self.assertEqual((await self.check("BC1025", "headphones", "used")).data["reason"], "used")

    async def test_missing_packaging_not_eligible(self):
        r = await self.check("BC1025", "headphones", "unused", pack=False)
        self.assertEqual(r.data["reason"], "no_packaging")

    async def test_window_expired(self):
        r = await self.check("BC1026", "keyboard", "unused")
        self.assertEqual(r.data["reason"], "window_expired")
        self.assertIn("15 day(s) past", r.summary)

    async def test_gift_card_and_final_sale_not_returnable(self):
        self.assertEqual((await self.check("BC1028", "gift card", "unused")).data["reason"], "non_returnable")
        self.assertEqual((await self.check("BC1028", "phone case", "unused")).data["reason"], "non_returnable")

    async def test_damaged_goes_to_review_not_refund(self):
        r = await self.check("BC1028", "lamp", "damaged")
        self.assertEqual(r.data["eligible"], "review")
        self.assertIn("No refund can be promised", r.summary)

    async def test_not_delivered_yet(self):
        self.assertEqual((await self.check("BC1024", "headphones", "unused")).data["reason"], "not_delivered")

    async def test_unverified_order_needs_code_then_verifies(self):
        r = await self.check("BC1025", "headphones", "unused", code=None)
        self.assertEqual((r.status, r.missing), ("needs_verification", ["verification_code"]))
        r = await self.check("BC1025", "headphones", "unused", code="3308")
        self.assertEqual(r.status, "ok")
        r = await self.check("BC1025", "headphones", "unused", code=None)   # now remembered
        self.assertEqual(r.status, "ok")

    async def test_wrong_code_blocked(self):
        r = await self.check("BC1025", "headphones", "unused", code="1111")
        self.assertEqual(r.status, "not_found")
        self.assertNotIn("BC1025", self.session.verified_orders)

    async def test_item_not_on_order_lists_contents(self):
        r = await self.check("BC1025", "keyboard", "unused")
        self.assertEqual(r.status, "not_found")
        self.assertIn("WaveSound", r.summary)


class PolicyBoundaryTests(unittest.TestCase):
    def test_day_30_ok_day_31_not(self):
        self.assertIs(evaluate("delivered", 30, False, "unused", True)[0], True)
        self.assertEqual(evaluate("delivered", 31, False, "unused", True)[1], "window_expired")

    def test_damaged_checked_before_window(self):
        self.assertEqual(evaluate("delivered", 90, False, "damaged", True)[0], "review")


class RegistryTests(unittest.TestCase):
    def test_every_tool_has_valid_schema(self):
        self.assertEqual({s["name"] for s in tool_schemas()},
                         {"search_products", "lookup_order", "check_return_eligibility", "customer_profile"})
        for s in tool_schemas():
            self.assertTrue(s["name"] and s["description"])
            self.assertEqual(s["parameters"]["type"], "object")
            for req in s["parameters"]["required"]:
                self.assertIn(req, s["parameters"]["properties"], f"{s['name']}.{req}")


if __name__ == "__main__":
    unittest.main()
