"""Conversation manager + tools integration (text and voice share one code path)."""
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.config import VOICE_NUM_PREDICT  # noqa: E402
from app.conversation_manager import ConversationManager  # noqa: E402
from app.orchestrator import ToolOrchestrator  # noqa: E402
from app.rag.retriever import Hit, RetrievalResult  # noqa: E402
from app.tools.db import ensure_db  # noqa: E402
from fakes import RecordingEngine, RouterEngine, StaticRetriever  # noqa: E402

SCRIPT = {
    "wireless headphones": {"tool": "search_products", "query": "wireless headphones", "max_price": 10000},
    "happening with order": {"tool": "lookup_order", "order_id": "BC1024"},
}
HIT = Hit("returns/returns_033_eligibility.md#0", "returns/returns_033_eligibility.md", "Return Eligibility Policy",
          "returns", "Return Eligibility Policy\nItems may be returned within 30 days.", 0.8)


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        db = ensure_db(self.tmp / "t.db", force=True)
        self.answer = RecordingEngine(reply="Here you go.")
        self.router = RouterEngine(SCRIPT)
        orch = ToolOrchestrator(self.router, db_path=db, today=lambda: date(2026, 10, 24))
        self.retr = StaticRetriever(RetrievalResult("ok", [HIT], ms=2.0))
        self.mgr = ConversationManager(engine=self.answer, retriever=self.retr, orchestrator=orch)
        self.s = self.mgr.create_session()

    async def turn(self, text, **kw):
        events = [e async for e in self.mgr.handle_user_message(self.s, text, **kw)]
        return events, next((p for t, p in events if t == "done"), None)


class ToolTurnTests(Base):
    async def test_result_turn_uses_tool_text_and_skips_excerpts(self):
        _, done = await self.turn("find wireless headphones under 10000 rupees")
        prompt = self.answer.prompts[-1][-1]["content"]
        self.assertIn("Tool search_products returned", prompt)
        self.assertIn("WaveSound H200", prompt)
        self.assertNotIn("Knowledge base excerpts", prompt)
        self.assertEqual(done["tools"][0]["name"], "search_products")
        self.assertEqual(done["sources"][0]["doc_id"], "products/product_001_wavesound_headphones.md")
        self.assertGreater(done["timing"]["router_ms"], -1)

    async def test_ask_turn_streams_question_without_calling_the_llm(self):
        events, done = await self.turn("what's happening with order BC1024")
        text = "".join(p for t, p in events if t == "token")
        self.assertIn("last four digits", text)
        self.assertEqual(self.answer.prompts, [])                   # no generation call at all
        self.assertTrue(done["timing"]["asked"])
        self.assertEqual([t.role for t in self.s.history], ["user", "assistant"])   # still recorded

    async def test_followup_answer_completes_the_lookup(self):
        await self.turn("what's happening with order BC1024")
        _, done = await self.turn("4721")
        self.assertIn("status shipped", self.answer.prompts[-1][-1]["content"])
        self.assertEqual(done["tools"][0]["status"], "ok")
        self.assertIn("BC1024", self.s.verified_orders)

    async def test_non_tool_turn_still_uses_rag(self):
        _, done = await self.turn("how many days do I have to send something back")
        self.assertIn("Knowledge base excerpts", self.answer.prompts[-1][-1]["content"])
        self.assertEqual(done["tools"], [])
        self.assertEqual(done["sources"][0]["n"], 1)

    async def test_text_events_keep_assignment1_vocabulary(self):
        events, _ = await self.turn("find wireless headphones under 10000 rupees")
        self.assertLessEqual({t for t, _ in events}, {"stage", "token", "done"})

    async def test_router_failure_falls_back_to_rag_answer(self):
        self.mgr.orchestrator.engine = RouterEngine(raw="garbage")
        _, done = await self.turn("find wireless headphones under 10000 rupees")
        self.assertEqual(done["tools"], [])
        self.assertTrue(self.answer.prompts)                         # still answered

    async def test_no_orchestrator_means_pure_assignment2_behaviour(self):
        self.mgr.orchestrator = None
        _, done = await self.turn("find wireless headphones under 10000 rupees")
        self.assertEqual(done["tools"], [])


class VoiceModeTests(Base):
    async def test_voice_flag_shortens_style_and_caps_length(self):
        captured = {}
        orig = self.answer.stream_chat

        async def spy(messages, options=None):
            captured["options"] = options
            async for x in orig(messages, options):
                yield x
        self.answer.stream_chat = spy
        await self.turn("how many days do I have to send something back", voice=True)
        self.assertIn("spoken aloud", self.answer.prompts[-1][-1]["content"])
        self.assertEqual(captured["options"], {"num_predict": VOICE_NUM_PREDICT})

    async def test_text_mode_has_no_voice_note(self):
        await self.turn("how many days do I have to send something back")
        self.assertNotIn("spoken aloud", self.answer.prompts[-1][-1]["content"])

    async def test_voice_and_text_share_history_and_verification(self):
        await self.turn("what's happening with order BC1024", voice=True)
        await self.turn("4721", voice=False)                         # answered by typing
        self.assertEqual(len(self.s.history), 4)
        self.assertIn("BC1024", self.s.verified_orders)


if __name__ == "__main__":
    unittest.main()
