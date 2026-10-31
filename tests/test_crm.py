"""CRM bonus: remembers customers across sessions; LLM-callable; works in voice mode."""
import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.conversation_manager import ConversationManager  # noqa: E402
from app.domain import GREETING_MESSAGE  # noqa: E402
from app.orchestrator import ToolOrchestrator  # noqa: E402
from app.rag.retriever import RetrievalResult  # noqa: E402
from app.tools import crm  # noqa: E402
from app.tools.db import ensure_db  # noqa: E402
from fakes import FakeASR, FakeTTS, RecordingEngine, RouterEngine, StaticRetriever, silence, tone  # noqa: E402

SCRIPT = {
    "prefer black headphones": {"tool": "customer_profile", "action": "update", "name": "Ayesha", "preference": "black headphones"},
    "my name is ayesha": {"tool": "customer_profile", "action": "update", "name": "Zed"},   # model invents a different name
    "i prefer red": {"tool": "customer_profile", "action": "update", "preference": "blue shoes"},   # not what they said
    "what do you remember": {"tool": "customer_profile", "action": "get"},
    "forget my details": {"tool": "customer_profile", "action": "forget"},
    "wireless headphones": {"tool": "search_products", "query": "wireless headphones", "max_price": 10000},
}


class CRMCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = ensure_db(self.tmp / "t.db", force=True)
        self.answer = RecordingEngine(reply="Nice to meet you.")
        orch = ToolOrchestrator(RouterEngine(SCRIPT), db_path=self.db, today=lambda: date(2026, 10, 24))
        self.mgr = ConversationManager(engine=self.answer, retriever=StaticRetriever(RetrievalResult("no_match")),
                                       orchestrator=orch, crm_db=self.db)

    async def turn(self, session, text):
        events = [e async for e in self.mgr.handle_user_message(session, text)]
        return next(p for t, p in events if t == "done")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = ensure_db(self.tmp / "t.db", force=True)

    def test_roundtrip_dedupe_and_forget(self):
        crm.save_fields(self.db, "u1", name="Ayesha", preference="Black headphones")
        crm.save_fields(self.db, "u1", contact="03001234567", preference="black headphones")   # duplicate ignored
        p = crm.load_profile(self.db, "u1")
        self.assertEqual((p["name"], p["contact"], p["preferences"]), ("Ayesha", "03001234567", ["Black headphones"]))
        crm.forget(self.db, "u1")
        self.assertIsNone(crm.load_profile(self.db, "u1"))

    def test_interactions_are_capped_and_ordered(self):
        for i in range(15):
            crm.record_interaction(self.db, "u2", f"topic {i}")
        p = crm.load_profile(self.db, "u2")
        self.assertEqual(len(p["interactions"]), crm.MAX_INTERACTIONS)
        self.assertEqual(crm.last_topic(p), "topic 14")

    def test_profiles_are_isolated_per_user(self):
        crm.save_fields(self.db, "a", name="Ayesha")
        self.assertIsNone(crm.load_profile(self.db, "b"))

    def test_greeting_variants(self):
        self.assertIsNone(crm.greeting_for(None))
        self.assertIsNone(crm.greeting_for({"name": None, "interactions": []}))
        self.assertEqual(crm.greeting_for({"name": "Ayesha", "interactions": []}),
                         "Welcome back, Ayesha! How can I help you today?")
        self.assertIn("Last time you checked order BC1024.",
                      crm.greeting_for({"name": "Ayesha", "interactions": [{"topic": "checked order BC1024"}]}))
        self.assertTrue(crm.greeting_for({"name": None, "interactions": [{"topic": "looked for mice"}]}).startswith("Welcome back!"))

    def test_name_extraction_is_strict(self):
        n = crm.name_from_text
        self.assertEqual(n("my name is ayesha khan and I want headphones"), "Ayesha Khan")
        self.assertEqual(n("My name is Bilal and I need help"), "Bilal")
        self.assertEqual(n("call me Sana please"), "Sana")
        self.assertIsNone(n("I'm looking for a gift"))
        self.assertIsNone(n("name of the product"))

    def test_contact_and_action(self):
        self.assertEqual(crm.contact_from_text("email is a.b@example.com"), "a.b@example.com")
        self.assertEqual(crm.contact_from_text("call 03001234567"), "03001234567")
        self.assertIsNone(crm.contact_from_text("order BC1024 code 4721"))
        self.assertEqual(crm.action_from_text("please forget my details"), "forget")
        self.assertEqual(crm.action_from_text("what do you remember about me"), "get")
        self.assertEqual(crm.action_from_text("my name is Ayesha"), "update")


class CrossSessionTests(CRMCase):
    async def test_llm_calls_tool_then_returning_session_greets_by_name(self):
        s1 = self.mgr.create_session("u1")
        self.assertEqual(self.mgr.greeting(s1), GREETING_MESSAGE)                  # first visit: standard greeting
        done = await self.turn(s1, "my name is Ayesha and I prefer black headphones")
        self.assertEqual(done["tools"][0]["name"], "customer_profile")
        self.assertIn("Saved to the customer's profile: name Ayesha; preference black headphones",
                      self.answer.prompts[-1][-1]["content"])
        s2 = self.mgr.create_session("u1")                                         # NEW session, same customer
        self.assertEqual(self.mgr.greeting(s2), "Welcome back, Ayesha! How can I help you today?")
        self.assertEqual(s2.profile["preferences"], ["black headphones"])

    async def test_profile_reaches_the_prompt_in_the_next_session(self):
        crm.save_fields(self.db, "u1", name="Ayesha", preference="black headphones")
        s = self.mgr.create_session("u1")
        await self.turn(s, "how many days do I have to return an item")
        system = " ".join(m["content"] for m in self.answer.prompts[-1] if m["role"] == "system")
        self.assertIn("Returning customer", system)
        self.assertIn("Ayesha", system)

    async def test_last_topic_is_recalled_next_time(self):
        s1 = self.mgr.create_session("u1")
        await self.turn(s1, "find wireless headphones under 10000 rupees")
        s2 = self.mgr.create_session("u1")
        self.assertIn("Last time you looked for wireless headphones.", self.mgr.greeting(s2))

    async def test_customer_can_ask_what_is_remembered(self):
        crm.save_fields(self.db, "u1", name="Ayesha", preference="black headphones")
        s = self.mgr.create_session("u1")
        await self.turn(s, "what do you remember about me")
        self.assertIn("Saved profile: name Ayesha", self.answer.prompts[-1][-1]["content"])

    async def test_forget_deletes_everything(self):
        crm.save_fields(self.db, "u1", name="Ayesha")
        s = self.mgr.create_session("u1")
        await self.turn(s, "forget my details please")
        self.assertIsNone(crm.load_profile(self.db, "u1"))
        self.assertEqual(self.mgr.greeting(self.mgr.create_session("u1")), GREETING_MESSAGE)

    async def test_invented_name_and_preference_are_not_saved(self):
        s = self.mgr.create_session("u1")
        await self.turn(s, "my name is Ayesha")                                    # model said "Zed": text says Ayesha
        self.assertEqual(crm.load_profile(self.db, "u1")["name"], "Ayesha")
        self.mgr.orchestrator.engine = RouterEngine({"i prefer red": SCRIPT["i prefer red"]})
        await self.turn(s, "i prefer red ones")                                    # model invented "blue shoes"
        self.assertEqual(crm.load_profile(self.db, "u1")["preferences"], [])

    async def test_anonymous_session_saves_nothing(self):
        s = self.mgr.create_session()                                              # no user_id
        await self.turn(s, "my name is Ayesha and I prefer black headphones")
        self.assertIn("anonymous session", self.answer.prompts[-1][-1]["content"])

    async def test_crm_disabled_means_no_profile_lookup(self):
        mgr = ConversationManager(engine=self.answer, retriever=StaticRetriever(RetrievalResult("no_match")))
        s = mgr.create_session("u1")
        self.assertIsNone(s.user_id)
        self.assertEqual(mgr.greeting(s), GREETING_MESSAGE)

    async def test_broken_crm_database_never_blocks_a_chat(self):
        mgr = ConversationManager(engine=self.answer, crm_db=self.tmp / "no" / "such.db")
        s = mgr.create_session("u1")
        self.assertIsNone(s.profile)
        self.assertEqual(mgr.greeting(s), GREETING_MESSAGE)


class VoiceCRMTests(CRMCase):
    async def test_returning_customer_is_greeted_by_voice_and_can_update_by_voice(self):
        from test_main_voice import FakeSocket
        from app import main
        crm.save_fields(self.db, "u1", name="Ayesha")
        crm.record_interaction(self.db, "u1", "checked order BC1024")
        saved = {k: getattr(main, k) for k in ("manager", "asr", "tts")}
        self.addCleanup(lambda: [setattr(main, k, v) for k, v in saved.items()])
        main.manager, main.asr, main.tts = self.mgr, FakeASR("my name is Ayesha and I prefer black headphones"), FakeTTS()
        sid = self.mgr.create_session("u1").session_id

        frames = [tone() for _ in range(4)] + [silence() for _ in range(3)]
        # like a real client: wait for the greeting to finish, send "resume", then speak
        ws = FakeSocket([json.dumps({"type": "start", "session_id": sid}), 0.2, json.dumps({"type": "resume"}), *frames],
                        until="voice_metrics")
        await main.ws_voice(ws)
        ready = next(m for m in ws.json if m["type"] == "ready")
        self.assertTrue(ready["greeting"])
        spoken = " ".join(main.tts.texts)
        self.assertIn("Welcome back, Ayesha!", spoken)
        self.assertIn("Last time you checked order B C 1 0 2 4.", spoken)        # spelled out for the TTS engine
        done = next(m for m in ws.json if m["type"] == "done")                      # the spoken update used the tool
        self.assertEqual(done["tools"][0]["name"], "customer_profile")
        self.assertIn("black headphones", crm.load_profile(self.db, "u1")["preferences"])


if __name__ == "__main__":
    unittest.main()
