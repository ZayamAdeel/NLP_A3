"""
Drives the real /ws/voice and /ws/chat handlers in app.main with a fake socket and fake models
(works with real FastAPI as well: the route decorators return the original coroutine functions).
"""
import asyncio
import json
import shutil
import struct
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi import WebSocketDisconnect  # noqa: E402

from app import main  # noqa: E402
from app.conversation_manager import ConversationManager  # noqa: E402
from app.orchestrator import ToolOrchestrator  # noqa: E402
from app.rag.retriever import Hit, RetrievalResult  # noqa: E402
from app.tools.db import ensure_db  # noqa: E402
from fakes import FakeASR, FakeTTS, RecordingEngine, RouterEngine, StaticRetriever, silence, tone  # noqa: E402

HIT = Hit("returns/returns_033_eligibility.md#0", "returns/returns_033_eligibility.md", "Return Eligibility Policy",
          "returns", "Return Eligibility Policy\nItems may be returned within 30 days.", 0.8)


class FakeSocket:
    """Plays a script of client messages, then waits for `until` in the server output before disconnecting."""

    def __init__(self, script, until=None, timeout=3.0):
        self.script, self.until, self.timeout = list(script), until, timeout
        self.json, self.audio, self.accepted = [], [], False

    async def accept(self):
        self.accepted = True

    async def receive(self):
        while self.script and isinstance(self.script[0], (int, float)):
            await asyncio.sleep(self.script.pop(0))          # a number in the script = "client waits N seconds"
        if self.script:
            item = self.script.pop(0)
            await asyncio.sleep(0.005)
            return {"type": "websocket.receive", "bytes": item} if isinstance(item, bytes) else \
                   {"type": "websocket.receive", "text": item}
        waited = 0.0
        while self.until and not any(m.get("type") == self.until for m in self.json) and waited < self.timeout:
            await asyncio.sleep(0.01)
            waited += 0.01
        return {"type": "websocket.disconnect"}

    async def receive_text(self):
        m = await self.receive()
        if m["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()
        return m["text"]

    async def send_text(self, text):
        self.json.append(json.loads(text))

    async def send_bytes(self, data):
        self.audio.append(struct.unpack("<II", data[:8]) + (len(data) - 8,))

    def types(self):
        return [m["type"] for m in self.json]


class MainCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        db = ensure_db(self.tmp / "t.db", force=True)
        self.saved = {k: getattr(main, k) for k in ("manager", "asr", "tts", "engine")}
        self.engine = RecordingEngine(reply="You have thirty days. It must be unused.")
        orch = ToolOrchestrator(RouterEngine({"order": {"tool": "lookup_order"}}), db_path=db, today=lambda: date(2026, 10, 24))
        main.manager = ConversationManager(engine=self.engine, retriever=StaticRetriever(RetrievalResult("ok", [HIT], ms=2.0)),
                                           orchestrator=orch)
        main.asr, main.tts = FakeASR(), FakeTTS()
        self.addCleanup(lambda: [setattr(main, k, v) for k, v in self.saved.items()])
        self.sid = main.manager.create_session().session_id

    def start_msg(self, sid=None):
        return json.dumps({"type": "start", "session_id": sid or self.sid})


class VoiceEndpointTests(MainCase):
    async def test_complete_voice_conversation_over_the_socket(self):
        frames = [tone() for _ in range(4)] + [silence() for _ in range(3)]
        ws = FakeSocket([self.start_msg(), *frames], until="voice_metrics")
        await main.ws_voice(ws)
        self.assertTrue(ws.accepted)
        t = ws.types()
        for expected in ("ready", "partial", "final", "stage", "token", "done", "tts_done", "voice_metrics"):
            self.assertIn(expected, t)
        self.assertTrue(ws.audio)
        self.assertEqual(ws.audio[0][:2], (0, 22050))
        done = next(m for m in ws.json if m["type"] == "done")
        self.assertEqual(done["sources"][0]["doc_id"], "returns/returns_033_eligibility.md")   # RAG in voice mode
        self.assertEqual(len(main.manager.get_session(self.sid).history), 2)                    # memory shared

    async def test_text_and_voice_share_one_session(self):
        await main.ws_voice(FakeSocket([self.start_msg(), *[tone() for _ in range(4)], *[silence() for _ in range(3)]],
                                       until="voice_metrics"))
        chat = FakeSocket([json.dumps({"type": "message", "session_id": self.sid, "text": "and exchanges?"})],
                          until="done")
        await main.ws_chat(chat)
        self.assertEqual(len(main.manager.get_session(self.sid).history), 4)
        self.assertIn("done", chat.types())

    async def test_unknown_session_and_bad_messages_do_not_crash(self):
        ws = FakeSocket([self.start_msg("ghost"), "{not json", json.dumps({"type": "bogus"}), tone()])
        await main.ws_voice(ws)
        msgs = [m["message"] for m in ws.json if m["type"] == "error"]
        self.assertEqual(len(msgs), 3)
        self.assertIn("Unknown or expired", msgs[0])
        self.assertIn("Malformed", msgs[1])
        self.assertIn("Unknown message type", msgs[2])

    async def test_voice_models_not_loaded_gives_clear_error(self):
        main.asr.ready = False
        main.voice_state["error"] = "Piper voice not found"
        self.addCleanup(main.voice_state.update, error="")
        ws = FakeSocket([self.start_msg()])
        await main.ws_voice(ws)
        self.assertIn("Voice models are not loaded: Piper voice not found", ws.json[0]["message"])

    async def test_audio_before_start_is_ignored(self):
        ws = FakeSocket([tone(), tone()])
        await main.ws_voice(ws)
        self.assertEqual(ws.json, [])

    async def test_resume_message_is_accepted(self):
        ws = FakeSocket([self.start_msg(), json.dumps({"type": "resume"})])
        await main.ws_voice(ws)
        self.assertEqual(ws.types(), ["ready"])

    async def test_tool_flow_over_voice_socket(self):
        main.asr = FakeASR("what's happening with order BC1024")
        ws = FakeSocket([self.start_msg(), *[tone() for _ in range(4)], *[silence() for _ in range(3)]],
                        until="voice_metrics")
        await main.ws_voice(ws)
        done = next(m for m in ws.json if m["type"] == "done")
        self.assertTrue(done["timing"]["asked"])
        self.assertTrue(ws.audio)                                           # the question was spoken
        self.assertEqual(self.engine.prompts, [])


class HealthAndTextTests(MainCase):
    async def test_health_reports_voice_and_tools(self):
        class E:
            async def health_check(self):
                return True
        main.engine = E()
        main.manager.retriever.ready = True
        h = await main.health()
        self.assertTrue(h.voice_ready)
        self.assertTrue(h.tools_enabled)
        self.assertEqual(h.index_chunks, 50)

    async def test_text_endpoint_contract_unchanged(self):
        ws = FakeSocket([json.dumps({"type": "message", "session_id": self.sid, "text": "how long do I have to return?"}),
                         "{bad json"], until="done")
        await main.ws_chat(ws)
        self.assertLessEqual({t for t in ws.types()} - {"error"}, {"stage", "token", "done"})
        self.assertEqual(sum(t == "error" for t in ws.types()), 1)


if __name__ == "__main__":
    unittest.main()
