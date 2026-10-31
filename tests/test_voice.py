"""Voice pipeline tests with fake ASR/TTS: streaming order, half-duplex, failures, concurrency."""
import asyncio
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

from app.conversation_manager import ConversationManager  # noqa: E402
from app.orchestrator import ToolOrchestrator  # noqa: E402
from app.rag.retriever import Hit, RetrievalResult  # noqa: E402
from app.tools.db import ensure_db  # noqa: E402
from app.voice.chunker import SpeechChunker  # noqa: E402
from app.voice.pipeline import VoiceConnection  # noqa: E402
from app.voice.tts import TTSError  # noqa: E402
from fakes import (FakeASR, FakeTTS, RecordingEngine, RouterEngine, StaticRetriever,  # noqa: E402
                   silence, tone)

HIT = Hit("returns/returns_033_eligibility.md#0", "returns/returns_033_eligibility.md", "Return Eligibility Policy",
          "returns", "Return Eligibility Policy\nItems may be returned within 30 days.", 0.8)


class Rig:
    """One voice connection wired to a real manager, with fake models and a recording socket."""

    def __init__(self, tmp, reply="You have thirty days. The item must be unused.", asr=None, tts=None,
                 token_delay=0.0, router=None):
        db = ensure_db(Path(tmp) / "t.db", force=True)
        self.engine = RecordingEngine(reply=reply, token_delay=token_delay)
        orch = ToolOrchestrator(router or RouterEngine({"order": {"tool": "lookup_order"}}), db_path=db,
                                today=lambda: date(2026, 10, 24))
        self.manager = ConversationManager(engine=self.engine, retriever=StaticRetriever(RetrievalResult("ok", [HIT], ms=3.0)),
                                           orchestrator=orch)
        self.session = self.manager.create_session()
        self.asr, self.tts = asr or FakeASR(), tts or FakeTTS()
        self.log = []                                    # ordered: ("json", obj) | ("audio", seq, rate, nbytes)
        self.conn = VoiceConnection(self.asr, self.tts, self.manager, self.session, self._json, self._bytes)
        self.conn.start()

    async def _json(self, obj):
        self.log.append(("json", obj))

    async def _bytes(self, data):
        seq, rate = struct.unpack("<II", data[:8])
        self.log.append(("audio", seq, rate, len(data) - 8))

    def types(self):
        return [e[1]["type"] if e[0] == "json" else "audio" for e in self.log]

    def json_of(self, t):
        return [e[1] for e in self.log if e[0] == "json" and e[1]["type"] == t]

    async def speak(self, silence_chunks=3):
        for _ in range(4):
            await self.conn.on_audio(tone())
        for _ in range(silence_chunks):
            await self.conn.on_audio(silence())

    async def until(self, event_type, timeout=3.0):
        async def wait():
            while event_type not in self.types():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait(), timeout)


class VoiceCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    async def asyncTearDown(self):
        for r in getattr(self, "rigs", []):
            await r.conn.close()

    def rig(self, **kw):
        r = Rig(self.tmp, **kw)
        self.rigs = getattr(self, "rigs", []) + [r]
        return r


class FlowTests(VoiceCase):
    async def test_full_spoken_turn(self):
        r = self.rig()
        await r.speak()
        await r.until("voice_metrics")
        types = r.types()
        for expected in ("partial", "final", "stage", "token", "done", "audio", "tts_done", "voice_metrics"):
            self.assertIn(expected, types)
        self.assertEqual(r.json_of("final")[0]["text"], "how many days do I have to send something back")
        self.assertEqual(types.index("partial") < types.index("final") < types.index("stage"), True)
        audio = [e for e in r.log if e[0] == "audio"]
        self.assertEqual([a[1] for a in audio], list(range(len(audio))))      # in-order sequence numbers
        self.assertEqual({a[2] for a in audio}, {22050})
        self.assertEqual(r.tts.texts, ["You have thirty days.", "The item must be unused."])

    async def test_audio_starts_before_the_llm_has_finished(self):
        """The Phase II requirement: speak sentence by sentence while tokens still stream."""
        r = self.rig(reply="First sentence is here. Second sentence follows now. Third one ends it.",
                     token_delay=0.05)
        await r.speak()
        await r.until("voice_metrics")
        types = r.types()
        self.assertLess(types.index("audio"), types.index("done"))

    async def test_same_manager_path_as_text(self):
        r = self.rig()
        await r.speak()
        await r.until("voice_metrics")
        self.assertIn("spoken aloud", r.engine.prompts[-1][-1]["content"])      # voice style note
        self.assertIn("Knowledge base excerpts", r.engine.prompts[-1][-1]["content"])  # RAG ran
        self.assertEqual([t.role for t in r.session.history], ["user", "assistant"])   # shared history

    async def test_metrics_contain_every_stage(self):
        r = self.rig()
        await r.speak()
        await r.until("voice_metrics")
        m = r.json_of("voice_metrics")[0]
        for key in ("asr_ms", "retrieval_ms", "tools_ms", "pre_generation_ms", "llm_ttft_ms", "first_sentence_ms",
                    "tts_first_audio_ms", "first_audio_after_final_ms", "first_audio_after_speech_end_ms"):
            self.assertIn(key, m)
            self.assertIsNotNone(m[key], key)
        self.assertGreaterEqual(m["first_audio_after_speech_end_ms"], m["first_audio_after_final_ms"])
        self.assertGreater(m["audio_seconds"], 0)

    async def test_transcript_is_normalised_for_ids(self):
        r = self.rig(asr=FakeASR("where is order B C one zero two four"))
        await r.speak()
        await r.until("voice_metrics")
        self.assertEqual(r.json_of("final")[0]["text"], "where is order BC1024")

    async def test_speech_text_is_cleaned_for_tts(self):
        r = self.rig(reply="Your order BC1024 is on its way (see [1]). It costs Rs 9,500.")
        await r.speak()
        await r.until("voice_metrics")
        spoken = " ".join(r.tts.texts)
        self.assertNotIn("[1]", spoken)
        self.assertIn("B C 1 0 2 4", spoken)
        self.assertIn("9,500 rupees", spoken)


class ToolByVoiceTests(VoiceCase):
    async def test_clarifying_question_is_spoken_without_llm(self):
        r = self.rig(asr=FakeASR("what's happening with order BC1024"))
        await r.speak()
        await r.until("voice_metrics")
        self.assertEqual(r.engine.prompts, [])                                    # no generation call
        self.assertIn("last four digits", " ".join(r.tts.texts))                  # but it was spoken
        self.assertEqual(r.json_of("done")[0]["timing"]["asked"], True)


class HalfDuplexTests(VoiceCase):
    async def test_mic_audio_ignored_during_turn_until_resume(self):
        r = self.rig()
        await r.speak()
        await r.until("final")
        fed_before = r.asr.streams[0].loud
        for _ in range(5):
            await r.conn.on_audio(tone())                                          # echo / user talking over it
        self.assertEqual(r.asr.streams[0].loud, fed_before)
        await r.until("voice_metrics")
        r.conn.resume()
        self.assertTrue(r.conn.listening)
        await r.conn.on_audio(tone())
        self.assertEqual(r.asr.streams[0].loud, fed_before + 1)

    async def test_second_turn_after_resume(self):
        r = self.rig()
        await r.speak()
        await r.until("voice_metrics")
        r.conn.resume()
        r.asr.streams[0].loud, r.asr.streams[0].finished = 0, False                # fake ASR: new utterance
        r.log.clear()
        await r.speak()
        await r.until("voice_metrics")
        self.assertEqual(len(r.manager.get_session(r.session.session_id).history), 4)

    async def test_auto_resume_safety_net(self):
        r = self.rig()
        await r.speak()
        await r.until("voice_metrics")
        self.assertIsNotNone(r.conn._resume_timer)                                 # armed even without a client "resume"


class FailureTests(VoiceCase):
    async def test_tts_failure_reports_error_but_text_and_turn_complete(self):
        r = self.rig(tts=FakeTTS(fail=TTSError("voice crashed")))
        await r.speak()
        await r.until("voice_metrics")
        self.assertIn("done", r.types())
        self.assertEqual(len(r.json_of("error")), 1)                               # once, not per sentence
        self.assertIn("showing text only", r.json_of("error")[0]["message"])
        self.assertNotIn("audio", r.types())

    async def test_asr_failure_is_reported_once_not_per_frame(self):
        r = self.rig(asr=FakeASR(fail=RuntimeError("model blew up")))
        for _ in range(6):
            await r.conn.on_audio(tone())
        self.assertEqual(len(r.json_of("error")), 1)

    async def test_llm_failure_is_reported_and_connection_survives(self):
        from app.llm_engine import LLMEngineError
        r = self.rig()
        r.engine.fail = LLMEngineError("ollama down")
        await r.speak()
        await r.until("voice_metrics")
        self.assertIn("ollama down", r.json_of("error")[0]["message"])
        r.engine.fail = None
        r.conn.resume()
        r.asr.streams[0].loud, r.asr.streams[0].finished = 0, False
        r.log.clear()
        await r.speak()
        await r.until("done")

    async def test_off_topic_redirect_is_spoken(self):
        r = self.rig(asr=FakeASR("what is the weather today"))
        await r.speak()
        await r.until("voice_metrics")
        self.assertIn("blocked", r.types())
        self.assertIn("order support assistant", " ".join(r.tts.texts))


class ConcurrencyTests(VoiceCase):
    async def test_two_users_do_not_block_each_other(self):
        a = self.rig(tts=FakeTTS(delay=0.15), token_delay=0.02)
        b = self.rig(tts=FakeTTS(delay=0.15), token_delay=0.02)
        t0 = asyncio.get_running_loop().time()
        await asyncio.gather(a.speak(), b.speak())
        await asyncio.gather(a.until("voice_metrics"), b.until("voice_metrics"))
        elapsed = asyncio.get_running_loop().time() - t0
        for r in (a, b):
            self.assertTrue(any(e[0] == "audio" for e in r.log))
            self.assertEqual(len(r.session.history), 2)
        self.assertNotEqual(a.session.session_id, b.session.session_id)
        self.assertLess(elapsed, 1.5)                    # serial would be roughly twice a single turn


class ChunkerTests(unittest.TestCase):
    def run_chunker(self, text, step=3):
        c, out = SpeechChunker(), []
        for i in range(0, len(text), step):
            out += c.feed(text[i:i + step])
        return out + c.flush()

    def test_first_chunk_flushes_at_clause(self):
        self.assertEqual(self.run_chunker("The WaveSound H200 is available at 9,500 rupees, and it is in stock."),
                         ["The WaveSound H200 is available at 9,500 rupees,", "and it is in stock."])

    def test_abbreviation_and_decimal_not_sentence_end(self):
        self.assertEqual(self.run_chunker("It costs Rs. 9,500 today. Done."), ["It costs Rs. 9,500 today.", "Done."])
        self.assertEqual(self.run_chunker("Delivery takes 3.5 days. Done."), ["Delivery takes 3.5 days.", "Done."])

    def test_overlong_text_is_cut(self):
        out = self.run_chunker("word " * 100)
        self.assertTrue(all(len(c) <= 165 for c in out))
        self.assertEqual(" ".join(out).split(), ["word"] * 100)


if __name__ == "__main__":
    unittest.main()
