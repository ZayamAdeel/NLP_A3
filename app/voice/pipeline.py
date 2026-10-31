"""
Voice pipeline for ONE WebSocket connection.

   mic PCM16 16 kHz --feed--> Moonshine stream --partial/final--> normalise transcript
        final --> ConversationManager.handle_user_message(voice=True)   <- the SAME path as typed text
                    tokens --> SpeechChunker --> speakable() --> Piper --> PCM chunks --> client
   (text events stage/token/done are forwarded unchanged, so the chat window works as in text mode)

Half-duplex: after a final transcript the server ignores mic audio until the client says
"resume" (playback finished) so the assistant cannot hear itself. A safety timer resumes
anyway if the client never does.

Per-turn latency stages are reported in a final `voice_metrics` event:
  asr_ms                  last voiced audio frame -> final transcript
  pre_generation_ms       retrieval and tool routing run concurrently; the slower one
  llm_ttft_ms             LLM request -> first token (includes prompt processing)
  first_sentence_ms       final transcript -> first speakable chunk ready
  tts_first_audio_ms      first chunk handed to TTS -> first audio frame sent
  first_audio_after_speech_end_ms   THE headline number: last voiced frame -> first audio frame
"""
import asyncio
import logging
import struct
import time

import numpy as np

from app.llm_engine import LLMEngineError
from app.tools.normalize import normalize_transcript, speakable
from app.voice.asr import ASRError
from app.voice.chunker import SpeechChunker
from app.voice.tts import TTSError

logger = logging.getLogger("ecommerce-assistant.voice")

VOICED_RMS = 0.012          # ~ -38 dBFS: simple energy gate used ONLY to timestamp end of speech
RESUME_SAFETY_SECONDS = 2.0


def audio_frame(seq: int, sample_rate: int, pcm: bytes) -> bytes:
    """Binary frame sent to the browser: uint32 seq, uint32 sample rate (little endian) + PCM16."""
    return struct.pack("<II", seq, sample_rate) + pcm


class VoiceConnection:
    def __init__(self, asr, tts, manager, session, send_json, send_bytes):
        self.asr, self.tts, self.manager, self.session = asr, tts, manager, session
        self.send_json, self.send_bytes = send_json, send_bytes
        self.listening, self.last_voiced = True, None
        self._loop = asyncio.get_running_loop()
        self._events: asyncio.Queue = asyncio.Queue()
        self._stream, self._pump, self._turn, self._resume_timer = None, None, None, None
        self._feed_failed = False

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        self._stream = self.asr.new_stream(self._emit)
        self._pump = asyncio.create_task(self._pump_events())

    def _emit(self, kind: str, text: str):            # called from ASR thread(s): must be thread-safe
        self._loop.call_soon_threadsafe(self._events.put_nowait, (kind, text, time.perf_counter()))

    def resume(self):
        if self._resume_timer:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.listening, self.last_voiced = True, None

    async def close(self):
        for t in (self._pump, self._turn):
            if t and not t.done():
                t.cancel()
        if self._resume_timer:
            self._resume_timer.cancel()
        if self._stream:
            await asyncio.to_thread(self._stream.close)

    # -------------------------------------------------------------------- input
    async def on_audio(self, pcm: bytes):
        if not self.listening or len(pcm) < 2:
            return                                       # half-duplex: drop mic audio during a turn
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        if x.size and float(np.sqrt(np.mean(x * x))) > VOICED_RMS:
            self.last_voiced = time.perf_counter()
        try:
            await asyncio.to_thread(self._stream.feed, pcm)
        except Exception as e:  # noqa: BLE001
            logger.warning("ASR feed failed: %s", e)
            if not self._feed_failed:                    # report once, not per 100 ms frame
                self._feed_failed = True
                await self.send_json({"type": "error", "message": f"Speech recognition error: {e}"})

    async def _pump_events(self):
        while True:
            kind, text, t_event = await self._events.get()
            text = text.strip()
            if not self.listening or not text:
                continue
            clean = normalize_transcript(text)
            if kind == "partial":
                await self.send_json({"type": "partial", "text": clean})
            elif kind == "final":
                self.listening = False
                await self.send_json({"type": "final", "text": clean})
                self._turn = asyncio.create_task(self._run_turn(clean, t_event, self.last_voiced))

    # --------------------------------------------------------------------- turn
    async def _run_turn(self, text: str, t_final: float, t_voiced: float | None):
        m = {"asr_ms": _ms(t_voiced, t_final)}
        sentences: asyncio.Queue = asyncio.Queue()
        stamps = {"first_sentence": None, "first_audio": None, "audio_seconds": 0.0}
        speaker = asyncio.create_task(self._speak(sentences, stamps))
        chunker, done = SpeechChunker(), {}

        def enqueue(parts):
            for p in parts:
                if stamps["first_sentence"] is None:
                    stamps["first_sentence"] = time.perf_counter()
                sentences.put_nowait(p)

        try:
            async for kind, payload in self.manager.handle_user_message(self.session, text, voice=True):
                if kind == "stage":
                    await self.send_json({"type": "stage", "stage": payload})
                elif kind == "token":
                    await self.send_json({"type": "token", "text": payload})
                    enqueue(chunker.feed(payload))
                elif kind == "blocked":
                    await self.send_json({"type": "blocked", "text": payload})
                    enqueue([payload])
                elif kind == "done":
                    done = payload
                    await self.send_json({"type": "done", **payload})
            enqueue(chunker.flush())
        except LLMEngineError as e:
            await self.send_json({"type": "error", "message": f"The local model backend hit an error: {e}"})
        except asyncio.CancelledError:
            speaker.cancel()
            raise
        except Exception:  # noqa: BLE001 - a failed turn must not kill the connection
            logger.exception("voice turn failed")
            await self.send_json({"type": "error", "message": "Something went wrong while answering. Please try again."})
        finally:
            sentences.put_nowait(None)

        try:
            await speaker
        finally:
            timing = done.get("timing", {})
            ttft = done.get("time_to_first_token")
            fa, fs = stamps["first_audio"], stamps["first_sentence"]
            m.update({
                "retrieval_ms": timing.get("retrieval_ms"), "tools_ms": timing.get("tools_ms"),
                "router_ms": timing.get("router_ms"),
                "pre_generation_ms": max(timing.get("retrieval_ms") or 0, timing.get("tools_ms") or 0),
                "llm_ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                "first_sentence_ms": _ms(t_final, fs),
                "tts_first_audio_ms": _ms(fs, fa),
                "first_audio_after_final_ms": _ms(t_final, fa),
                "first_audio_after_speech_end_ms": _ms(t_voiced, fa),
                "audio_seconds": round(stamps["audio_seconds"], 2),
            })
            await self.send_json({"type": "tts_done"})
            await self.send_json({"type": "voice_metrics", **m})
            # the client sends "resume" when playback ends; this timer is only a safety net
            self._resume_timer = self._loop.call_later(
                stamps["audio_seconds"] + RESUME_SAFETY_SECONDS, self.resume)

    async def say(self, text: str):
        """Speak a fixed text (e.g. the returning-customer greeting) with the mic closed meanwhile."""
        self.listening = False
        sentences: asyncio.Queue = asyncio.Queue()
        stamps = {"first_sentence": None, "first_audio": None, "audio_seconds": 0.0}
        chunker = SpeechChunker()
        for part in chunker.feed(text + " ") + chunker.flush():
            sentences.put_nowait(part)
        sentences.put_nowait(None)
        await self._speak(sentences, stamps)
        await self.send_json({"type": "tts_done"})
        self._resume_timer = self._loop.call_later(stamps["audio_seconds"] + RESUME_SAFETY_SECONDS, self.resume)

    async def _speak(self, sentences: asyncio.Queue, stamps: dict):
        seq, reported = 0, False
        while (sentence := await sentences.get()) is not None:
            text = speakable(sentence)
            if not text.strip(" .,;:!?-"):
                continue
            try:
                async for pcm in self.tts.synthesize(text):
                    if stamps["first_audio"] is None:
                        stamps["first_audio"] = time.perf_counter()
                    await self.send_bytes(audio_frame(seq, self.tts.sample_rate, pcm))
                    stamps["audio_seconds"] += len(pcm) / 2 / self.tts.sample_rate
                    seq += 1
            except (TTSError, Exception) as e:  # noqa: BLE001 - keep draining so the turn can finish
                logger.warning("TTS failed: %s", e)
                if not reported:
                    reported = True
                    await self.send_json({"type": "error", "message": "Speech synthesis failed; showing text only."})


def _ms(a: float | None, b: float | None) -> float | None:
    return round(max(0.0, b - a) * 1000, 1) if a is not None and b is not None else None
