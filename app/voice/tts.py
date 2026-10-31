"""
Local text-to-speech with Piper (https://github.com/OHF-Voice/piper1-gpl, `pip install piper-tts`).

`synthesize(text)` is an async generator of raw PCM16 byte chunks. The CPU-bound synthesis
runs in a worker thread (so the event loop, and other users' streams, never block) and a
lock serialises synthesis between users because the phonemiser is not guaranteed
thread-safe; Piper is much faster than real time, so this costs a second user little.
"""
import asyncio
import logging
import threading
from pathlib import Path
from typing import AsyncIterator

from app.config import TTS_VOICE_PATH

logger = logging.getLogger("ecommerce-assistant.tts")


class TTSError(Exception):
    pass


class PiperTTS:
    def __init__(self, voice_path: Path = TTS_VOICE_PATH):
        self.voice_path, self.voice, self.sample_rate = Path(voice_path), None, 22050
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.voice is not None

    def load(self):
        if not self.voice_path.exists():
            raise TTSError(f"Piper voice not found at {self.voice_path}. Download it with: "
                           f"python -m piper.download_voices en_US-lessac-medium --data-dir {self.voice_path.parent}")
        from piper import PiperVoice  # imported lazily: optional dependency
        self.voice = PiperVoice.load(str(self.voice_path))
        self.sample_rate = int(self.voice.config.sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not self.ready:
            raise TTSError("TTS voice not loaded")
        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def work():
            try:
                with self._lock:
                    for chunk in self.voice.synthesize(text):
                        data = getattr(chunk, "audio_int16_bytes", None)
                        if data is None:
                            data = chunk.audio_int16_array.tobytes()
                        loop.call_soon_threadsafe(q.put_nowait, bytes(data))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, work)
        while (item := await q.get()) is not None:
            if isinstance(item, Exception):
                raise TTSError(f"synthesis failed: {item}") from item
            yield item
        await fut
