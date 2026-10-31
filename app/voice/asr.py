"""
Local streaming speech-to-text with Moonshine Voice (https://github.com/moonshine-ai/moonshine,
`pip install moonshine-voice`, MIT licence).

One Transcriber (the model) is shared; every WebSocket connection gets its own stream.
Moonshine segments speech itself and reports three events per spoken "line":
started / text changed (-> our `partial`) and completed (-> our `final`, i.e. end of speech).
Listener callbacks may fire on another thread, so `emit(kind, text)` MUST be thread-safe;
the voice pipeline passes a loop.call_soon_threadsafe-based emitter.
"""
import logging
from typing import Callable

import numpy as np

from app.config import ASR_LANGUAGE, ASR_MODEL_ARCH, ASR_UPDATE_INTERVAL, SPEECH_SAMPLE_RATE

logger = logging.getLogger("ecommerce-assistant.asr")


class ASRError(Exception):
    pass


class ASRStream:
    def __init__(self, transcriber, update_interval: float, emit: Callable[[str, str], None]):
        from moonshine_voice import TranscriptEventListener

        class _Listener(TranscriptEventListener):
            def on_line_started(self, event):
                emit("partial", event.line.text)

            def on_line_text_changed(self, event):
                emit("partial", event.line.text)

            def on_line_completed(self, event):
                emit("final", event.line.text)

        self._stream = transcriber.create_stream(update_interval=update_interval)
        self._stream.add_listener(_Listener())
        self._stream.start()

    def feed(self, pcm16: bytes):
        """16 kHz mono little-endian int16 audio."""
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.add_audio(samples.tolist(), SPEECH_SAMPLE_RATE)

    def close(self):
        for name in ("stop", "close"):
            try:
                getattr(self._stream, name)()
            except Exception:  # noqa: BLE001 - closing must never raise
                pass


class MoonshineASR:
    def __init__(self, language: str = ASR_LANGUAGE, arch: str = ASR_MODEL_ARCH,
                 update_interval: float = ASR_UPDATE_INTERVAL):
        self.language, self.arch, self.update_interval = language, arch, update_interval
        self.transcriber = None

    @property
    def ready(self) -> bool:
        return self.transcriber is not None

    def load(self):
        """Downloads the model on first use (needs internet once), then runs offline."""
        from moonshine_voice import Transcriber, get_model_for_language, string_to_model_arch
        path, arch = get_model_for_language(self.language, string_to_model_arch(self.arch))
        try:
            self.transcriber = Transcriber(model_path=path, model_arch=arch)
        except TypeError:
            self.transcriber = Transcriber(path, arch)

    def new_stream(self, emit: Callable[[str, str], None]) -> ASRStream:
        if not self.ready:
            raise ASRError("ASR model not loaded")
        return ASRStream(self.transcriber, self.update_interval, emit)
