"""Small audio helpers shared by the smoke test, sample generator and benchmark."""
import wave

import numpy as np


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int = 16000) -> bytes:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if src_rate == dst_rate or x.size == 0:
        return pcm
    n = int(round(x.size * dst_rate / src_rate))
    y = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
    return y.astype("<i2").tobytes()


def write_wav(path, pcm: bytes, rate: int = 16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def read_wav_16k(path) -> bytes:
    with wave.open(str(path), "rb") as w:
        pcm, rate = w.readframes(w.getnframes()), w.getframerate()
        if w.getnchannels() != 1:
            raise ValueError("expected a mono WAV file")
    return resample_pcm16(pcm, rate, 16000)
