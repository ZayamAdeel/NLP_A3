"""
Environment smoke test - run this FIRST on your machine:   python scripts/smoke_voice.py

Checks, in order, with a PASS/FAIL and a fix hint for each:
  1. Ollama reachable, chat + embedding models pulled
  2. Piper voice loads and synthesises (reports time to first audio, real-time factor)
  3. Moonshine loads (downloads its model on first run: needs internet once)
  4. Round trip: Piper speaks a sentence -> streamed in real time to Moonshine -> transcript
  5. Ollama honours JSON-schema constrained output (needed by the tool router)
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import EMBED_MODEL, OLLAMA_HOST, OLLAMA_MODEL  # noqa: E402
from app.llm_engine import OllamaEngine  # noqa: E402
from app.orchestrator import ROUTER_PROMPT, ROUTER_SCHEMA  # noqa: E402
from app.tools.normalize import normalize_transcript  # noqa: E402
from app.voice.asr import MoonshineASR  # noqa: E402
from app.voice.audio import resample_pcm16  # noqa: E402
from app.voice.tts import PiperTTS  # noqa: E402

SENTENCE = "Where is order B C one zero two four"
failures = []


def report(name, ok, detail="", hint=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)
        if hint:
            print(f"       fix: {hint}")


async def check_ollama():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            tags = (await c.get(f"{OLLAMA_HOST}/api/tags")).json()
        names = " ".join(m["name"] for m in tags.get("models", []))
        for model in (OLLAMA_MODEL, EMBED_MODEL):
            report(f"Ollama model {model}", model.split(":")[0] in names, hint=f"ollama pull {model}")
    except Exception as e:  # noqa: BLE001
        report("Ollama reachable", False, str(e), "start it with: ollama serve")


async def check_tts(tts: PiperTTS):
    try:
        t = time.perf_counter()
        tts.load()
        load = time.perf_counter() - t
        t = time.perf_counter()
        first, pcm = None, b""
        async for chunk in tts.synthesize("Your order has shipped and should arrive on Monday."):
            first = first or time.perf_counter() - t
            pcm += chunk
        total = time.perf_counter() - t
        secs = len(pcm) / 2 / tts.sample_rate
        report("Piper TTS", True, f"load {load:.1f}s, first audio {first*1000:.0f} ms, "
               f"{secs:.1f}s of speech in {total:.2f}s (real-time factor {total/secs:.2f}), {tts.sample_rate} Hz")
        return pcm
    except Exception as e:  # noqa: BLE001
        report("Piper TTS", False, str(e), "pip install piper-tts && python -m piper.download_voices "
               "en_US-lessac-medium --data-dir voices")


async def check_asr(asr: MoonshineASR, tts: PiperTTS):
    try:
        t = time.perf_counter()
        await asyncio.to_thread(asr.load)
        report("Moonshine ASR loads", True, f"{asr.arch}, {time.perf_counter()-t:.1f}s")
    except Exception as e:  # noqa: BLE001
        report("Moonshine ASR loads", False, str(e), "pip install moonshine-voice (wheels: Windows x64, Linux, "
               "macOS 15+ Apple Silicon); first run needs internet to fetch the model")
        return
    pcm = b"".join([c async for c in tts.synthesize(SENTENCE)])
    audio = resample_pcm16(pcm, tts.sample_rate, 16000)
    lines, done = [], threading.Event()
    stamps = {}

    def emit(kind, text):
        if kind == "final":
            lines.append(text)
            stamps["final"] = time.perf_counter()
            done.set()

    stream = asr.new_stream(emit)
    step = 1600 * 2                                           # 100 ms of PCM16 @ 16 kHz
    for i in range(0, len(audio), step):                      # speech, streamed in real time
        stream.feed(audio[i:i + step])
        await asyncio.sleep(0.1)
    stamps["speech_end"] = time.perf_counter()
    for _ in range(25):                                       # then up to 2.5 s of silence
        stream.feed(b"\x00\x00" * 1600)
        await asyncio.sleep(0.1)
        if done.is_set():
            break
    stream.close()
    text = " ".join(lines)
    ok = bool(lines)
    lat = (stamps.get("final", 0) - stamps["speech_end"]) * 1000
    report("Moonshine round trip", ok,
           f'heard "{text}" -> normalised "{normalize_transcript(text)}"; final transcript {lat:.0f} ms after the last speech frame'
           if ok else "no final transcript produced", "check the microphone-free path: is the model downloaded?")


async def check_structured_output():
    engine = OllamaEngine()
    try:
        t = time.perf_counter()
        raw = await asyncio.wait_for(engine.chat_once(
            [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": "Customer: what's happening with order BC1024"}],
            options={"temperature": 0, "num_predict": 96}, format=ROUTER_SCHEMA), 60)
        import json
        data = json.loads(raw)
        report("Constrained JSON (tool router)", data.get("tool") == "lookup_order",
               f"{raw.strip()} in {(time.perf_counter()-t)*1000:.0f} ms",
               "update Ollama (structured outputs need a recent version) or try a larger model")
    except Exception as e:  # noqa: BLE001
        report("Constrained JSON (tool router)", False, str(e), "is the chat model pulled and Ollama running?")


async def main():
    tts, asr = PiperTTS(), MoonshineASR()
    await check_ollama()
    await check_tts(tts)
    if tts.ready:
        await check_asr(asr, tts)
    await check_structured_output()
    print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
