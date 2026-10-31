"""
Voice latency benchmark against a RUNNING server (uvicorn app.main:app) with Ollama up.

Spoken test dialogues (data/voice_scenarios.json) are synthesised with Piper, then streamed to
/ws/voice in REAL TIME (100 ms frames every 100 ms) followed by silence, exactly like a mic.
For every spoken turn it records the server-side stages and an independent client-side timer:

  asr_ms                        last speech frame -> final transcript          (server)
  pre_generation_ms             retrieval / tool routing (concurrent; the slower one)
  llm_ttft_ms                   LLM request -> first token                     (server)
  tts_first_audio_ms            first text chunk -> first audio frame          (server)
  first_audio_after_speech_end  last speech frame -> first audio frame         (server)
  client_first_audio_ms         last speech frame SENT -> first audio frame RECEIVED (client)

    python benchmark/benchmark_voice.py                 # one user, all scenarios
    python benchmark/benchmark_voice.py --users 2       # two simultaneous users
    python benchmark/benchmark_voice.py --json results/voice_latency.json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.voice.audio import resample_pcm16, read_wav_16k  # noqa: E402
from app.voice.tts import PiperTTS  # noqa: E402

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "voice_scenarios.json"
STAGES = ["asr_ms", "pre_generation_ms", "llm_ttft_ms", "tts_first_audio_ms",
          "first_audio_after_speech_end_ms", "client_first_audio_ms"]
FRAME = 3200          # 100 ms of 16 kHz PCM16


async def synth(tts: PiperTTS, text: str) -> bytes:
    pcm = b"".join([c async for c in tts.synthesize(text)])
    return resample_pcm16(pcm, tts.sample_rate, 16000)


async def speak_turn(ws, audio: bytes, timeout: float = 90.0) -> dict:
    """Stream one utterance in real time, collect events until voice_metrics."""
    ev = {"first_audio": None, "final": None, "done": None, "metrics": None, "text": ""}
    t_last_voice = None

    async def reader():
        async for raw in ws:
            now = time.perf_counter()
            if isinstance(raw, bytes):
                ev["first_audio"] = ev["first_audio"] or now
                continue
            msg = json.loads(raw)
            if msg["type"] == "final":
                ev["final"] = msg["text"]
            elif msg["type"] == "token":
                ev["text"] += msg["text"]
            elif msg["type"] == "done":
                ev["done"] = msg
            elif msg["type"] == "voice_metrics":
                ev["metrics"] = msg
                return
            elif msg["type"] == "error":
                ev["error"] = msg["message"]

    rd = asyncio.create_task(reader())
    for i in range(0, len(audio), FRAME):                       # speech
        await ws.send(audio[i:i + FRAME])
        t_last_voice = time.perf_counter()
        await asyncio.sleep(0.1)
    for _ in range(40):                                         # silence until the server ends the turn
        if ev["final"] is not None:
            break
        await ws.send(b"\x00\x00" * 1600)
        await asyncio.sleep(0.1)
    await asyncio.wait_for(rd, timeout)
    m = dict(ev["metrics"] or {})
    if ev["first_audio"] and t_last_voice:
        m["client_first_audio_ms"] = round((ev["first_audio"] - t_last_voice) * 1000, 1)
    m["heard"], m["reply"] = ev["final"], ev["text"].strip()
    m["kind"] = _kind(ev["done"])
    return m


def _kind(done) -> str:
    if not done:
        return "error"
    t = done.get("timing", {})
    if t.get("asked"):
        return "tool-question"
    if done.get("tools"):
        return "tool-answer"
    return "rag" if (done.get("retrieval") or {}).get("status") == "ok" else "other"


async def run_user(uid: int, base: str, scenarios: dict, tts: PiperTTS, wav_dir: Path | None) -> list[dict]:
    rows = []
    async with httpx.AsyncClient() as http:
        for name, utterances in scenarios.items():
            sid = (await http.post(f"{base}/api/session")).json()["session_id"]
            ws_url = base.replace("http", "ws") + "/ws/voice"
            async with websockets.connect(ws_url, max_size=None) as ws:
                await ws.send(json.dumps({"type": "start", "session_id": sid}))
                ready = json.loads(await ws.recv())
                if ready["type"] != "ready":
                    raise SystemExit(f"voice endpoint not ready: {ready}")
                for i, text in enumerate(utterances):
                    wav = (wav_dir / f"{name}_{i}.wav") if wav_dir else None
                    audio = read_wav_16k(wav) if wav and wav.exists() else await synth(tts, text)
                    m = await speak_turn(ws, audio)
                    await ws.send(json.dumps({"type": "resume"}))
                    m.update(user=uid, scenario=name, turn=i)
                    rows.append(m)
                    print(f"  user {uid} {name}[{i}] {m['kind']:<13} heard={m['heard']!r}  "
                          f"speech-end->audio {m.get('first_audio_after_speech_end_ms')} ms "
                          f"(client {m.get('client_first_audio_ms')} ms)")
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in sorted({r["kind"] for r in rows} | {"ALL"}):
        sel = [r for r in rows if kind == "ALL" or r["kind"] == kind]
        out[kind] = {"n": len(sel)}
        for s in STAGES:
            xs = sorted(r[s] for r in sel if r.get(s) is not None)
            if xs:
                out[kind][s] = {"mean": round(statistics.mean(xs)), "p50": round(statistics.median(xs)),
                                "p95": round(xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))])}
    return out


def report(summary: dict, users: int):
    print(f"\nPer-stage latency in ms (mean / p50 / p95), {users} concurrent user(s)\n")
    head = "| turn type (n) | " + " | ".join(STAGES) + " |"
    print(head + "\n|" + "---|" * (len(STAGES) + 1))
    for kind, st in summary.items():
        cells = [f"{st[s]['mean']} / {st[s]['p50']} / {st[s]['p95']}" if s in st else "-" for s in STAGES]
        print(f"| {kind} ({st['n']}) | " + " | ".join(cells) + " |")
    allv = summary["ALL"].get("first_audio_after_speech_end_ms")
    if allv:
        print(f"\nTarget: end of speech -> first audio <= ~2000 ms. Measured p50 {allv['p50']} ms, p95 {allv['p95']} ms.")


async def main(args):
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    tts = PiperTTS()
    tts.load()
    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_user(u, args.server, scenarios, tts, Path(args.wav_dir) if args.wav_dir else None)
                                     for u in range(args.users)])
    rows = [r for user_rows in results for r in user_rows]
    summary = summarize(rows)
    report(summary, args.users)
    print(f"(wall time {time.perf_counter() - t0:.0f} s)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--wav-dir", help="optional dir of recorded <scenario>_<turn>.wav files (mono) used instead of TTS")
    ap.add_argument("--json")
    asyncio.run(main(ap.parse_args()))
