"""
Manual stress / failure-injection script for Phase VI evaluation. Run this
against a LIVE server (`uvicorn app.main:app`) with Ollama running.

    python tests/manual_stress_test.py

It exercises:
  1. Several concurrent sessions sending messages at the same time
     (checks that one slow generation doesn't block another user).
  2. A malformed message.
  3. A disconnect mid-stream (opens a socket, sends a message, closes
     immediately without reading the stream).
  4. Many sessions opened back-to-back.

This is a manual/exploratory script (not a pytest suite) because it talks to
a real running server + real model, which is what Phase VI explicitly asks
you to test by hand.
"""
import asyncio
import json
import time

import httpx
import websockets

BASE_HTTP = "http://localhost:8000"
BASE_WS = "ws://localhost:8000/ws/chat"


async def new_session() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_HTTP}/api/session")
        return resp.json()["session_id"]


async def concurrent_user(user_id: int, message: str):
    session_id = await new_session()
    start = time.perf_counter()
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "message", "session_id": session_id, "text": message}))
        tokens = 0
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg["type"] == "token":
                tokens += 1
            if msg["type"] in ("done", "error", "blocked"):
                break
        elapsed = time.perf_counter() - start
        r = msg.get("retrieval") or {}
        print(f"[user {user_id}] done in {elapsed:.2f}s, {tokens} token chunks, "
              f"final event={msg['type']}, retrieval={r.get('status')} {r.get('ms')} ms, "
              f"sources={[s['doc_id'].split('/')[-1] for s in msg.get('sources', [])]}")


async def test_concurrency(n_users: int = 4):
    print(f"\n--- Concurrency test: {n_users} simultaneous users ---")
    messages = [
        "Where is my order?",
        "Can I return my headphones?",
        "Does the blender come in red?",
        "What's your shipping policy?",
    ]
    start = time.perf_counter()
    await asyncio.gather(*[
        concurrent_user(i, messages[i % len(messages)]) for i in range(n_users)
    ])
    print(f"All {n_users} users completed in {time.perf_counter() - start:.2f}s total "
          f"(should be well under n_users * single-response-time if truly concurrent)")


async def test_malformed_message():
    print("\n--- Malformed message test ---")
    async with websockets.connect(BASE_WS) as ws:
        await ws.send("{this is not json")
        raw = await ws.recv()
        msg = json.loads(raw)
        assert msg["type"] == "error", "Expected an error event for malformed JSON"
        print("PASS: server returned a clean error instead of crashing:", msg["message"])


async def test_unknown_session():
    print("\n--- Unknown session_id test ---")
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "message", "session_id": "ghost", "text": "hi"}))
        raw = await ws.recv()
        msg = json.loads(raw)
        assert msg["type"] == "error"
        print("PASS: server returned a clean error for unknown session:", msg["message"])


async def test_mid_stream_disconnect():
    print("\n--- Mid-stream disconnect test ---")
    session_id = await new_session()
    ws = await websockets.connect(BASE_WS)
    await ws.send(json.dumps({
        "type": "message", "session_id": session_id,
        "text": "Tell me about your full return and shipping policy in detail.",
    }))
    await ws.recv()  # read just the first "stage" event
    await ws.close()  # then abruptly disconnect
    print("PASS: closed connection mid-stream; check server logs -- it should log the "
          "disconnect and NOT crash or hang.")

    # Confirm the server is still alive and responsive afterwards.
    healthy = httpx.get(f"{BASE_HTTP}/api/health").status_code == 200
    print(f"Server still responsive after abrupt disconnect: {healthy}")


async def test_rag_paths():
    """Grounded answer (sources), no-match query (no sources), huge message (budget)."""
    print("\n--- RAG paths ---")
    cases = {
        "grounded": "Can I use the WaveSound H200 headphones with my laptop?",
        "no match": "Do you sell dishwashers and refrigerators?",
        "max-length message": "Tell me about shipping. " + "I really need to know. " * 160,
    }
    for label, text in cases.items():
        session_id = await new_session()
        async with websockets.connect(BASE_WS) as ws:
            await ws.send(json.dumps({"type": "message", "session_id": session_id, "text": text[:4000]}))
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] in ("done", "error", "blocked"):
                    break
        r = msg.get("retrieval") or {}
        print(f"[{label}] final={msg['type']} retrieval={r.get('status')} "
              f"sources={len(msg.get('sources', []))}")


async def main():
    await test_rag_paths()
    await test_malformed_message()
    await test_unknown_session()
    await test_mid_stream_disconnect()
    await test_concurrency(4)


if __name__ == "__main__":
    asyncio.run(main())
