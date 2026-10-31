"""
FastAPI backend for the E-Commerce Order Support Assistant.

Endpoints:
  GET  /api/health          -- health check (also pings Ollama)
  POST /api/session         -- create a new chat session
  GET  /api/session/{id}    -- inspect session state
  DELETE /api/session/{id}  -- end a session
  WS   /ws/chat             -- real-time streaming chat

The WebSocket protocol is a small JSON envelope in both directions; see
app/schemas.py for the exact shapes and README.md for examples.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.config import DB_PATH, EMBED_MODEL, OLLAMA_MODEL, TOOLS_ENABLED
from app.conversation_manager import ConversationManager
from app.domain import GREETING_MESSAGE as GREETING_GENERIC

from app.llm_engine import LLMEngineError, OllamaEngine
from app.orchestrator import ToolOrchestrator
from app.rag.embedder import OllamaEmbedder
from app.rag.retriever import Retriever
from app.tools.db import ensure_db
from app.voice.asr import MoonshineASR
from app.voice.pipeline import VoiceConnection
from app.voice.tts import PiperTTS
from app.schemas import (
    ClientChatMessage,
    HealthResponse,
    NewSessionResponse,
    SessionInfoResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecommerce-assistant")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

engine = OllamaEngine()
embedder = OllamaEmbedder()
retriever = Retriever(embedder)
orchestrator = ToolOrchestrator(engine) if TOOLS_ENABLED else None
manager = ConversationManager(engine=engine, retriever=retriever, orchestrator=orchestrator,
                              crm_db=DB_PATH if TOOLS_ENABLED else None)
asr, tts = MoonshineASR(), PiperTTS()
voice_state = {"error": ""}


async def _load_voice():
    """Load ASR + TTS off the event loop (first run downloads the Moonshine model)."""
    try:
        await asyncio.to_thread(asr.load)
        await asyncio.to_thread(tts.load)
        logger.info("Voice models ready (ASR %s, TTS sample rate %s)", asr.arch, tts.sample_rate)
    except Exception as e:  # noqa: BLE001 - the text assistant must keep working without voice
        voice_state["error"] = str(e)
        logger.warning("Voice models not available: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the on-disk index (never raises: a missing index just disables RAG and
    # the assistant keeps working). Warm the embedding model in the background so
    # the first real query doesn't pay the model-load cost, without delaying startup.
    if retriever.load():
        app.state.warmup = asyncio.create_task(retriever.warm_up())
    if TOOLS_ENABLED:
        ensure_db(DB_PATH)                              # create + seed the mock backend if missing
    app.state.voice_load = asyncio.create_task(_load_voice())
    yield
    await embedder.aclose()


app = FastAPI(title="E-Commerce Order Support Assistant", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/demo project; tighten for real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
async def health():
    reachable = await engine.health_check()
    r = manager.retriever
    ready = bool(r and r.ready)
    return HealthResponse(
        status="ok" if reachable and ready else "degraded",
        ollama_reachable=reachable,
        model=OLLAMA_MODEL,
        retrieval_ready=ready,
        index_chunks=len(r.store) if ready else 0,
        embed_model=EMBED_MODEL,
        voice_ready=asr.ready and tts.ready,
        tools_enabled=manager.orchestrator is not None,
        voice_error=voice_state["error"],
    )


@app.post("/api/session", response_model=NewSessionResponse)
async def create_session(user_id: str | None = None):
    """Optional ?user_id= links the chat to a remembered customer (CRM bonus)."""
    session = manager.create_session(user_id)
    return NewSessionResponse(session_id=session.session_id, greeting=manager.greeting(session),
                              returning=bool(session.profile and manager.greeting(session) != GREETING_GENERIC))


@app.get("/api/session/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    session = manager.get_session(session_id)
    if session is None:
        return SessionInfoResponse(
            session_id=session_id, stage="unknown", active_intent=None,
            turn_count=0, history_length=0,
        )
    return SessionInfoResponse(
        session_id=session.session_id,
        stage=session.stage,
        active_intent=session.active_intent,
        turn_count=session.turn_count,
        history_length=len(session.history),
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    manager.end_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    Real-time streaming chat endpoint.

    Client sends: {"type": "message", "session_id": "...", "text": "..."}
    Server streams back a sequence of JSON events per turn:
        {"type": "stage", "stage": "..."}
        {"type": "token", "text": "..."}          (repeated, streamed word/piece by piece)
        {"type": "done", "time_to_first_token": ..., "total_time": ..., "tokens_per_second": ..., "stage": "..."}
      or, if the message is off-topic:
        {"type": "blocked", "text": "..."}
      or, on any error:
        {"type": "error", "message": "..."}

    Each connection is handled by its own asyncio task, and the underlying
    LLM call is a fully async streaming HTTP request, so one user's slow
    generation never blocks another connected user's turn.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

            # --- Robust error handling: malformed JSON / schema -------------
            try:
                payload = json.loads(raw)
                client_msg = ClientChatMessage(**payload)
            except (json.JSONDecodeError, ValidationError) as e:
                await _safe_send(websocket, {
                    "type": "error",
                    "message": f"Malformed request: {e}",
                })
                continue

            session = manager.get_session(client_msg.session_id)
            if session is None:
                await _safe_send(websocket, {
                    "type": "error",
                    "message": "Unknown or expired session_id. Create a new session via "
                               "POST /api/session or the 'New chat' button.",
                })
                continue

            # --- Run one full turn, streaming events back -------------------
            try:
                async for event_type, payload in manager.handle_user_message(
                    session, client_msg.text
                ):
                    if event_type == "stage":
                        await _safe_send(websocket, {"type": "stage", "stage": payload})
                    elif event_type == "token":
                        await _safe_send(websocket, {"type": "token", "text": payload})
                    elif event_type == "blocked":
                        await _safe_send(websocket, {"type": "blocked", "text": payload})
                    elif event_type == "done":
                        await _safe_send(websocket, {"type": "done", **payload})
            except LLMEngineError as e:
                logger.warning("LLM engine error: %s", e)
                await _safe_send(websocket, {
                    "type": "error",
                    "message": f"The local model backend hit an error: {e}",
                })
            except Exception as e:  # noqa: BLE001 - last-resort guard, must never crash the socket
                logger.exception("Unexpected error handling turn")
                await _safe_send(websocket, {
                    "type": "error",
                    "message": "An unexpected server error occurred while generating a response. "
                               "Please try again.",
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected (outer)")
    except Exception:
        logger.exception("Fatal error in websocket handler")


@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """
    Real-time voice chat. The text endpoint /ws/chat is unchanged; this one is added beside it.

    Client -> server
      {"type":"start","session_id":"..."}   bind to an existing session (shared with text chat)
      <binary>                              mic audio: PCM16 mono 16 kHz, ~100 ms per frame
      {"type":"resume"}                     assistant playback finished; start listening again
    Server -> client (JSON)
      ready | partial {text} | final {text} | stage | token | done | blocked | error |
      tts_done | voice_metrics {asr_ms, pre_generation_ms, llm_ttft_ms, ...}
    Server -> client (binary)
      audio frame: uint32 seq, uint32 sample_rate (little endian) + PCM16 mono
    """
    await websocket.accept()
    conn = None
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if conn is not None:
                    await conn.on_audio(msg["bytes"])
                continue
            try:
                data = json.loads(msg.get("text") or "")
                kind = data["type"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                await _safe_send(websocket, {"type": "error", "message": f"Malformed request: {e}"})
                continue
            if kind == "start":
                session = manager.get_session(str(data.get("session_id", "")))
                if session is None:
                    await _safe_send(websocket, {"type": "error", "message": "Unknown or expired session_id."})
                elif not (asr.ready and tts.ready):
                    await _safe_send(websocket, {"type": "error", "message": "Voice models are not loaded"
                                                 + (f": {voice_state['error']}" if voice_state["error"] else " yet; try again shortly.")})
                else:
                    if conn is not None:
                        await conn.close()
                    conn = VoiceConnection(asr, tts, manager, session, lambda o: _safe_send(websocket, o),
                                           lambda b: _safe_send_bytes(websocket, b))
                    conn.start()
                    returning = bool(session.profile) and manager.greeting(session) != GREETING_GENERIC
                    await _safe_send(websocket, {"type": "ready", "sample_rate": tts.sample_rate,
                                                 "greeting": returning})
                    if returning:                       # CRM bonus: greet returning customers by voice too
                        asyncio.create_task(conn.say(manager.greeting(session)))
            elif kind == "resume" and conn is not None:
                conn.resume()
            else:
                await _safe_send(websocket, {"type": "error", "message": f"Unknown message type: {kind}"})
    except WebSocketDisconnect:
        logger.info("Voice client disconnected")
    except Exception:
        logger.exception("Fatal error in voice websocket handler")
    finally:
        if conn is not None:
            await conn.close()


async def _safe_send(websocket: WebSocket, data: dict):
    """Send JSON, swallowing errors from a socket that already closed mid-stream
    (e.g. client disconnected mid-response) so we don't raise inside the loop."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        logger.info("Could not send to websocket (likely disconnected mid-stream)")


async def _safe_send_bytes(websocket: WebSocket, data: bytes):
    try:
        await websocket.send_bytes(data)
    except Exception:
        logger.info("Could not send audio to websocket (likely disconnected mid-stream)")


# Serve the simple chat frontend at /
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
