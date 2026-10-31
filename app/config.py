"""
Central configuration for the E-Commerce Order Support Assistant.

All values can be overridden with environment variables so the same
code runs unchanged on any teammate's machine / any grading machine.
"""
import os

# ---- Ollama connection -----------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct-q4_K_M")

# Request timeout for a single generation call (seconds). Local CPU inference
# on small models is slow-ish on first token, so keep this generous.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# ---- Conversation memory management ----------------------------------------
# We never send the *entire* raw history once a session grows long. Instead we
# use a hybrid "sliding window + rolling summary" scheme (see README, Phase II
# section, for the full rationale):
#
#   1. The domain system prompt is ALWAYS sent (it defines who the bot is).
#   2. A single "rolling summary" message captures anything older than the
#      window (updated in place, never grows unbounded).
#   3. The most recent MAX_TURNS_IN_WINDOW user/assistant turn-pairs are sent
#      verbatim so the model has precise, faithful access to recent context.
#
# This keeps prompt size roughly constant (O(1) growth) no matter how long
# the conversation runs, which matters a lot for CPU inference latency.
MAX_TURNS_IN_WINDOW = int(os.getenv("MAX_TURNS_IN_WINDOW", "6"))

# Token counts are estimated as len(text) / CHARS_PER_TOKEN_ESTIMATE (no tokenizer
# dependency). The prompt budget itself is derived from num_ctx; see the RAG section.
CHARS_PER_TOKEN_ESTIMATE = 3.5  # conservative: numbers/punctuation tokenize worse than prose

# When the window needs to be compacted, we ask the *same* local model to
# summarize the turns falling out of the window. This is plain prompting of
# the model we already loaded -- not RAG, not an external tool/agent.
SUMMARY_TRIGGER_TURNS = MAX_TURNS_IN_WINDOW + 2

# ---- Session management -----------------------------------------------------
SESSION_IDLE_TIMEOUT_SECONDS = int(os.getenv("SESSION_IDLE_TIMEOUT_SECONDS", "1800"))

# ---- Generation parameters ---------------------------------------------------
GENERATION_OPTIONS = {
    "temperature": 0.4,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,
    "num_predict": 320,  # hard cap on answer length: bounds worst-case latency
}

# Keep models resident between turns; on CPU a cold reload costs several seconds.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


# =============================================================================
# RAG (Assignment 2)
# =============================================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", PROJECT_ROOT / "data" / "corpus"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "index"))

# ---- Embedding model (served by the same local Ollama) ----------------------
# `all-minilm` = 22M-param MiniLM, 384-dim, 256-token limit. Pull with:
#   ollama pull all-minilm
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "5"))

# ---- Chunking ---------------------------------------------------------------
# Max ~600 chars (~150-170 tokens) keeps every chunk well under MiniLM's
# 256-token limit even after the title prefix is added.
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))
# Markdown sections dropped before embedding (identical boilerplate in every
# corpus file would otherwise inflate similarity for unrelated queries).
BOILERPLATE_SECTIONS = {"source"}

# ---- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
# Cosine-similarity floor. A query whose BEST chunk scores below this is
# treated as "no relevant match". MUST be calibrated per embedding model:
#   python scripts/eval_retrieval.py   (prints score distributions + suggestion)
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
RETRIEVAL_TIMEOUT = float(os.getenv("RETRIEVAL_TIMEOUT", "1.5"))  # seconds, embed+search
CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", "256"))
# Circuit breaker: after N consecutive failures skip retrieval for COOLDOWN s
# so a dead embedder costs ~0 ms per turn instead of RETRIEVAL_TIMEOUT.
BREAKER_FAILURES = int(os.getenv("BREAKER_FAILURES", "3"))
BREAKER_COOLDOWN = float(os.getenv("BREAKER_COOLDOWN", "30"))

# ---- Context-window budget (num_ctx = GENERATION_OPTIONS["num_ctx"]) --------
# prompt budget = num_ctx - RESPONSE_RESERVE_TOKENS
# priority when packing: system > current question > retrieved chunks
#                        (capped at RAG_MAX_CONTEXT_TOKENS) > recent history
RESPONSE_RESERVE_TOKENS = int(os.getenv("RESPONSE_RESERVE_TOKENS", "512"))
RAG_MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "800"))


# =============================================================================
# TOOLS (Assignment 3)
# =============================================================================
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "brightcart.db"))
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "1") == "1"
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", "2.0"))        # one tool call
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT", "6.0"))    # LLM tool-selection call
ROUTER_NUM_PREDICT = int(os.getenv("ROUTER_NUM_PREDICT", "96"))
MAX_VERIFY_ATTEMPTS = int(os.getenv("MAX_VERIFY_ATTEMPTS", "3"))
RETURN_WINDOW_DAYS = int(os.getenv("RETURN_WINDOW_DAYS", "30"))  # corpus: returns_033

# =============================================================================
# VOICE (Assignment 3)
# =============================================================================
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
ASR_MODEL_ARCH = os.getenv("ASR_MODEL_ARCH", "small-streaming")  # tiny-streaming | small-streaming | medium-streaming
ASR_UPDATE_INTERVAL = float(os.getenv("ASR_UPDATE_INTERVAL", "0.3"))  # seconds between partial transcripts
TTS_VOICE_PATH = Path(os.getenv("TTS_VOICE_PATH", PROJECT_ROOT / "voices" / "en_US-lessac-medium.onnx"))
VOICE_NUM_PREDICT = int(os.getenv("VOICE_NUM_PREDICT", "110"))   # spoken answers are short
SPEECH_SAMPLE_RATE = 16000                                       # mic audio format (PCM16 mono)
# TTS chunking: flush the FIRST chunk early (at a clause boundary) to cut first-audio latency.
TTS_FIRST_CHUNK_MIN_CHARS = int(os.getenv("TTS_FIRST_CHUNK_MIN_CHARS", "28"))
TTS_MAX_CHUNK_CHARS = int(os.getenv("TTS_MAX_CHUNK_CHARS", "160"))
