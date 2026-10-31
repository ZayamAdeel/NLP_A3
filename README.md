# BrightCart Support Assistant (Assignments 1 + 2 + 3)

A fully local, CPU-only e-commerce support assistant ("Ava"). It runs entirely on your machine:
a quantized LLM served by Ollama, retrieval over a 50-document knowledge base, four LLM-callable
tools, and a real-time voice pipeline (streaming speech-to-text and text-to-speech).

| Assignment | What it added |
|---|---|
| 1 | FastAPI + WebSocket streaming chat, session memory, domain flow, off-topic guard, web UI |
| 2 | RAG: offline indexing pipeline, retrieval, context-window budget, caching, failure handling, **citations (bonus)** |
| 3 | **Voice** (Moonshine ASR, Piper TTS), **tools** (`search_products`, `lookup_order`, `check_return_eligibility`), per-turn latency metrics, **CRM tool (bonus)** |

> **Verification status — please read.** This was built in a sandbox with **no network access**
> (no Ollama, no model downloads, no microphone). Executed there: chunker, indexer, retriever,
> tools, orchestrator, conversation manager, the voice pipeline and the real `/ws/voice` and
> `/ws/chat` handlers (driven through a fake socket with fake ASR/TTS): **159 tests passing**
> (149 in the new suites + the 10 Assignment 1 manager tests). The browser audio path was also
> run in headless Chrome with a fake microphone and a mocked server. **Not run:** the real Moonshine,
> Piper and Ollama models (so **every latency number and the LLM's tool-routing accuracy are
> unmeasured**), the FAISS path, and `tests/test_api.py` (needs real FastAPI). §8 gives the exact
> commands that produce those numbers; tables marked *(paste)* are placeholders until you run them.
> Run `python scripts/smoke_voice.py` first.

---

## 1. Architecture

```
 ┌───────────────────────────── Browser (static/) ─────────────────────────────┐
 │ chat UI + citation chips + tool cards        mic-worklet.js → 16 kHz PCM16 │
 │ app.js ── /ws/chat (text, unchanged)         voice.js ── /ws/voice (audio)  │
 └───────────────┬──────────────────────────────────────┬───────────────────────┘
                 │ JSON events                          │ PCM16 frames ↑   PCM16 frames + JSON ↓
 ┌───────────────▼──────────────────────────────────────▼───────────────────────┐
 │ FastAPI backend (app/main.py)   REST: /api/health, /api/session              │
 │                                                                              │
 │  /ws/chat ─────────────┐        /ws/voice ──► VoiceConnection (per user)     │
 │                        │          ASR stream ─ partial/final ─► normalise    │
 │                        │          ▲ Moonshine (shared model)         │       │
 │                        │          │                                  ▼       │
 │                        ▼          └──────────────  handle_user_message(voice=True)
 │         ┌─────────────────────────── Conversation Manager ────────────────┐  │
 │         │ off-topic guard → stage → ┬─ Tool Orchestrator ─┐  (concurrent)  │  │
 │         │                           │  gate → LLM router  │                │  │
 │         │                           │  → verify → slots   │                │  │
 │         │                           │  → tool (timeout)   │                │  │
 │         │                           └─ Retriever (RAG) ───┘                │  │
 │         │ → budgeted prompt → LLM stream → record → done(+sources,tools)   │  │
 │         └───────┬─────────────────┬───────────────────┬───────────────────┘  │
 │                 ▼                 ▼                   ▼                      │
 │   Ollama /api/chat        Ollama /api/embed     SQLite mock backend          │
 │   qwen2.5:1.5b Q4_K_M     all-minilm → vectors  products · orders · CRM      │
 │                                                                              │
 │   tokens ─► SpeechChunker ─► speakable() ─► Piper (thread, locked) ─► audio ─┼─► client
 └──────────────────────────────────────────────────────────────────────────────┘
 OFFLINE:  data/corpus/*.md ─ chunking ─ embed ─ index/ (scripts/build_index.py)   data/brightcart.db ← scripts/seed_db.py
```

Text and voice are **one system**: a spoken turn calls the same `ConversationManager.handle_user_message`
as a typed one, so history, RAG, tools, verification state and the CRM profile are shared. You can
speak a question, type the answer to the follow-up, and keep going in the same session.

---

## 2. Model choices

| Role | Model | Why | Alternatives considered |
|---|---|---|---|
| LLM | `qwen2.5:1.5b-instruct-q4_K_M` (Ollama) | Inside the 0.5–4B range; interactive speed on CPU; follows structured prompts; supports JSON-schema constrained output through Ollama | `qwen2.5:3b` / `phi3.5` (better judgement, ~2× slower: use only if the tool eval below shows the 1.5B routing is too weak) |
| Embeddings | `all-minilm` via Ollama (22M, 384-d) | CPU-light, async through the same Ollama, no PyTorch install | `nomic-embed-text` (slower, needs prefixes) |
| **ASR** | **Moonshine Voice**, `small-streaming` (`pip install moonshine-voice`, MIT) | Built for streaming: emits live partial transcripts and a completed line when speech ends, so the final transcript is ready right after the end-of-speech silence instead of after a whole-utterance decode. Small enough to share the CPU with the LLM. Python API, models cached locally after first download | Whisper/faster-whisper: no live partials and decodes only after speech ends (adds to the 2 s budget). Vosk: streams, but noticeably less accurate on product names and accents |
| **TTS** | **Piper** (`pip install piper-tts`, GPL-3.0; voice `en_US-lessac-medium`) | Fast ONNX synthesis, much faster than real time on CPU; yields audio per sentence; small voice files | Kokoro: more natural, but one independent M2 measurement put first-sentence latency at roughly 2–3× Piper's, which would consume the whole latency budget on a CPU shared with the LLM |

Tiny/Small/Medium Moonshine variants are selectable with `ASR_MODEL_ARCH`. **Platform note:** the
`moonshine-voice` wheels cover Windows x64, Linux (x64/arm64) and **macOS 15+ on Apple Silicon only**;
if you are on an Intel Mac or older macOS the install will fail (the smoke test tells you immediately).
Models are downloaded on first use (internet needed once), then run offline.

---

## 3. Document collection and retrieval setup (Assignment 2)

* **Corpus:** `data/corpus/`, 50 synthetic markdown documents in 6 categories (15 product specs, 10 product FAQs,
  7 shipping, 7 returns, 5 payments, 6 orders/accounts). It contains **no prices, stock or order data**; the
  tools below supply those, so retrieval and tools never contradict each other.
* **Chunking:** drop the identical `## Source` boilerplate (it would inflate similarity for every query), split
  into sentences, pack ≤ 600 characters with a 1-sentence overlap, prefix each chunk with its document title.
  With ~440-character documents this yields 50 chunks. (`app/rag/chunking.py`)
* **Index:** `scripts/build_index.py`, re-runnable and incremental (vectors reused by `sha256(chunk text)`).
  Exact inner-product search over L2-normalised vectors (FAISS `IndexFlatIP` if installed, else NumPy).
* **Retrieval:** top-k = 4, cosine floor `MIN_SCORE` (**calibrate with `scripts/eval_retrieval.py`**), LRU
  result cache, 1.5 s timeout, circuit breaker; smalltalk skips retrieval; short follow-ups borrow the
  previous substantive question. Results are injected into the **last user message** so the static system
  prompt stays a stable prefix.
* **Context budget:** `num_ctx` 4096 − 512 reserved for the answer; priority system > current message > tool
  result / excerpts (≤ 800 tokens) > recent history (oldest dropped first).
* **Citations (Assignment 2 bonus):** each answer shows clickable source chips with the excerpt the model saw.
  Tool results appear as sources too (e.g. the product document behind a catalogue hit).

---

## 4. Conversation design (Assignment 1)

Stages: `greeting → intent_identification → info_gathering → resolution → confirmation → closing`. A message
that matches a *different* intent than the active one resets to `info_gathering` while **keeping history**.
Off-topic messages (weather, coding, jailbreak phrases) are blocked by a whole-word filter before any
retrieval, tool or LLM call. Memory: sliding window of 6 turn-pairs plus a rolling summary, run *after* each
answer so it never delays one.

---

## 5. Tools (Assignment 3, Phase IV)

### How the orchestrator decides (be ready to explain this)
It is **prompt-based function calling with constrained JSON output**, in six steps (`app/orchestrator.py`):

1. **Gate**: a cheap regex asks "could this message need a tool?". Chit-chat never pays for an LLM call.
2. **Route**: one short LLM call whose output is decoded against a JSON schema (Ollama structured outputs),
   so a 1.5B model cannot emit broken syntax. It picks `tool` and extracts free-text arguments.
3. **Verify ("the LLM proposes, code verifies")**: order ids, codes, condition, packaging and prices are only
   accepted if they actually appear in what the customer said. A model that *claims* "unused, with box" for a
   customer who said neither is overruled, and the customer is asked.
4. **Slots**: missing required arguments are asked for **deterministically** (one question at a time, no LLM
   call); the next answer ("4721", "yes I have the box") fills the slot by regex, again without an LLM call.
5. **Execute**: async handler under `asyncio.wait_for` (2 s). Timeouts and crashes become a plain "I couldn't
   look that up right now" result instead of a hang or an error.
6. **Answer**: the tool's result summary goes into the LLM prompt and the final answer streams (and is spoken).
   Clarifying questions skip the LLM entirely, which makes them the fastest voice turns.

Privileged context (the session, DB path) is injected by the orchestrator; the LLM can never claim an order is
"verified". Router failures (bad JSON, timeout) fall back to a normal RAG answer.

### Tool 1 — `search_products`
*Why it earns its place:* the corpus has no prices or stock, and "under 10,000 rupees" is a structured filter.
```json
{"name": "search_products",
 "description": "Search BrightCart's product catalogue by keywords, category, maximum price in rupees, and stock availability. Returns up to 3 matches with prices.",
 "parameters": {"type": "object", "properties": {
   "query": {"type": "string", "description": "product keywords, e.g. 'wireless headphones'"},
   "category": {"type": "string", "enum": ["audio","charging","computer accessories","gift cards","home office","phone accessories","storage","wearables"]},
   "max_price": {"type": "number", "description": "maximum price in rupees"},
   "in_stock_only": {"type": "boolean"}}, "required": []}}
```
*Example.* Customer (voice): "find wireless headphones under ten thousand rupees" → transcript normalised to
`under 10000 rupees` → router output `{"tool":"search_products","query":"wireless headphones","max_price":10000}` →
tool result *"Found 1 matching product(s): 1) WaveSound H200 Wireless Headphones (BC-HEAD-001): 9,500 rupees, in stock."* →
Ava: "The WaveSound H200 wireless headphones are 9,500 rupees and in stock." If filters exclude everything, the
result names the closest match and why it was excluded. *(Illustrative; capture a real transcript for submission.)*

### Tool 2 — `lookup_order`
*Why:* order status is per-customer state the documents cannot hold (the corpus explicitly says not to claim an
order status without an order system).
```json
{"name": "lookup_order",
 "description": "Look up an order's status, carrier, tracking reference and estimated delivery. Requires the order number and the customer's verification code (last four digits of the phone number on the order).",
 "parameters": {"type": "object", "properties": {
   "order_id": {"type": "string", "description": "e.g. BC1024"},
   "verification_code": {"type": "string", "description": "last 4 digits of the phone number on the order"}},
  "required": ["order_id", "verification_code"]}}
```
*Example.* "where is order B C one zero two four" → `BC1024`; code missing → Ava asks *"For security, what are the
last four digits of the phone number on the order?"* (no LLM call) → "four seven two one" → `4721` → tool →
*"Order BC1024: status shipped; carrier TCS, tracking reference TCS-884201; estimated delivery Monday 26 October…"* →
Ava gives a two-sentence spoken summary; the tracking reference is **shown on screen** in a tool card, not read aloud.
Security: a wrong code and an unknown order give the *same* reply; after 3 failures the tool locks for the session.

### Tool 3 — `check_return_eligibility`
*Why:* the policy (30 days from delivery, unused, original packaging, gift cards/final sale excluded, damaged →
review) needs date arithmetic and rule combination that an LLM gets wrong; here it is computed in code from the
order's real delivery date, only for an order the customer has verified.
```json
{"name": "check_return_eligibility",
 "description": "Check whether an item from a delivered order can be returned under BrightCart's return policy (30 days, unused, original packaging).",
 "parameters": {"type": "object", "properties": {
   "order_id": {"type": "string"},
   "item": {"type": "string", "description": "product name or id, e.g. 'headphones'"},
   "condition": {"type": "string", "enum": ["unused", "used", "damaged"]},
   "has_original_packaging": {"type": "boolean"}},
  "required": ["order_id", "item", "condition", "has_original_packaging"]}}
```
*Example (4 spoken turns).* "can I return the headphones from order B C one zero two five" → asks condition → "they are unused" →
asks packaging → "yes I still have the box" → asks code → "three three zero eight" → tool →
*"…ELIGIBLE. Delivered 10 days ago, so 20 day(s) remain in the 30-day window…"*. The router LLM ran **once**; the
other three turns were regex slot-filling. Rules (all from the corpus): not delivered → no; gift card/final sale → no;
damaged → *needs review, no refund promised*; past 30 days → no (says by how many days); used → no; no packaging → no.
Changes from the original proposal: added `has_original_packaging` (the policy requires it) and replaced `product_id`
with free-text `item` (people say "headphones", not "BC-HEAD-001").

### Bonus — CRM tool `customer_profile` (`app/tools/crm.py`)
```json
{"name": "customer_profile",
 "description": "Remember or recall a returning customer's details across chats: save their name, contact number or email, or a stated preference; read back what is saved; or delete it when they ask to be forgotten.",
 "parameters": {"type": "object", "properties": {
   "action": {"type": "string", "enum": ["get", "update", "forget"]},
   "name": {"type": "string"},
   "contact": {"type": "string", "description": "email address or phone number"},
   "preference": {"type": "string", "description": "e.g. 'prefers black headphones'"}}, "required": []}}
```
* **Storage:** SQLite `crm_profiles` (name, contact, preferences, last 10 interactions), keyed by a random `user_id`
  the browser keeps in `localStorage` (so "New chat" is a *returning* session).
* **Retrieve at session start:** `POST /api/session?user_id=…` loads the profile; the greeting is built
  deterministically ("Welcome back, Ayesha! Last time you looked for wireless headphones. How can I help you today?")
  and the profile is added to the prompt. In voice mode the greeting is **spoken** when the mic is enabled.
* **Update during the conversation:** "my name is Ayesha and I prefer black headphones" → router →
  `customer_profile{action:update,name,preference}`. Names/contacts are taken from the text by regex and a
  preference is accepted only if its words were actually said (tests cover a model that invents a different name).
  Interactions (searched X, checked order Y, asked about returning Z) are recorded after each answer.
* **Privacy:** only volunteered data is stored; "forget my details" deletes it. See limitations for the security caveat.

### Mock backend
`data/brightcart.db` (created automatically at startup, or `python scripts/seed_db.py`): 15 corpus products
(ids and names read from the corpus) + a gift card, with **invented PKR prices and stock**, and 7 demo orders
(`BC1024`–`BC1030`, verification codes `4721, 3308, 9156, 2087, 6642, 1190, 5523`) covering shipped, delivered 10/45/3/5
days ago, processing, out-for-delivery, a gift card and a final-sale item. Dates are stored as "days ago" so the demo
never goes stale. Voice-friendly: `BC1024` can be said "B C one zero two four" or "B C ten twenty four".

### Measuring the LLM's tool routing
`data/tool_eval.json` holds 44 labelled utterances (search / order / return / profile / none).
`python scripts/eval_tools.py` reports tool-selection accuracy, argument accuracy, false-positive rate,
the share skipped by the gate, and router latency. *(paste)*

| tool accuracy | arg accuracy | false positives | gate-skipped | router ms mean / p95 |
|---|---|---|---|---|
| | | | | |

The gate and verifiers alone are guarded by a no-LLM test: with a perfect "oracle" router all 44 cases pass.

---

## 6. Voice pipeline (Assignment 3, Phases I–III)

**Input.** The browser captures the mic with an `AudioWorklet` (`static/mic-worklet.js`), downsamples to 16 kHz
mono PCM16 and sends 100 ms frames (3,200 bytes) as binary WebSocket messages. Moonshine returns `partial`
transcripts while the user speaks and a completed line at end of speech, which becomes the `final` transcript. It is
normalised for the domain ("B C ten twenty four" → `BC1024`, "ten thousand rupees" → `10000`) and passed to the
**same** conversation manager as typed text.

**Output.** As LLM tokens stream, `SpeechChunker` cuts speakable chunks (the first chunk flushes at a clause boundary
after ≥ 28 characters, later ones at sentence ends, never splitting "Rs. 9,500" or "3.5"). `speakable()` removes
`[1]` citations and markdown, spells out ids ("B C 1 0 2 4") and says "rupees". Piper synthesises each chunk in a
worker thread while the LLM keeps generating (test: first audio is sent **before** the `done` event). Audio goes back
as binary frames and the browser queues them gap-free, in order, while the text still streams into the chat.

**Half-duplex.** After a final transcript the server ignores mic audio until the client says `resume` (playback
finished) so Ava never hears herself; a safety timer resumes anyway. No barge-in.

**Voice style.** In voice mode the prompt asks for at most two short, plain sentences and caps generation at
`VOICE_NUM_PREDICT` tokens; long details (tracking numbers) go on screen.

**Concurrency.** One ASR stream and one `VoiceConnection` per user, a shared model; ASR feed and TTS run in worker
threads so the event loop never blocks; Piper synthesis is serialised by a lock (it is much faster than real time).
Tested with two simultaneous voice users. Generation concurrency is bounded by Ollama (`OLLAMA_NUM_PARALLEL`).

---

## 7. Latency (Phase V)

### What is measured, per voice turn (`voice_metrics` event, shown in the UI)
| Stage | Definition |
|---|---|
| `asr_ms` | last voiced audio frame → final transcript (includes Moonshine's end-of-speech wait) |
| `pre_generation_ms` | retrieval and tool routing run concurrently; the slower one (`retrieval_ms`, `tools_ms`, `router_ms` also given) |
| `llm_ttft_ms` | LLM request → first token (includes prompt processing) |
| `tts_first_audio_ms` | first speakable chunk handed to Piper → first audio frame sent |
| `first_audio_after_speech_end_ms` | **headline:** last voiced frame → first audio frame |
| `client_first_audio_ms` | same event measured independently in the benchmark client |

"Last voiced frame" is timestamped by a simple energy gate on the server (accurate to about ±100 ms), because the ASR
does its own segmentation. The 2-second target is measured from there, so the end-of-speech wait is **inside** it.

### Run it
```bash
python scripts/smoke_voice.py                                   # everything installed? models load? timings
python benchmark/benchmark_retrieval.py --json results/retrieval_latency.json   # retrieval alone
python benchmark/benchmark_voice.py --json results/voice_latency.json            # 1 user, all scenarios, real-time audio
python benchmark/benchmark_voice.py --users 2                                    # two concurrent users
python scripts/eval_tools.py --json results/tool_eval.json                       # router accuracy + latency
python benchmark/benchmark_latency.py                                            # LLM alone (Assignment 1)
```
`benchmark_voice.py` synthesises spoken dialogues (`data/voice_scenarios.json`: RAG policy, RAG product, product
search, order lookup, 4-turn return check, off-topic) with Piper and streams them in real time, or use your own
recordings with `--wav-dir`. *Spoken test audio is clean synthetic speech, so ASR accuracy is optimistic; use live
microphone runs too.*

### Results *(paste)* — hardware: CPU / cores / RAM / OS: ______
| turn type (n) | asr_ms | pre_generation_ms | llm_ttft_ms | tts_first_audio_ms | speech-end → first audio | client-side |
|---|---|---|---|---|---|---|
| rag | | | | | | |
| tool-question (no LLM) | | | | | | |
| tool-answer | | | | | | |
| 2 concurrent users | | | | | | |

Retrieval alone (Assignment 2 table): *(paste from `benchmark_retrieval.py`)*.

### What I expect (estimates, not measurements)
End of speech needs a short silence to be detected, so even an instant pipeline spends a few hundred ms there. The LLM
must then read a ~1–1.5k-token prompt and produce a first clause. On a laptop CPU I expect plain RAG turns near
**2–3 s**, tool turns **higher** (an extra router call before generation), and clarifying questions (no LLM call)
**fastest**. The assignment's 2 s target will probably be met only on faster hardware or for question-only turns.
Mitigations already built: concurrent retrieval + routing, gate (no router call on most turns), clause-level first
chunk, short voice prompts with a token cap, models kept loaded and warmed, stable prompt prefix for the LLM's cache.
If your numbers miss, the levers are `ASR_MODEL_ARCH=tiny-streaming`, a smaller `TOP_K`, `ROUTER_NUM_PREDICT`,
or a faster LLM. Report the breakdown honestly: it shows where the time goes.

---

## 8. Setup

Prerequisites: Python 3.11+, [Ollama](https://ollama.com), a microphone, Chrome/Edge/Firefox.

```bash
ollama serve &
ollama pull qwen2.5:1.5b-instruct-q4_K_M       # chat + tool router
ollama pull all-minilm                         # embeddings

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # includes moonshine-voice and piper-tts
python -m piper.download_voices en_US-lessac-medium --data-dir voices

python scripts/build_index.py                  # RAG index (re-run after editing data/corpus)
python scripts/smoke_voice.py                  # verifies Ollama, Piper, Moonshine (downloads its model once), constrained JSON
python scripts/eval_retrieval.py               # then set MIN_SCORE to the recommended value

uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **http://localhost:8000** (browsers allow the microphone on `localhost`; from another device you would need
HTTPS). Click **Speak**, allow the microphone, and talk. Typed chat works at the same time and shares the session.
The voice models load in the background (check `GET /api/health` → `voice_ready`); the text assistant works meanwhile.

Environment variables (all optional): `OLLAMA_MODEL`, `EMBED_MODEL`, `TOP_K`, `MIN_SCORE`, `ASR_MODEL_ARCH`
(`tiny-/small-/medium-streaming`), `ASR_UPDATE_INTERVAL`, `TTS_VOICE_PATH`, `TOOL_TIMEOUT`, `ROUTER_TIMEOUT`,
`TOOLS_ENABLED`, `DB_PATH`, `VOICE_NUM_PREDICT`, `INDEX_DIR`, `CORPUS_DIR` (see `app/config.py`).
For two simultaneous users start Ollama with `OLLAMA_NUM_PARALLEL=2`.

**Demo script:** *"find wireless headphones under ten thousand rupees"* · *"where is order B C one zero two four"* → "four seven
two one" · *"can I return the headphones from order B C one zero two five"* → unused → yes → "three three zero eight" ·
*"how many days do I have to return an item"* (RAG + citation) · *"my name is Ayesha and I prefer black headphones"*, then
**New chat** to be greeted by name · *"what's the weather"* (refused).

---

## 9. APIs

**Text (unchanged since Assignment 1):** `ws://…/ws/chat` — request `{"type":"message","session_id","text"}`; events
`stage`, `token`, `done`, `blocked`, `error`. `done` carries the original fields plus the optional additions
`sources`, `retrieval` (Assignment 2) and `tools`, `timing` (Assignment 3). Clients ignoring unknown fields are unaffected.

**Voice (added beside it):** `ws://…/ws/voice`
| Direction | Message |
|---|---|
| client → server | `{"type":"start","session_id"}` · binary mic frames (PCM16 mono 16 kHz, ~100 ms) · `{"type":"resume"}` |
| server → client (JSON) | `ready{sample_rate, greeting}` · `partial{text}` · `final{text}` · `stage` · `token` · `done` · `blocked` · `error` · `tts_done` · `voice_metrics{…}` |
| server → client (binary) | audio frame: `uint32 seq, uint32 sample_rate` (little-endian) + PCM16 mono |

**REST:** `GET /api/health` (adds `voice_ready`, `tools_enabled`, `voice_error`), `POST /api/session?user_id=…`
(optional; enables the CRM), `GET|DELETE /api/session/{id}`.

---

## 10. Failure handling

| Failure | Behaviour |
|---|---|
| Voice models missing / not yet loaded | `/ws/voice` returns a clear error (with the reason); text chat is unaffected |
| ASR error | reported **once** (not per 100 ms frame); connection stays open |
| TTS error | "showing text only" reported once; the text answer and turn still complete |
| LLM error mid-turn | error event; user turn not recorded; the voice connection survives and the next turn works |
| Router garbage / timeout | no tool is used; the question is answered from RAG |
| Tool timeout / crash / missing DB | plain "couldn't look that up right now" result, never a hang or a stack trace |
| Customer never answers a clarifying question | pending request dropped (or re-asked ≤ 3 times), then back to normal conversation |
| Wrong verification code / unknown order | identical neutral reply; lockout after 3 failures per session |
| Retrieval failure / no match / context overflow | see Assignment 2 behaviour (status, circuit breaker, priority budget); tool results replace excerpts |
| Echo / mic during assistant speech | ignored server-side until `resume`; muted client-side |
| Malformed message, unknown session, disconnect mid-stream | clean `error` event or clean drop |

---

## 11. Tests

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_rag.py` | 43 | chunking, incremental index, retriever (cache/timeout/breaker), context budget, guards |
| `tests/test_tools.py` | 24 | catalogue, order lookup + lockout, return policy matrix incl. day 30/31 boundary |
| `tests/test_orchestrator.py` | 26 | gate, routing, hallucination guards, slot-filling flows, every failure mode |
| `tests/test_manager_tools.py` | 10 | manager ↔ tools seam, voice prompt style, shared history |
| `tests/test_voice.py` | 18 | streaming order (audio before `done`), half-duplex, failures, two concurrent users, chunker |
| `tests/test_main_voice.py` | 9 | the real `/ws/voice` and `/ws/chat` handlers over a fake socket |
| `tests/test_tool_eval.py` | 3 | eval harness; gate + verifiers lose nothing (oracle router) |
| `tests/test_crm.py` | 16 | storage, cross-session greeting/recall, anti-hallucination, forget, voice |
| `tests/test_conversation_manager.py`, `tests/test_api.py` | 10 + 10 | Assignment 1/2 (the 10 API tests need real FastAPI) |
```bash
pytest -v        # or: python -m unittest discover -s tests -v
python tests/manual_stress_test.py       # live: concurrency, malformed input, mid-stream disconnect (server + Ollama running)
```

---

## 12. Known limitations
* **Unmeasured here:** real-model latency, ASR accuracy, and the 1.5B router's accuracy. A 1.5B model may mis-route
  ambiguous utterances; the gate, verifiers and deterministic slot-filling limit the damage, and `eval_tools.py` quantifies it.
* **Latency target:** see §7: tool turns pay for an extra LLM call and the 2 s goal is unlikely on a modest CPU.
* **Utterance splitting:** Moonshine segments speech itself; a long mid-sentence pause can end one turn and start another
  (no debounce/merge). **No barge-in** (half-duplex by design).
* **One tool per message:** "my name is Ayesha and find me headphones" runs one tool; the user may need to repeat the other part.
* **ASR and identifiers:** spoken order numbers/codes can be misheard; there is no read-back confirmation, only the 3-attempt lockout.
* **Mock data:** prices, stock and orders are invented demo data (PKR); statuses never change. The assistant cannot cancel, edit or refund.
* **CRM security:** the `user_id` is an unauthenticated random id in `localStorage`; anyone with it can read that profile.
  It stores only volunteered, low-sensitivity data in plaintext SQLite; this is a demo, not a production identity system.
* **Voice quality/licensing:** Piper voices are good, not human; Piper is GPL-3.0. Moonshine wheels exclude Intel Macs/macOS < 15.
* **First run needs internet** (Moonshine model, Piper voice). The microphone needs `localhost` or HTTPS.
* **Assignment 2 limitations still apply:** `MIN_SCORE` needs calibration; "related but unanswerable" questions rely on the
  model saying "I don't know"; citations show what was retrieved, not verified attribution; sessions/caches are in memory.
* **Keyword-based** stage and intent detection (Assignment 1) can mislabel a stage; it does not affect retrieval, tools or answers.

---

## 13. Where the prompts live (for the viva)
| Prompt | Location |
|---|---|
| System prompt, per-turn notes (`ANSWER_FROM_EXCERPTS_NOTE`, `NO_MATCH_NOTE`, `UNAVAILABLE_NOTE`, `TOOL_ANSWER_NOTE`, `VOICE_NOTE`) | `app/domain.py` |
| Tool router prompt + JSON schema | `app/orchestrator.py` (`ROUTER_PROMPT`, `ROUTER_SCHEMA`) |
| Per-turn prompt assembly and token budget | `ConversationManager._build_messages` |
| Summarisation prompt | `ConversationManager._maybe_summarize` |

## 14. Changes made to earlier assignments in this iteration
* System prompt: the "no access to live order/stock systems" rule is replaced by "state order/price/eligibility facts only
  from tool results" (the assistant still cannot cancel, edit, refund or take payments).
* Prompt assembly now accepts tool results and a voice flag; clarifying questions bypass the LLM; interactions are recorded after `done`.
* `/api/health` and `/api/session` gained optional fields; **the text WebSocket protocol is unchanged**.
* Defects fixed in Assignments 1–2 are listed in the earlier README history (static mount, substring off-topic filter,
  `total_time`, orphaned user turns).

## 15. Repository layout
```
app/                 main.py · conversation_manager.py · orchestrator.py · llm_engine.py · domain.py · config.py · schemas.py
app/rag/             chunking.py · embedder.py · store.py · retriever.py
app/tools/           registry.py · catalogue.py · orders.py · returns.py · crm.py · db.py · normalize.py · types.py
app/voice/           asr.py (Moonshine) · tts.py (Piper) · chunker.py · pipeline.py · audio.py
scripts/             build_index.py · eval_retrieval.py · seed_db.py · smoke_voice.py · eval_tools.py
benchmark/           benchmark_retrieval.py · benchmark_voice.py · benchmark_e2e.py · benchmark_latency.py
data/                corpus/ (50 docs) · eval_queries.json · tool_eval.json · voice_scenarios.json
static/              index.html · app.js · voice.js · mic-worklet.js · style.css
tests/               test_*.py · fakes.py · manual_stress_test.py
```
