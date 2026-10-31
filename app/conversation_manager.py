"""
Conversation Manager: session handling, history management, prompt
orchestration, turn-taking logic and (Assignment 2) retrieval integration.
It sits between the WebSocket layer, the Retriever and the LLM engine.

Per-turn pipeline:
  off-topic guard -> stage update -> retrieve -> budgeted prompt build ->
  stream generation -> record turn -> `done` (+sources) -> maybe summarize
"""
import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field

from app.config import (
    CHARS_PER_TOKEN_ESTIMATE,
    GENERATION_OPTIONS,
    MAX_TURNS_IN_WINDOW,
    RAG_MAX_CONTEXT_TOKENS,
    VOICE_NUM_PREDICT,
    RESPONSE_RESERVE_TOKENS,
    SUMMARY_TRIGGER_TURNS,
)
from app.domain import (
    ANSWER_FROM_EXCERPTS_NOTE,
    BASE_SYSTEM_PROMPT,
    GREETING_MESSAGE,
    INTENT_KEYWORDS,
    NO_MATCH_NOTE,
    OFF_TOPIC_KEYWORDS,
    OFF_TOPIC_REDIRECT,
    STAGE_CLOSING,
    STAGE_CONFIRMATION,
    STAGE_GREETING,
    STAGE_GUIDANCE,
    STAGE_INFO_GATHERING,
    STAGE_INTENT_ID,
    STAGE_RESOLUTION,
    TOOL_ANSWER_NOTE,
    UNAVAILABLE_NOTE,
    VOICE_NOTE,
)
from app.llm_engine import OllamaEngine
from app.orchestrator import Outcome, ToolOrchestrator
from app.rag.retriever import Hit, RetrievalResult, Retriever
from app.tools import crm

CLOSING_TRIGGERS = ["bye", "goodbye", "thank you, that's all", "thats all", "no that's all",
                     "nothing else", "that's everything", "thanks that's all"]
CONFIRM_POSITIVE = ["yes", "yep", "yeah", "sure", "please", "go ahead"]
CONFIRM_NEGATIVE = ["no", "nope", "not really", "that's all", "nothing else", "no thanks"]

# Tokens reserved for the per-turn note + labels when budgeting the prompt.
NOTE_RESERVE_TOKENS = 150
# Messages shorter than this are treated as follow-ups ("yes please", "what about silver?")
# and get the previous user message prepended to the retrieval query.
SHORT_FOLLOWUP_WORDS = 6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def _detect_intent(message: str) -> str | None:
    lower = message.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# Whole-word matching: plain substring matching wrongly blocked real support
# questions ("investigate" contains "invest"; "contact assistance" contains "act as").
_OFF_TOPIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in OFF_TOPIC_KEYWORDS) + r")\b", re.IGNORECASE)


def is_off_topic(message: str) -> bool:
    """Fast, deterministic first-pass filter. The system prompt handles
    subtler cases the model itself must catch."""
    return bool(_OFF_TOPIC_RE.search(message))


# Messages made ONLY of these words need no knowledge-base lookup.
_SMALLTALK_WORDS = {
    "hi", "hello", "hey", "thanks", "thank", "thx", "you", "ok", "okay", "cool", "great",
    "awesome", "perfect", "nice", "bye", "goodbye", "good", "morning", "afternoon", "evening",
    "see", "later", "so", "much", "very", "ava", "there", "that's", "thats", "all",
    "nothing", "else", "no",
}


def is_smalltalk(message: str) -> bool:
    words = re.findall(r"[a-z']+", message.lower())
    return 0 < len(words) <= 6 and all(w in _SMALLTALK_WORDS for w in words)


def pack_excerpts(hits: list[Hit], budget_tokens: int) -> list[tuple[int, Hit, str]]:
    """
    Fit retrieved chunks into `budget_tokens`, best-ranked first, never splitting a
    chunk mid-way except when even the best one is too big (then it is cut at a
    sentence boundary). Returns [(number, hit, rendered_block)].
    """
    packed, used = [], 0
    for hit in hits:
        block = f"[{len(packed) + 1}] {hit.text}"
        cost = estimate_tokens(block)
        if used + cost <= budget_tokens:
            packed.append((len(packed) + 1, hit, block))
            used += cost
        elif not packed and budget_tokens > 20:
            max_chars = int(budget_tokens * CHARS_PER_TOKEN_ESTIMATE)
            cut = block[:max_chars]
            if ". " in cut:
                cut = cut[: cut.rfind(". ") + 1]
            packed.append((1, hit, cut.rstrip() + " [truncated]"))
            break
        else:
            break  # keep rank order: stop at the first chunk that does not fit
    return packed


_INTENT_TOPIC = {"order_tracking": "asked about order tracking",
                 "returns_shipping": "asked about returns or shipping",
                 "product_question": "asked a product question"}


def _topic_of(outcome, session, user_message: str) -> str | None:
    """A short, human phrase for 'what they did', used to recall the last visit."""
    for c in outcome.calls:
        a = c.args
        if c.name == "search_products":
            return f"looked for {a.get('query') or a.get('category') or 'products'}"
        if c.name == "lookup_order" and c.status == "ok":
            return f"checked order {a.get('order_id')}"
        if c.name == "check_return_eligibility" and c.status == "ok":
            return f"asked about returning the {a.get('item')} from order {a.get('order_id')}"
    if outcome.calls or is_smalltalk(user_message):
        return None
    return _INTENT_TOPIC.get(session.active_intent or "")


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    stage: str = STAGE_GREETING
    active_intent: str | None = None
    history: list[Turn] = field(default_factory=list)
    rolling_summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0
    # tool state (privileged: only the orchestrator/tools write these, never the LLM)
    verified_orders: set = field(default_factory=set)
    failed_verifications: int = 0
    pending_tool: dict | None = None
    # CRM (bonus): who this is, and what we already know about them
    user_id: str | None = None
    profile: dict | None = None

    def touch(self):
        self.last_active = time.time()


class ConversationManager:
    """
    Owns all sessions and drives the domain-specific conversation flow:
    greeting -> intent identification -> info gathering -> resolution ->
    confirmation -> closing, with support for the user changing topic at
    any point (which resets the active_intent and drops back into
    info_gathering for the new topic without losing the message history).
    """

    def __init__(self, engine: OllamaEngine | None = None, retriever: Retriever | None = None,
                 orchestrator: ToolOrchestrator | None = None, crm_db=None):
        self.sessions: dict[str, Session] = {}
        self.crm_db = crm_db              # path of the SQLite file holding crm_profiles; None => CRM off
        self.engine = engine or OllamaEngine()
        self.retriever = retriever        # None => no RAG
        self.orchestrator = orchestrator  # None => no tools

    # ---- Session lifecycle --------------------------------------------------
    def create_session(self, user_id: str | None = None) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        if user_id and self.crm_db:
            session.user_id = user_id
            try:
                session.profile = crm.load_profile(self.crm_db, user_id)
            except Exception:  # noqa: BLE001 - a broken CRM must never block a chat
                session.profile = None
        self.sessions[session_id] = session
        return session

    def greeting(self, session: Session) -> str:
        """Personalised for returning customers, the standard greeting otherwise."""
        return crm.greeting_for(session.profile) or GREETING_MESSAGE

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def reset_session(self, session_id: str) -> Session:
        self.sessions.pop(session_id, None)
        new_session = Session(session_id=session_id)
        self.sessions[session_id] = new_session
        return new_session

    def end_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ---- Turn-taking / stage logic -------------------------------------------
    def _advance_stage(self, session: Session, user_message: str):
        lower = user_message.lower().strip()

        detected_intent = _detect_intent(user_message)

        # Topic switch mid-conversation: user asks about a different supported
        # intent than the one currently being handled. We keep the history
        # (for context/faithfulness) but reset the slot-filling stage so the
        # assistant gathers whatever info the NEW topic needs, rather than
        # continuing to press for the old topic's info.
        if (
            detected_intent
            and session.active_intent
            and detected_intent != session.active_intent
            and session.stage in (STAGE_INFO_GATHERING, STAGE_RESOLUTION, STAGE_CONFIRMATION)
        ):
            session.active_intent = detected_intent
            session.stage = STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_GREETING:
            session.active_intent = detected_intent
            session.stage = STAGE_INTENT_ID if not detected_intent else STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_INTENT_ID:
            session.active_intent = detected_intent or session.active_intent
            session.stage = STAGE_INFO_GATHERING
            return

        if session.stage == STAGE_INFO_GATHERING:
            session.stage = STAGE_RESOLUTION
            return

        if session.stage == STAGE_RESOLUTION:
            session.stage = STAGE_CONFIRMATION
            return

        if session.stage == STAGE_CONFIRMATION:
            if any(t in lower for t in CLOSING_TRIGGERS) or any(
                lower == n for n in CONFIRM_NEGATIVE
            ):
                session.stage = STAGE_CLOSING
            elif detected_intent and detected_intent != session.active_intent:
                session.active_intent = detected_intent
                session.stage = STAGE_INFO_GATHERING
            else:
                # Customer has a follow-up on the same topic.
                session.stage = STAGE_RESOLUTION
            return

        if session.stage == STAGE_CLOSING:
            # A closed conversation can always reopen if the user keeps talking.
            if detected_intent:
                session.active_intent = detected_intent
                session.stage = STAGE_INFO_GATHERING
            else:
                session.stage = STAGE_GREETING
            return

    # ---- Retrieval ------------------------------------------------------------
    def _retrieval_query(self, session: Session, user_message: str) -> str:
        """Short follow-ups ("yes please", "what about silver?") are meaningless alone,
        so prepend the previous user message to give the embedding its topic."""
        if len(user_message.split()) < SHORT_FOLLOWUP_WORDS:
            # Anchor on the latest *substantive* user message, so a chain of short
            # follow-ups ("yes please" -> "and exchanges?") keeps the original topic.
            user_turns = [t.content for t in reversed(session.history) if t.role == "user"]
            anchor = next((c for c in user_turns if len(c.split()) >= SHORT_FOLLOWUP_WORDS),
                          user_turns[0] if user_turns else "")
            if anchor:
                return f"{anchor} {user_message}"
        return user_message

    async def _retrieve(self, session: Session, user_message: str) -> RetrievalResult:
        if self.retriever is None:
            return RetrievalResult("disabled")
        if is_smalltalk(user_message):
            return RetrievalResult("skipped")
        return await self.retriever.retrieve(self._retrieval_query(session, user_message))

    # ---- Prompt construction + context-window budget --------------------------
    def _build_messages(self, session: Session, user_message: str, result: RetrievalResult,
                        tool_text: str | None = None, voice: bool = False):
        """
        Build the exact `messages` list for this turn within the context window.

        Budget = num_ctx - RESPONSE_RESERVE_TOKENS. Priority when space is short:
          1. system prompt (+ rolling summary)        always kept
          2. the current customer message             always kept
          3. retrieved excerpts, capped at RAG_MAX_CONTEXT_TOKENS, best-ranked first
          4. recent history, newest first -- the OLDEST turns are dropped first

        Retrieved text is injected into the final user message only (never stored in
        history), so excerpts from old turns cannot pile up, and the static system
        prompt + history stay a stable prefix the LLM server can reuse from its cache.
        """
        budget = GENERATION_OPTIONS["num_ctx"] - RESPONSE_RESERVE_TOKENS
        messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
        if session.rolling_summary:
            messages.append({"role": "system",
                             "content": f"Summary of the earlier conversation: {session.rolling_summary}"})

        if (known := crm.profile_prompt(session.profile)):
            messages.append({"role": "system", "content": known})

        fixed = (sum(estimate_tokens(m["content"]) for m in messages)
                 + estimate_tokens(user_message) + estimate_tokens(tool_text or "") + NOTE_RESERVE_TOKENS)
        remaining = max(0, budget - fixed)

        packed = []
        if result.status == "ok" and not tool_text:      # an authoritative tool result replaces excerpts
            packed = pack_excerpts(result.hits, min(RAG_MAX_CONTEXT_TOKENS, remaining))
        remaining -= sum(estimate_tokens(block) for _, _, block in packed)

        kept = []
        for turn in reversed(session.history[-(MAX_TURNS_IN_WINDOW * 2):]):
            cost = estimate_tokens(turn.content)
            if cost > remaining:
                break
            kept.append(turn)
            remaining -= cost
        kept.reverse()
        while kept and kept[0].role == "assistant":
            kept.pop(0)  # never start the visible history mid-exchange
        messages += [{"role": t.role, "content": t.content} for t in kept]

        if tool_text:
            note, body = TOOL_ANSWER_NOTE, f"{tool_text}\n\n"
        elif packed:
            note = ANSWER_FROM_EXCERPTS_NOTE
            body = "Knowledge base excerpts:\n" + "\n\n".join(b for _, _, b in packed) + "\n\n"
        elif result.status in ("ok", "no_match"):
            note, body = NO_MATCH_NOTE, ""
        elif result.status == "unavailable":
            note, body = UNAVAILABLE_NOTE, ""
        else:  # skipped / disabled: plain conversation, use the stage guidance
            note, body = f"[Stage: {session.stage}. {STAGE_GUIDANCE.get(session.stage, '')}]", ""
        if voice:
            note += VOICE_NOTE
        messages.append({"role": "user", "content": f"{body}{note}\n\nCustomer message: {user_message}"})
        return messages, packed

    async def _maybe_summarize(self, session: Session):
        """
        When the raw history grows past the window, fold the turns that are
        about to fall out of the window into a running summary using a
        single extra call to the SAME local model (plain prompting -- not an
        external tool/agent/RAG). This keeps memory usage/prompt size flat
        for arbitrarily long sessions while still preserving gist-level
        context from earlier in the conversation.
        """
        if len(session.history) < SUMMARY_TRIGGER_TURNS * 2:
            return

        cutoff = len(session.history) - (MAX_TURNS_IN_WINDOW * 2)
        to_fold = session.history[:cutoff]
        if not to_fold:
            return

        transcript = "\n".join(f"{t.role}: {t.content}" for t in to_fold)
        summarizer_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following customer support exchange in 2-3 sentences, "
                    "keeping any concrete facts (order numbers, product names, decisions "
                    "made). Be terse."
                ),
            },
            {"role": "user", "content": transcript},
        ]
        try:
            summary = await self.engine.chat_once(summarizer_messages)
            if session.rolling_summary:
                session.rolling_summary = (session.rolling_summary + " " + summary).strip()
            else:
                session.rolling_summary = summary.strip()
        except Exception:
            # Summarization is a best-effort optimization; if it fails we
            # simply keep the window as-is rather than crashing the turn.
            pass
        finally:
            session.history = session.history[cutoff:]

    # ---- Public API used by the WebSocket routes (text AND voice share this) -----
    async def handle_user_message(self, session: Session, user_message: str, voice: bool = False):
        """
        Async generator. Yields (event_type, payload) tuples:
          ("stage", stage_name)
          ("token", text_chunk)
          ("done", metrics_dict)          includes `sources`, `retrieval`, `tools`, `timing`
          ("blocked", redirect_message)   -- off-topic, no retrieval / tool / LLM call made
        `voice=True` only changes the prompt style (short, speakable) and the answer length cap.
        """
        session.touch()
        session.turn_count += 1

        if is_off_topic(user_message):
            session.history.append(Turn("user", user_message))
            session.history.append(Turn("assistant", OFF_TOPIC_REDIRECT))
            yield ("stage", session.stage)
            yield ("blocked", OFF_TOPIC_REDIRECT)
            return

        self._advance_stage(session, user_message)
        yield ("stage", session.stage)

        # Retrieval and tool routing run concurrently; tool routing is the slower one.
        retrieval_task = asyncio.create_task(self._retrieve(session, user_message))
        outcome = (await self.orchestrator.run(session, user_message)
                   if self.orchestrator else Outcome())
        if outcome.kind != "none":
            retrieval_task.cancel()                    # a tool answer replaces document excerpts
            result = RetrievalResult("skipped")
        else:
            result = await retrieval_task

        full_response, final_metrics, packed = "", None, []
        if outcome.kind == "ask":                      # deterministic clarifying question: no LLM call
            for piece in re.findall(r"\S+\s*", outcome.ask_text):
                full_response += piece
                yield ("token", piece)
        else:
            messages, packed = self._build_messages(
                session, user_message, result,
                tool_text=outcome.result_text if outcome.kind == "result" else None, voice=voice)
            options = {"num_predict": VOICE_NUM_PREDICT} if voice else None
            async for token, metrics in self.engine.stream_chat(messages, options=options):
                full_response += token
                final_metrics = metrics
                yield ("token", token)

        # Record the exchange only after a successful generation, so an LLM error
        # does not leave an orphaned user turn in the history.
        session.history.append(Turn("user", user_message))
        session.history.append(Turn("assistant", full_response))

        sources = [{"n": n, "title": h.title, "doc_id": h.doc_id, "category": h.category,
                    "score": h.score, "snippet": h.text.split("\n", 1)[-1][:300]} for n, h, _ in packed]
        sources += [{"n": len(sources) + i, "title": s["title"], "doc_id": s["doc_id"], "category": s["category"],
                     "score": 1.0, "snippet": s["snippet"][:300]} for i, s in enumerate(outcome.sources, 1)]
        yield (
            "done",
            {
                "time_to_first_token": final_metrics.time_to_first_token if final_metrics else None,
                "total_time": final_metrics.total_time if final_metrics else None,
                "tokens_per_second": final_metrics.tokens_per_second if final_metrics else None,
                "stage": session.stage,
                "sources": sources,
                "retrieval": {"status": result.status, "ms": result.ms, "cached": result.cached},
                "tools": [{"name": c.name, "args": c.args, "status": c.status, "ms": c.ms,
                           "summary": c.summary} for c in outcome.calls],
                "timing": {"retrieval_ms": result.ms, "tools_ms": outcome.tools_ms,
                           "router_ms": outcome.router_ms, "asked": outcome.kind == "ask"},
            },
        )

        # Housekeeping AFTER the answer is complete, so none of it delays the user's response.
        if self.crm_db and session.user_id and (topic := _topic_of(outcome, session, user_message)):
            try:
                await asyncio.to_thread(crm.record_interaction, self.crm_db, session.user_id, topic)
            except Exception:  # noqa: BLE001
                pass
        await self._maybe_summarize(session)
