"""
Tool orchestrator (prompt-based function calling with constrained JSON output).

Per user message:
  1. GATE      cheap regex: could this message need a tool?  (no -> zero extra latency)
  2. ROUTE     one short LLM call, decoded against a JSON schema (Ollama structured
               outputs), picks a tool and extracts arguments.  "LLM proposes ..."
  3. VERIFY    ... code verifies: ids, codes, condition, packaging and prices are only
               accepted if they are actually present in what the customer said.
  4. SLOTS     missing required arguments are asked for DETERMINISTICALLY (no LLM call)
               and the pending request is kept on the session; the customer's next
               answer fills the slot by regex, again without an LLM call.
  5. EXECUTE   async handler under asyncio.wait_for; any failure becomes a sensible
               message instead of a hang or a crash.
  6. The result summary is handed to the conversation manager, which gives it to the
     LLM to phrase the final (streamed, spoken) answer.
"""
import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date

from app.config import DB_PATH, ROUTER_NUM_PREDICT, ROUTER_TIMEOUT, TOOL_TIMEOUT
from app.tools.crm import action_from_text, contact_from_text, name_from_text
from app.tools.db import CATEGORIES
from app.tools.normalize import canonical_order_ids, find_code, find_order_id
from app.tools.registry import TOOLS
from app.tools.types import ToolContext, ToolResult

logger = logging.getLogger("ecommerce-assistant.tools")

MAX_ASKS_PER_FIELD = 3

_GATE = re.compile(
    r"\b(order|orders|track|tracking|status|where is|where's|shipped|delivered|arrive|"
    r"return|returned|refund|exchange|eligible|eligibility|send (?:it )?back|"
    r"price|prices|cost|how much|cheap|cheapest|under|below|less than|budget|in stock|"
    r"available|availability|find|show me|looking for|search|recommend|do you (?:sell|have)|"
    r"BC\d{3,6}|my name is|my name's|call me|remember (?:that|my|me)|i prefer|my (?:email|phone|number) is|"
    r"forget (?:me|my)|delete my (?:data|details|profile)|what do you (?:know|remember)|do you remember)\b",
    re.IGNORECASE)

ROUTER_PROMPT = """You route customer messages for BrightCart support. Reply with ONE JSON object.
Tools:
- search_products: customer wants to find or compare products, prices or stock. Fields: query, category, max_price (rupees), in_stock_only.
- lookup_order: customer asks about the status or delivery of THEIR order. Fields: order_id, verification_code (4 digits).
- check_return_eligibility: customer asks if THEIR item from a specific order can be returned. Fields: order_id, item, condition (unused/used/damaged), has_original_packaging (true/false).
- customer_profile: customer shares personal details to remember (name, contact, preference), asks what is remembered, or asks to be forgotten. Fields: action (get/update/forget), name, contact, preference.
- none: policy questions, general product specs or compatibility, greetings, anything else.
Only include fields the customer actually said. Never guess.
Customer: find wireless headphones under 10000 rupees
{"tool":"search_products","query":"wireless headphones","max_price":10000}
Customer: what's happening with order BC1024
{"tool":"lookup_order","order_id":"BC1024"}
Customer: can I return the headphones from order BC1025, they are unused and in the box
{"tool":"check_return_eligibility","order_id":"BC1025","item":"headphones","condition":"unused","has_original_packaging":true}
Customer: my name is Ayesha and I prefer black headphones
{"tool":"customer_profile","action":"update","name":"Ayesha","preference":"black headphones"}
Customer: what is your return policy
{"tool":"none"}
Customer: does the H200 work with my laptop
{"tool":"none"}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": [*TOOLS, "none"]},
        "query": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "max_price": {"type": "number"},
        "in_stock_only": {"type": "boolean"},
        "order_id": {"type": "string"},
        "verification_code": {"type": "string"},
        "item": {"type": "string"},
        "condition": {"type": "string", "enum": ["unused", "used", "damaged"]},
        "has_original_packaging": {"type": "boolean"},
        "action": {"type": "string", "enum": ["get", "update", "forget"]},
        "name": {"type": "string"},
        "contact": {"type": "string"},
        "preference": {"type": "string"},
    },
    "required": ["tool"],
}


# ---------------------------------------------------------------- evidence extractors
def condition_from_text(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(unused|unopened|sealed|brand new|never (?:been )?(?:used|opened)|not (?:been )?used|"
                 r"haven'?t (?:used|opened)|still new)\b", t):
        return "unused"
    if re.search(r"damag|broke|crack|defect|faulty|not working|scratch|shatter", t):
        return "damaged"
    if re.search(r"\b(used|opened|worn|tried)\b", t):
        return "used"
    return None


def packaging_from_text(text: str, bare_yes_no: bool = False) -> bool | None:
    t = text.lower()
    if re.search(r"\b(no|without|don'?t have|do not have|threw|thrown|lost|discarded|missing)\b[^.]{0,30}"
                 r"\b(box|packaging|package|wrapping)\b", t):
        return False
    if re.search(r"\b(box|packaging|package|wrapping)\b", t) and re.search(
            r"\b(have|has|with|in|original|still|kept|intact|sealed|got)\b", t):
        return True
    if bare_yes_no:
        if re.match(r"\s*(yes|yeah|yep|yup|sure|i do|i have)\b", t):
            return True
        if re.match(r"\s*(no|nope|nah|i don'?t|i do not)\b", t):
            return False
    return None


def price_from_text(text: str) -> float | None:
    m = re.search(r"(?:under|below|less than|within|up ?to|max(?:imum)?|budget of|around)\s*(?:rs\.?|pkr)?\s*"
                  r"(\d[\d,]*)", text, re.IGNORECASE)
    return float(m.group(1).replace(",", "")) if m else None


_FILLER = {"the", "my", "a", "an", "it", "its", "it's", "is", "i", "want", "to", "return", "one", "please",
           "from", "that", "this", "its", "would", "like", "about", "item", "yes", "yeah", "the"}


def item_from_text(text: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", text) if w.lower() not in _FILLER]
    return " ".join(words)


# ---------------------------------------------------------------------- data classes
@dataclass
class ToolCall:
    name: str
    args: dict
    status: str
    ms: float
    summary: str


@dataclass
class Outcome:
    kind: str = "none"                 # none | ask | result
    ask_text: str = ""
    result_text: str = ""              # given to the LLM when kind == "result"
    calls: list[ToolCall] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    router_ms: float = 0.0
    tools_ms: float = 0.0              # total orchestrator time (router + tool)
    routed_by_llm: bool = False


class ToolOrchestrator:
    def __init__(self, engine, db_path=DB_PATH, today=date.today, tool_timeout=TOOL_TIMEOUT,
                 router_timeout=ROUTER_TIMEOUT):
        self.engine, self.db_path, self.today = engine, db_path, today
        self.tool_timeout, self.router_timeout = tool_timeout, router_timeout
        self._route_cache: OrderedDict[str, dict] = OrderedDict()

    # ------------------------------------------------------------------ public
    async def run(self, session, text: str) -> Outcome:
        t0 = time.perf_counter()
        try:
            out = await self._run(session, canonical_order_ids(text))
        except Exception:  # noqa: BLE001 - the orchestrator must never break a turn
            logger.exception("orchestrator failed; continuing without tools")
            session.pending_tool = None
            out = Outcome()
        out.tools_ms = round((time.perf_counter() - t0) * 1000, 1)
        return out

    # ----------------------------------------------------------------- internals
    async def _run(self, session, text: str) -> Outcome:
        out = Outcome()
        pending = session.pending_tool
        tool, args, asked = None, {}, []

        if pending:                                    # answer to our own question? (no LLM call)
            tool, args, asked = TOOLS[pending["tool"]], dict(pending["args"]), list(pending["missing"])
            if not self._fill_from_text(tool, args, text, asked):
                session.pending_tool, pending, tool, args = None, None, None, {}

        if tool is None:
            if not _GATE.search(text):
                return out
            t = time.perf_counter()
            proposal = await self._route(text)
            out.router_ms, out.routed_by_llm = round((time.perf_counter() - t) * 1000, 1), True
            if not proposal or proposal.get("tool") not in TOOLS:
                return out
            tool = TOOLS[proposal["tool"]]
            args = self._verified_args(tool, proposal, text)
            self._fill_from_text(tool, args, text, [])

        return await self._execute_or_ask(session, tool, args, out)

    async def _execute_or_ask(self, session, tool, args: dict, out: Outcome) -> Outcome:
        missing = [f for f in tool.required if args.get(f) in (None, "")]
        if missing:
            return self._ask(session, tool, args, missing[0], out)

        ctx = ToolContext(session, self.db_path, self.today())
        t = time.perf_counter()
        try:
            res = await asyncio.wait_for(tool.handler(args, ctx), timeout=self.tool_timeout)
        except asyncio.TimeoutError:
            res = ToolResult("error", f"The {tool.name} lookup timed out.")
        except Exception as e:  # noqa: BLE001
            logger.warning("tool %s failed: %s", tool.name, e)
            res = ToolResult("error", f"The {tool.name} lookup failed.")
        ms = round((time.perf_counter() - t) * 1000, 1)
        out.calls.append(ToolCall(tool.name, {k: v for k, v in args.items() if k != "verification_code"},
                                  res.status, ms, res.summary))

        if res.status == "needs_verification":
            return self._ask(session, tool, args, "verification_code", out)
        if res.status == "needs_info":
            if res.data.get("options"):
                return self._ask(session, tool, {**args, "item": ""}, "item", out,
                                 text=f"Which one do you mean: {' or '.join(res.data['options'])}?")
            return self._ask(session, tool, args, (res.missing or ["query"])[0], out)

        session.pending_tool = None
        out.kind = "result"
        out.sources = res.sources
        if res.status == "error":
            out.result_text = (f"Tool {tool.name} FAILED ({res.summary}). Tell the customer you could not look "
                               "that up right now and suggest trying again shortly or contacting support.")
        else:
            out.result_text = f"Tool {tool.name} returned: {res.summary}"
        return out

    def _ask(self, session, tool, args: dict, field_name: str, out: Outcome, text: str | None = None) -> Outcome:
        asks = dict(session.pending_tool["asks"]) if session.pending_tool else {}
        asks[field_name] = asks.get(field_name, 0) + 1
        if asks[field_name] > MAX_ASKS_PER_FIELD:       # customer is not answering: stop looping
            session.pending_tool = None
            return Outcome(router_ms=out.router_ms, routed_by_llm=out.routed_by_llm)
        session.pending_tool = {"tool": tool.name, "args": args, "missing": [field_name], "asks": asks}
        out.kind = "ask"
        out.ask_text = text or tool.questions.get(field_name, "Could you give me a bit more detail?")
        return out

    # ---- routing (LLM) ----------------------------------------------------------
    async def _route(self, text: str) -> dict | None:
        key = " ".join(text.lower().split())
        if key in self._route_cache:
            self._route_cache.move_to_end(key)
            return self._route_cache[key]
        messages = [{"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": f"Customer: {text}"}]
        try:
            raw = await asyncio.wait_for(
                self.engine.chat_once(messages, options={"temperature": 0, "num_predict": ROUTER_NUM_PREDICT},
                                      format=ROUTER_SCHEMA),
                timeout=self.router_timeout)
            proposal = json.loads(raw)
            if not isinstance(proposal, dict):
                raise ValueError("router returned non-object")
        except Exception as e:  # noqa: BLE001 timeouts, bad JSON, engine errors -> no tool, normal answer
            logger.warning("tool router failed (%s); answering without tools", e)
            return None
        self._route_cache[key] = proposal
        if len(self._route_cache) > 128:
            self._route_cache.popitem(last=False)
        return proposal

    # ---- "code verifies" ----------------------------------------------------------
    def _verified_args(self, tool, proposal: dict, text: str) -> dict:
        """Keep LLM-proposed free-text fields; take everything checkable from the text itself."""
        props, args = tool.parameters["properties"], {}
        for f in ("query", "category", "in_stock_only", "item"):
            if f in props and proposal.get(f) not in (None, ""):
                args[f] = proposal[f]
        if "in_stock_only" in args and not re.search(r"in stock|available|availability", text, re.I):
            args.pop("in_stock_only")                      # don't let the model invent a filter
        if "max_price" in props:
            price = price_from_text(text)
            if price is not None:
                args["max_price"] = price
        if "condition" in props and (c := condition_from_text(text)):
            args["condition"] = c
        if "has_original_packaging" in props and (p := packaging_from_text(text)) is not None:
            args["has_original_packaging"] = p
        if tool.name == "customer_profile":               # profile fields only if the customer really said them
            args["action"] = action_from_text(text)
            if n := name_from_text(text):
                args["name"] = n
            if c := contact_from_text(text):
                args["contact"] = c
            pref = str(proposal.get("preference") or "").strip()
            words = [w for w in re.findall(r"[a-z]{4,}", pref.lower())]
            if pref and words and all(w in text.lower() for w in words):
                args["preference"] = pref
        return args

    def _fill_from_text(self, tool, args: dict, text: str, asked: list[str]) -> bool:
        """Regex slot filling. Returns True if it supplied anything."""
        filled = False
        want = set(tool.required) | {"verification_code"}
        if (oid := find_order_id(text)) and "order_id" in want:
            filled |= args.get("order_id") != oid
            args["order_id"] = oid
        if "verification_code" in want and not args.get("verification_code") and (code := find_code(text)):
            args["verification_code"], filled = code, True
        if "condition" in want and not args.get("condition") and (c := condition_from_text(text)):
            args["condition"], filled = c, True
        if "has_original_packaging" in want and "has_original_packaging" not in args:
            p = packaging_from_text(text, bare_yes_no="has_original_packaging" in asked)
            if p is not None:
                args["has_original_packaging"], filled = p, True
        if "item" in asked and (item := item_from_text(text)):
            args["item"], filled = item, True
        if "query" in asked and text.strip():
            args["query"], filled = text.strip(), True
        if "preference" in asked and text.strip():
            args["preference"], filled = text.strip(), True
        return filled
