"""
CRM (bonus): remembers customers across sessions, keyed by a browser-generated user id.

  * storage     SQLite table `crm_profiles` (name, contact, preferences, recent interactions)
  * LLM tool    `customer_profile` with action get | update | forget  (registered in registry.py)
  * session start   the manager loads the profile, the greeting names the customer and recalls
                    the last topic, and the profile is added to the prompt
  * every turn  the manager records what the customer did (tool use / topic) after the answer

Only data the customer volunteers is stored; "forget" deletes it.
"""
import asyncio
import json
import re
import time

from app.tools.db import connect
from app.tools.types import ToolContext, ToolResult

DDL = ("CREATE TABLE IF NOT EXISTS crm_profiles (user_id TEXT PRIMARY KEY, name TEXT, contact TEXT, "
       "preferences TEXT DEFAULT '[]', interactions TEXT DEFAULT '[]', first_seen REAL, last_seen REAL)")
MAX_INTERACTIONS, MAX_PREFS = 10, 8


def _con(db_path):
    con = connect(db_path)
    con.execute(DDL)
    return con


def load_profile(db_path, user_id: str) -> dict | None:
    if not user_id:
        return None
    with _con(db_path) as con:
        row = con.execute("SELECT * FROM crm_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    p = dict(row)
    p["preferences"], p["interactions"] = json.loads(p["preferences"]), json.loads(p["interactions"])
    return p


def save_fields(db_path, user_id: str, name=None, contact=None, preference=None) -> dict:
    now = time.time()
    p = load_profile(db_path, user_id) or {"user_id": user_id, "name": None, "contact": None, "preferences": [],
                                           "interactions": [], "first_seen": now}
    if name:
        p["name"] = name
    if contact:
        p["contact"] = contact
    if preference and preference.lower() not in [x.lower() for x in p["preferences"]]:
        p["preferences"] = (p["preferences"] + [preference])[-MAX_PREFS:]
    _write(db_path, p, now)
    return p


def record_interaction(db_path, user_id: str, topic: str) -> None:
    now = time.time()
    p = load_profile(db_path, user_id) or {"user_id": user_id, "name": None, "contact": None, "preferences": [],
                                           "interactions": [], "first_seen": now}
    p["interactions"] = (p["interactions"] + [{"t": now, "topic": topic}])[-MAX_INTERACTIONS:]
    _write(db_path, p, now)


def forget(db_path, user_id: str) -> None:
    with _con(db_path) as con:
        con.execute("DELETE FROM crm_profiles WHERE user_id = ?", (user_id,))


def _write(db_path, p: dict, now: float):
    with _con(db_path) as con:
        con.execute("INSERT OR REPLACE INTO crm_profiles VALUES (?,?,?,?,?,?,?)",
                    (p["user_id"], p["name"], p["contact"], json.dumps(p["preferences"]),
                     json.dumps(p["interactions"]), p["first_seen"], now))


def last_topic(profile: dict | None) -> str | None:
    return profile["interactions"][-1]["topic"] if profile and profile["interactions"] else None


def greeting_for(profile: dict | None) -> str | None:
    """Deterministic returning-customer greeting (no LLM call, so it is instant and never invented)."""
    if not profile or not (profile.get("name") or profile.get("interactions")):
        return None
    who, topic = profile.get("name"), last_topic(profile)
    hello = f"Welcome back, {who}!" if who else "Welcome back!"
    recall = f" Last time you {topic}." if topic else ""
    return f"{hello}{recall} How can I help you today?"


def profile_prompt(profile: dict | None) -> str | None:
    if not profile:
        return None
    bits = []
    if profile.get("name"):
        bits.append(f"name {profile['name']}")
    if profile.get("preferences"):
        bits.append("preferences: " + "; ".join(profile["preferences"]))
    if (t := last_topic(profile)):
        bits.append(f"last time they {t}")
    return ("Returning customer, details they shared earlier: " + ", ".join(bits) +
            ". Use their name naturally; do not recite these details unless relevant.") if bits else None


# ---------------------------------------------------------------- the LLM-callable tool
def _run(ctx: ToolContext, args: dict) -> ToolResult:
    uid = getattr(ctx.session, "user_id", None)
    if not uid:
        return ToolResult("ok", "No profile can be saved in this chat (anonymous session).")
    action = args.get("action") or ("update" if any(args.get(k) for k in ("name", "contact", "preference")) else "get")
    if action == "forget":
        forget(ctx.db_path, uid)
        ctx.session.profile = None
        return ToolResult("ok", "Deleted the customer's saved profile.")
    if action == "update":
        if not any(args.get(k) for k in ("name", "contact", "preference")):
            return ToolResult("needs_info", "Nothing to save.", missing=["preference"])
        p = save_fields(ctx.db_path, uid, args.get("name"), args.get("contact"), args.get("preference"))
        ctx.session.profile = p
        saved = [f"{k} {args[k]}" for k in ("name", "contact", "preference") if args.get(k)]
        return ToolResult("ok", "Saved to the customer's profile: " + "; ".join(saved) + ".")
    p = load_profile(ctx.db_path, uid)
    if not p or not (p["name"] or p["contact"] or p["preferences"] or p["interactions"]):
        return ToolResult("ok", "No profile is saved for this customer yet.")
    bits = [f"name {p['name']}" if p["name"] else None, f"contact {p['contact']}" if p["contact"] else None,
            "preferences: " + "; ".join(p["preferences"]) if p["preferences"] else None,
            f"last topic: {last_topic(p)}" if last_topic(p) else None]
    return ToolResult("ok", "Saved profile: " + ", ".join(b for b in bits if b) + ".")


async def customer_profile(args: dict, ctx: ToolContext) -> ToolResult:
    return await asyncio.to_thread(_run, ctx, args)


# ---------------------------------------------------------------- evidence extractors (code verifies)
_NAME = re.compile(r"\b(?:my name is|my name's|call me)\s+([A-Za-z]{2,20})(?:\s+([A-Za-z]{2,20}))?", re.I)
_NOT_SURNAME = {"and", "but", "i", "im", "i'm", "my", "the", "a", "an", "please", "so", "also", "can", "could", "is"}


def name_from_text(text: str) -> str | None:
    m = _NAME.search(text)
    if not m:
        return None
    parts = [m.group(1)] + ([m.group(2)] if m.group(2) and m.group(2).lower() not in _NOT_SURNAME else [])
    return " ".join(w.capitalize() for w in parts)


def contact_from_text(text: str) -> str | None:
    if (m := re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)):
        return m.group(0)
    if (m := re.search(r"(?<!\d)\+?\d{10,13}(?!\d)", text)):
        return m.group(0)
    return None


def action_from_text(text: str) -> str:
    t = text.lower()
    if re.search(r"forget (me|my)|delete my (data|details|profile|information)|remove my (data|details)", t):
        return "forget"
    if re.search(r"what do you (know|remember)|do you remember|what.s saved|what details", t):
        return "get"
    return "update"
