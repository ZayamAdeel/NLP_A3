"""
Tool 3 - check_return_eligibility: applies BrightCart's written return policy
(corpus: returns_033 eligibility, returns_037 damaged items, returns_038 non-returnable)
to a specific, verified order item. The date arithmetic and rule combination are done in
code, because an LLM cannot be trusted to count days or apply every condition.
"""
import asyncio
import re

from app.config import RETURN_WINDOW_DAYS
from app.tools.db import connect
from app.tools.orders import norm_order_id, verify
from app.tools.types import ToolContext, ToolResult

POLICY = "returns/returns_033_eligibility.md"


def evaluate(status: str, days_since: int | None, non_returnable: bool, condition: str,
             packaging: bool, window: int = RETURN_WINDOW_DAYS) -> tuple:
    """Pure policy function -> (eligible: True | False | 'review', reason, policy_doc)."""
    if status != "delivered" or days_since is None:
        return False, "not_delivered", "The order has not been delivered yet, so it cannot be returned yet.", POLICY
    if non_returnable:
        return (False, "non_returnable", "This item is a gift card or final-sale item, and those cannot "
                "be returned.", "returns/returns_038_nonreturnable.md")
    if condition == "damaged":
        return ("review", "damaged", "Damaged items are not an automatic return. The customer should keep the "
                "item and packaging, take photos, and contact support for a damaged-item review. No refund "
                "can be promised yet.", "returns/returns_037_damaged_item.md")
    if days_since > window:
        return (False, "window_expired", f"It was delivered {days_since} days ago, which is "
                f"{days_since - window} day(s) past the {window}-day return window.", POLICY)
    if condition == "used":
        return False, "used", "Returns require an unused item, and this one has been used.", POLICY
    if not packaging:
        return False, "no_packaging", "Returns require the original packaging.", POLICY
    return True, "eligible", (f"Delivered {days_since} days ago, so {window - days_since} day(s) remain in "
                              f"the {window}-day window; the item is unused and has its original packaging."), POLICY


def _tok(text: str) -> set:
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.findall(r"[a-z0-9]+", text.lower())}


def _match_items(items, wanted: str):
    """Items whose name/id contains every word the customer said (fallback: any word)."""
    w = _tok(wanted or "") - {"the", "my", "a", "an", "from", "order", "item"}
    if not w:
        return []
    full = [it for it in items if w <= _tok(f"{it['name']} {it['product_id']}")]
    return full or [it for it in items if w & _tok(f"{it['name']} {it['product_id']}")]


def _run(ctx: ToolContext, args: dict) -> ToolResult:
    order_id = norm_order_id(args.get("order_id", ""))
    if not order_id:
        return ToolResult("not_found", "I couldn't find an order matching those details.")
    if order_id not in ctx.session.verified_orders:
        if not args.get("verification_code"):
            return ToolResult("needs_verification", "Order not verified yet.", missing=["verification_code"])
        _, err = verify(ctx, order_id, args["verification_code"])
        if err:
            return err
    with connect(ctx.db_path) as con:
        order = con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        items = con.execute(
            "SELECT oi.product_id, oi.final_sale, p.name, p.non_returnable FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id WHERE oi.order_id = ?", (order_id,)).fetchall()
    hits = _match_items(items, args.get("item", ""))
    if not hits:
        names = ", ".join(i["name"] for i in items)
        return ToolResult("not_found", f"No item matching '{args.get('item')}' is on order {order_id}. "
                                       f"The order contains: {names}.")
    if len(hits) > 1:
        names = [i["name"] for i in hits]
        return ToolResult("needs_info", f"More than one item matches: {' or '.join(names)}.",
                          {"options": names}, missing=["item"])
    it = hits[0]
    eligible, reason, text, doc = evaluate(
        order["status"], order["delivered_days_ago"], bool(it["non_returnable"] or it["final_sale"]),
        args.get("condition"), bool(args.get("has_original_packaging")))
    verdict = {True: "ELIGIBLE", False: "NOT ELIGIBLE", "review": "NEEDS REVIEW"}[eligible]
    return ToolResult("ok", f"Return check for {it['name']} on order {order_id}: {verdict}. {text}",
                      {"eligible": eligible, "reason": reason, "order_id": order_id, "product_id": it["product_id"]},
                      sources=[{"doc_id": doc, "title": "Return policy", "category": "returns", "snippet": text}])


async def check_return_eligibility(args: dict, ctx: ToolContext) -> ToolResult:
    return await asyncio.to_thread(_run, ctx, args)
