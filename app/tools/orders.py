"""
Tool 2 - lookup_order: order status from the mock order database.

Security behaviour: a wrong code and an unknown order produce the SAME reply (so order
ids cannot be probed), and after MAX_VERIFY_ATTEMPTS failures in a session the tool locks.
A verified order is remembered on the session so follow-up tools do not re-ask.
"""
import asyncio
import re
from datetime import timedelta

from app.config import MAX_VERIFY_ATTEMPTS
from app.tools.db import connect
from app.tools.normalize import canonical_order_ids
from app.tools.types import ToolContext, ToolResult

_NO_MATCH = "I couldn't find an order matching those details."


def norm_order_id(raw: str) -> str:
    m = re.search(r"BC\d{3,6}", canonical_order_ids(str(raw or "")).upper().replace(" ", ""))
    return m.group(0) if m else ""


def verify(ctx: ToolContext, order_id: str, code: str):
    """Return (row, None) when verified, else (None, ToolResult explaining why)."""
    s = ctx.session
    if s.failed_verifications >= MAX_VERIFY_ATTEMPTS:
        return None, ToolResult("locked", "Too many failed verification attempts in this chat. "
                                          "Order details are locked; the customer should contact BrightCart support.")
    with connect(ctx.db_path) as con:
        row = con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if row is None or str(code or "").strip() != row["phone_last4"]:
        s.failed_verifications += 1
        return None, ToolResult("not_found", _NO_MATCH)
    s.verified_orders.add(order_id)
    return row, None


def _fmt_day(ctx: ToolContext, offset: int) -> str:
    d = ctx.today + timedelta(days=offset)
    return d.strftime("%A %d %B")


def _run(ctx: ToolContext, args: dict) -> ToolResult:
    order_id = norm_order_id(args.get("order_id", ""))
    if not order_id:
        return ToolResult("not_found", _NO_MATCH)
    row, err = verify(ctx, order_id, args.get("verification_code", ""))
    if err:
        return err
    with connect(ctx.db_path) as con:
        items = con.execute(
            "SELECT p.name, oi.qty FROM order_items oi JOIN products p ON p.id = oi.product_id "
            "WHERE oi.order_id = ?", (order_id,)).fetchall()
    status = row["status"].replace("_", " ")
    bits = [f"Order {order_id}: status {status}"]
    if row["carrier"]:
        bits.append(f"carrier {row['carrier']}, tracking reference {row['tracking_ref']}")
    if row["status"] == "delivered":
        bits.append(f"delivered {row['delivered_days_ago']} day(s) ago")
    elif row["status"] == "out_for_delivery":
        bits.append("arriving today")
    elif row["eta_days_from_now"] is not None:
        bits.append(f"estimated delivery {_fmt_day(ctx, row['eta_days_from_now'])}")
    bits.append("items: " + ", ".join(f"{i['name']} x{i['qty']}" for i in items))
    return ToolResult("ok", "; ".join(bits) + ". (Demo order data.)",
                      {"order_id": order_id, "status": row["status"], "carrier": row["carrier"],
                       "tracking_ref": row["tracking_ref"]},
                      sources=[{"doc_id": "orders/orders_045_confirmation.md", "title": "Order status (demo data)",
                                "category": "orders", "snippet": f"Order {order_id}: {status}."}])


async def lookup_order(args: dict, ctx: ToolContext) -> ToolResult:
    return await asyncio.to_thread(_run, ctx, args)
