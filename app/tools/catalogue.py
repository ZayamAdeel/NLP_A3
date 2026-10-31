"""Tool 1 - search_products: query the local SQLite catalogue."""
import asyncio
import re

from app.tools.db import connect
from app.tools.types import ToolContext, ToolResult

_STOP = {"a", "an", "the", "for", "with", "and", "or", "me", "some", "any", "find", "show", "looking",
         "under", "below", "than", "less", "please", "i", "want", "need", "to", "buy", "get", "product",
         "products", "item", "items", "rupees", "rs", "pkr", "of", "in", "stock", "available", "cheap",
         "cheapest", "best", "good", "my", "your", "do", "you", "have", "sell", "is", "are", "there"}
LIMIT = 3


def _tokens(query: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t not in _STOP]
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks]   # crude plural strip


def _run(db_path, args: dict) -> ToolResult:
    toks = _tokens(args.get("query", ""))
    category, max_price = args.get("category") or None, args.get("max_price")
    if not (toks or category or max_price):
        return ToolResult("needs_info", "No search terms given.", missing=["query"])

    def fetch(use_price: bool, use_stock: bool):
        sql, params = ["SELECT * FROM products WHERE 1=1"], []
        for t in toks:
            sql.append("AND lower(name || ' ' || description || ' ' || category) LIKE ?")
            params.append(f"%{t}%")
        if category:
            sql.append("AND category = ?"); params.append(category)
        if use_price and max_price:
            sql.append("AND price_pkr <= ?"); params.append(float(max_price))
        if use_stock:
            sql.append("AND stock_qty > 0")
        sql.append("ORDER BY price_pkr ASC")
        with connect(db_path) as con:
            return con.execute(" ".join(sql), params).fetchall()

    rows = fetch(True, bool(args.get("in_stock_only")))
    if not rows:
        loose = fetch(False, False)   # explain what the filters excluded instead of a bare "none"
        if loose:
            p = loose[0]
            hint = (f"No products matched all your filters. The closest match ignoring filters is "
                    f"{p['name']} at {p['price_pkr']:,} rupees ({'in stock' if p['stock_qty'] else 'out of stock'}).")
            return ToolResult("ok", hint, {"count": 0, "closest": p["id"]},
                              sources=[_src(p)])
        return ToolResult("ok", "No products in the catalogue matched that search.", {"count": 0})

    shown = rows[:LIMIT]
    parts = [f"{i}) {r['name']} ({r['id']}): {r['price_pkr']:,} rupees, "
             f"{'in stock' if r['stock_qty'] > 0 else 'out of stock'}" for i, r in enumerate(shown, 1)]
    more = f" ({len(rows) - LIMIT} more not listed)" if len(rows) > LIMIT else ""
    return ToolResult("ok", f"Found {len(rows)} matching product(s){more}: " + "; ".join(parts) + ".",
                      {"count": len(rows), "products": [dict(r) for r in shown]},
                      sources=[_src(r) for r in shown])


def _src(r) -> dict:
    return {"doc_id": r["doc_id"], "title": r["name"], "category": r["category"],
            "snippet": f"{r['name']} - {r['price_pkr']:,} rupees, "
                       f"{'in stock' if r['stock_qty'] > 0 else 'out of stock'}."}


async def search_products(args: dict, ctx: ToolContext) -> ToolResult:
    return await asyncio.to_thread(_run, ctx.db_path, args)
