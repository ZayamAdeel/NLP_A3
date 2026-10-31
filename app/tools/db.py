"""
Mock BrightCart backend (SQLite) used by the tools.

Product ids, names and descriptions are read from the corpus documents so the
catalogue and the RAG knowledge base never disagree. Prices, stock and orders do
NOT exist in the corpus; they are demo data defined here (PKR). Order dates are
stored as "days ago" and resolved at query time, so the demo data never goes stale.
"""
import re
import sqlite3
from pathlib import Path

from app.config import CORPUS_DIR, DB_PATH

# product id -> (price PKR, stock quantity)
PRICING = {
    "BC-HEAD-001": (9500, 24), "BC-AUD-002": (6900, 12), "BC-CBL-003": (1200, 80),
    "BC-MOU-004": (2400, 35), "BC-KEY-005": (8900, 9), "BC-STA-006": (3800, 14),
    "BC-CAM-007": (7500, 0), "BC-PWR-008": (5600, 3), "BC-CASE-009": (1500, 40),
    "BC-SSD-010": (16500, 7), "BC-LMP-011": (4200, 18), "BC-WAT-012": (14900, 5),
    "BC-HST-013": (7200, 11), "BC-STD-014": (2600, 22), "BC-HUB-015": (6800, 0),
}
CATEGORY_BY_CODE = {
    "HEAD": "audio", "AUD": "audio", "HST": "audio", "CBL": "charging", "PWR": "charging",
    "MOU": "computer accessories", "KEY": "computer accessories", "STA": "computer accessories",
    "STD": "computer accessories", "HUB": "computer accessories", "CAM": "computer accessories",
    "SSD": "storage", "CASE": "phone accessories", "LMP": "home office", "WAT": "wearables",
}
CATEGORIES = sorted(set(CATEGORY_BY_CODE.values()) | {"gift cards"})

# order_id: (customer, phone_last4, status, carrier, tracking, placed_days_ago, eta_days_from_now,
#            delivered_days_ago, [(product_id, qty, final_sale)])
ORDERS = {
    "BC1024": ("Ayesha Khan", "4721", "shipped", "TCS", "TCS-884201", 3, 2, None,
               [("BC-HEAD-001", 1, 0), ("BC-CASE-009", 1, 0)]),
    "BC1025": ("Bilal Rauf", "3308", "delivered", "TCS", "TCS-771950", 14, None, 10,
               [("BC-HEAD-001", 1, 0), ("BC-CBL-003", 2, 0)]),
    "BC1026": ("Sana Malik", "9156", "delivered", "Leopards", "LEO-550317", 52, None, 45,
               [("BC-KEY-005", 1, 0)]),
    "BC1027": ("Hamza Tariq", "2087", "processing", None, None, 0, 4, None,
               [("BC-SSD-010", 1, 0)]),
    "BC1028": ("Zainab Farooq", "6642", "delivered", "Trax", "TRX-309988", 6, None, 3,
               [("BC-CASE-009", 1, 1), ("BC-GIFT-016", 1, 1), ("BC-LMP-011", 1, 0)]),
    "BC1029": ("Omar Siddiqui", "1190", "out_for_delivery", "TCS", "TCS-662044", 4, 0, None,
               [("BC-PWR-008", 1, 0)]),
    "BC1030": ("Maryam Ali", "5523", "delivered", "Leopards", "LEO-612809", 9, None, 5,
               [("BC-WAT-012", 1, 0)]),
}

SCHEMA = """
CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, price_pkr INTEGER,
  stock_qty INTEGER, description TEXT, doc_id TEXT, non_returnable INTEGER DEFAULT 0);
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer TEXT, phone_last4 TEXT, status TEXT,
  carrier TEXT, tracking_ref TEXT, placed_days_ago INTEGER, eta_days_from_now INTEGER,
  delivered_days_ago INTEGER);
CREATE TABLE order_items (order_id TEXT, product_id TEXT, qty INTEGER, final_sale INTEGER);
"""


def _corpus_products() -> list[tuple]:
    rows = []
    for f in sorted((CORPUS_DIR / "products").glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        name = raw.splitlines()[0].lstrip("# ").strip()
        pid = re.search(r"Product ID ([A-Z0-9-]+)", raw).group(1)
        body = " ".join(l for l in raw.split("## Source")[0].splitlines()[1:] if l.strip())
        rows.append((pid, name, body, f"products/{f.name}"))
    return rows


def ensure_db(path: Path = DB_PATH, force: bool = False) -> Path:
    """Create + seed the database if missing (or when force=True)."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for pid, name, body, doc_id in _corpus_products():
        price, qty = PRICING[pid]
        code = pid.split("-")[1]
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,0)",
                    (pid, name, CATEGORY_BY_CODE[code], price, qty, body, doc_id))
    con.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,1)",
                ("BC-GIFT-016", "BrightCart Gift Card", "gift cards", 5000, 999,
                 "Digital gift card redeemable at checkout. Gift cards cannot be returned.",
                 "payments/payments_042_giftcard.md"))
    for oid, (cust, last4, status, carrier, trk, placed, eta, deliv, items) in ORDERS.items():
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                    (oid, cust, last4, status, carrier, trk, placed, eta, deliv))
        con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", [(oid, *i) for i in items])
    con.commit()
    con.close()
    return path


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con
