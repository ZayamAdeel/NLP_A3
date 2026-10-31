"""Re-create the mock BrightCart database:  python scripts/seed_db.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DB_PATH
from app.tools.db import ensure_db

if __name__ == "__main__":
    print(f"Seeded {ensure_db(DB_PATH, force=True)}")
