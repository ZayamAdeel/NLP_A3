"""Shared types for tools (kept separate to avoid circular imports)."""
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class ToolContext:
    session: Any            # needs: verified_orders:set, failed_verifications:int
    db_path: Path
    today: date


@dataclass
class ToolResult:
    status: str             # ok | not_found | needs_info | needs_verification | locked | error
    summary: str            # compact, factual text the LLM turns into the spoken answer
    data: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # [{doc_id, title, category, snippet}]
