"""
Document loading + chunking.

Strategy (structure-aware, sentence-bounded):
  1. Title = first '# ' heading; category = parent folder name.
  2. Drop boilerplate sections (e.g. '## Source') -- they are identical in every
     corpus file, so embedding them would only add noise.
  3. Pack whole sentences into chunks of at most `max_chars`, repeating the last
     `overlap` sentence(s) at the start of the next chunk so facts that straddle
     a boundary stay retrievable.
  4. Every chunk is prefixed with its document title ("contextual header") so the
     embedding knows WHICH product/policy the text is about.

With this corpus (~440 chars/doc) almost every doc becomes exactly one chunk; the
splitting path exists so the pipeline still works when documents are updated or
replaced with longer ones.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import BOILERPLATE_SECTIONS, CHUNK_MAX_CHARS, CHUNK_OVERLAP_SENTENCES

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_SKIP_FILES = {"CORPUS_README.md"}


@dataclass
class Chunk:
    chunk_id: str   # "<doc_id>#<n>"
    doc_id: str     # path relative to corpus dir, e.g. "returns/returns_033_eligibility.md"
    title: str
    category: str
    text: str       # title-prefixed text: this is what gets embedded AND shown to the LLM
    key: str        # sha256(text): lets the indexer reuse vectors for unchanged chunks


def parse_markdown(raw: str, fallback_title: str) -> tuple[str, str]:
    """Return (title, body) with boilerplate sections removed."""
    title, kept, skipping = fallback_title, [], False
    for line in raw.splitlines():
        if line.startswith("# ") and title == fallback_title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            skipping = line[3:].strip().lower() in BOILERPLATE_SECTIONS
            if skipping:
                continue
        if not skipping:
            kept.append(line)
    return title, "\n".join(kept).strip()


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            parts.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return parts


def chunk_body(sentences: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks, cur = [], []
    for sent in sentences:
        if cur and len(" ".join(cur + [sent])) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(doc_id: str, raw: str, max_chars: int = CHUNK_MAX_CHARS,
                   overlap: int = CHUNK_OVERLAP_SENTENCES) -> list[Chunk]:
    category = Path(doc_id).parent.name or "general"
    title, body = parse_markdown(raw, fallback_title=Path(doc_id).stem)
    pieces = chunk_body(split_sentences(body), max_chars, overlap)
    out = []
    for i, piece in enumerate(pieces):
        text = f"{title}\n{piece}"
        out.append(Chunk(f"{doc_id}#{i}", doc_id, title, category, text,
                         hashlib.sha256(text.encode()).hexdigest()))
    return out


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """doc_id (posix path relative to corpus_dir) -> raw markdown text."""
    return {
        p.relative_to(corpus_dir).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(corpus_dir.rglob("*.md"))
        if p.name not in _SKIP_FILES
    }
