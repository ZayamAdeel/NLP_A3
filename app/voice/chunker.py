"""
Turns the LLM's token stream into speakable chunks so TTS can start before the answer ends.

Rules (in priority order):
  1. a sentence end ('.', '!', '?' followed by whitespace) ends a chunk;
  2. for the FIRST chunk only, a clause boundary (',', ';', ':') also ends it once at least
     `first_min` characters are buffered: this is what pulls first-audio latency in;
  3. a chunk never grows past `max_chars` (cut at the last space).
Abbreviations such as "Rs." and decimals such as "3.5" are not sentence ends.
"""
import re

from app.config import TTS_FIRST_CHUNK_MIN_CHARS, TTS_MAX_CHUNK_CHARS

_SENT_END = re.compile(r"[.!?][\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:]\s")
_ABBREV = {"rs", "mr", "mrs", "ms", "dr", "vs", "no", "st", "e.g", "i.e", "approx", "etc"}


class SpeechChunker:
    def __init__(self, first_min: int = TTS_FIRST_CHUNK_MIN_CHARS, max_chars: int = TTS_MAX_CHUNK_CHARS):
        self.first_min, self.max_chars = first_min, max_chars
        self.buf, self.emitted = "", 0

    def feed(self, token: str) -> list[str]:
        self.buf += token
        out = []
        while (cut := self._cut()) is not None:
            piece, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if piece:
                out.append(piece)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        if rest:
            self.emitted += 1
            return [rest]
        return []

    def _cut(self) -> int | None:
        for m in _SENT_END.finditer(self.buf):
            word = re.findall(r"[A-Za-z.]+", self.buf[: m.start() + 1])
            if word and word[-1].rstrip(".").lower() in _ABBREV:
                continue                                   # "Rs. 9,500" is not a sentence end
            return m.end()
        if self.emitted == 0:
            for m in _CLAUSE_END.finditer(self.buf):
                if m.end() >= self.first_min:
                    return m.end()
        if len(self.buf) >= self.max_chars:
            return self.buf.rfind(" ", 0, self.max_chars) + 1 or self.max_chars
        return None
