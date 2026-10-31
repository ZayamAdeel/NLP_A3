"""
Text <-> speech normalisation.

  words_to_digits   ASR output ("ten thousand rupees", "one zero two four") -> digits
  canonical_order_ids   "B C 1024" / "bc-1024" -> "BC1024"
  find_code         pull a spoken 4-digit verification code out of a message
  speakable         reply text -> text a TTS engine reads naturally
"""
import re

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000}
_MULT = {"double": 2, "triple": 3}
_NUMWORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_SCALES) | set(_MULT)
_TOKEN = re.compile(r"[A-Za-z']+|\d[\d,]*|\s+|[^\sA-Za-z\d]")


def _cardinal(words: list[str]) -> int:
    total = current = 0
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TEENS:
            current += _TEENS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in ("thousand", "lakh"):
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def _digit_string(words: list[str]) -> str:
    """No scale words: read as a digit sequence ('ten twenty four' -> '1024')."""
    out, i = "", 0
    while i < len(words):
        w = words[i]
        if w in _MULT and i + 1 < len(words) and words[i + 1] in _UNITS:
            out += str(_UNITS[words[i + 1]]) * _MULT[w]
            i += 2
        elif w in _TENS and i + 1 < len(words) and words[i + 1] in _UNITS and 0 < _UNITS[words[i + 1]]:
            out += str(_TENS[w] + _UNITS[words[i + 1]])
            i += 2
        else:
            out += str(_UNITS.get(w) if w in _UNITS else _TEENS.get(w) or _TENS.get(w))
            i += 1
    return out


def words_to_digits(text: str) -> str:
    """Convert spoken numbers to digits. Lone words below ten are left alone ('one of')."""
    toks = _TOKEN.findall(text)
    n = len(toks)

    def skip_ws(k: int) -> int:
        while k < n and toks[k].isspace():
            k += 1
        return k

    out, i = [], 0
    while i < n:
        if toks[i].lower() not in _NUMWORDS:
            out.append(toks[i])
            i += 1
            continue
        run, j = [], i
        while True:                                   # consume number words, bridging spaces
            run.append(toks[j].lower())
            end = j + 1
            k = skip_ws(end)
            if k < n and toks[k].lower() == "and" and any(w in _SCALES for w in run):
                k2 = skip_ws(k + 1)                   # "one hundred AND five"
                if k2 < n and toks[k2].lower() in _NUMWORDS:
                    k = k2
            if k < n and toks[k].lower() in _NUMWORDS:
                j = k
                continue
            break
        if len(run) == 1 and (run[0] in _UNITS or run[0] in _MULT):
            out.append(toks[i])                       # "one", "double" on their own stay words
        else:
            has_scale = any(w in _SCALES for w in run)
            out.append(str(_cardinal(run)) if has_scale else _digit_string(run))
        i = end
    return "".join(out)


_ORDER_RE = re.compile(r"\bb[\s.\-]?c[\s.\-]?(\d(?:[\s\-]?\d){2,5})(?!\d)", re.IGNORECASE)


def canonical_order_ids(text: str) -> str:
    return _ORDER_RE.sub(lambda m: "BC" + re.sub(r"\D", "", m.group(1)), text)


def find_order_id(text: str) -> str | None:
    m = re.search(r"\bBC\d{3,6}\b", canonical_order_ids(text))
    return m.group(0) if m else None


def find_code(text: str) -> str | None:
    """A standalone 4-digit number that is not part of an order id."""
    t = re.sub(r"BC\d{3,6}", " ", canonical_order_ids(text))
    t = re.sub(r"\b\d(?:[\s\-]\d){3}\b", lambda m: re.sub(r"\D", "", m.group(0)), t)  # "4 7 2 1"
    m = re.search(r"(?<!\d)\d{4}(?!\d)", t)
    return m.group(0) if m else None


def normalize_transcript(text: str) -> str:
    """Full clean-up applied to ASR output before it reaches the conversation manager."""
    return canonical_order_ids(words_to_digits(text)).strip()


_SPELL = lambda s: " ".join(s)  # noqa: E731


def speakable(text: str) -> str:
    t = re.sub(r"\(?\s*see\s*\[\d+(?:\s*,\s*\d+)*\]\s*\)?", "", text, flags=re.I)
    t = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", t)
    t = re.sub(r"[*_`#>~|]", "", t)
    t = re.sub(r"\bBC-([A-Z]+)-(\d+)\b", lambda m: f"B C {m.group(1).lower()} {_SPELL(m.group(2))}", t)
    t = re.sub(r"\bBC(\d{3,6})\b", lambda m: "B C " + _SPELL(m.group(1)), t)
    t = re.sub(r"\b(?:Rs\.?|PKR)\s?([\d,]+)", r"\1 rupees", t, flags=re.I)
    t = re.sub(r"[^\x20-\x7E\n]", "", t)           # emoji / exotic symbols
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\s+([.,!?])", r"\1", t)
    t = re.sub(r"([.!?])(\s*\.)+", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()
