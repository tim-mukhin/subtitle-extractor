from __future__ import annotations

import collections
import re
from collections.abc import Iterable

_SERBIAN_CYRILLIC = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj",
    "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S",
    "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č",
    "Џ": "Dž", "Ш": "Š", "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "ђ": "đ", "е": "e", "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k",
    "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h",
    "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
}
_TOKEN_RE = re.compile(r"[a-zčćžšđ0-9]+", re.IGNORECASE)


def serbian_cyrillic_to_latin(text: str) -> str:
    return "".join(_SERBIAN_CYRILLIC.get(char, char) for char in text)


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(serbian_cyrillic_to_latin(text).lower())


def ngrams(text: str, size: int = 3) -> set[str]:
    values = tokens(text)
    return {" ".join(values[i : i + size]) for i in range(len(values) - size + 1)}


def frequent_ngrams(texts: Iterable[str], size: int = 3, minimum_windows: int = 4) -> set[str]:
    counts: collections.Counter[str] = collections.Counter()
    for text in texts:
        counts.update(ngrams(text, size))
    return {gram for gram, count in counts.items() if count >= minimum_windows}


def repetition_flags(text: str, frequent: set[str], size: int = 3) -> list[str]:
    return sorted(ngrams(text, size) & frequent)


def suspicious_repetition(text: str, flags: Iterable[str] = ()) -> bool:
    values = tokens(text)
    if not values:
        return False
    max_run = run = 1
    for index in range(1, len(values)):
        run = run + 1 if values[index] == values[index - 1] else 1
        max_run = max(max_run, run)
    dominance = max(values.count(value) for value in set(values)) / len(values)
    return max_run >= 4 or (len(values) >= 8 and dominance >= 0.45) or (
        len(values) >= 8 and bool(tuple(flags))
    )
