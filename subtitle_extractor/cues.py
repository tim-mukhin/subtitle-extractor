from __future__ import annotations

import re
from collections import Counter

from .srt import Cue
from .text import suspicious_repetition

_PUNCTUATION = (",", ".", "!", "?", ":", ";", "%")


def join_words(words: list[dict]) -> str:
    return "".join(
        ("" if str(word["word"]).startswith(_PUNCTUATION) else " ") + str(word["word"])
        for word in words
    ).strip()


def wrap_two_lines(text: str, max_chars: int = 42) -> str:
    if len(text) <= max_chars:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    candidates = []
    for split in range(1, len(words)):
        left, right = " ".join(words[:split]), " ".join(words[split:])
        overflow = max(0, len(left) - max_chars) + max(0, len(right) - max_chars)
        candidates.append((overflow, abs(len(left) - len(right)), split, left, right))
    _, _, _, left, right = min(candidates)
    return f"{left}\n{right}"


def build_cues(
    words: list[dict],
    max_gap: float = 0.6,
    max_duration: float = 6.0,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_flush_duration: float = 1.2,
    start_pad: float = 0.03,
    end_pad: float = 0.12,
) -> list[Cue]:
    valid = [word for word in words if word.get("start") is not None and word.get("end") is not None]
    grouped: list[list[dict]] = []
    current: list[dict] = []
    max_chars = max_chars_per_line * max_lines
    for word in valid:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            duration = float(current[-1]["end"]) - float(current[0]["start"])
            text = join_words(current)
            sentence_end = bool(re.search(r"[.!?]$", str(current[-1]["word"]).strip()))
            if (
                gap > max_gap
                or duration >= max_duration
                or (len(text) >= max_chars and gap > 0.12)
                or (sentence_end and duration >= min_flush_duration)
            ):
                grouped.append(current)
                current = []
        current.append(word)
    if current:
        grouped.append(current)

    cues = [
        Cue(
            max(0.0, float(group[0]["start"]) - start_pad),
            float(group[-1]["end"]) + end_pad,
            wrap_two_lines(join_words(group), max_chars_per_line),
        )
        for group in grouped
    ]
    return remove_overlaps(cues)


def remove_overlaps(cues: list[Cue]) -> list[Cue]:
    output = sorted(cues, key=lambda cue: (cue.start, cue.end))
    for index in range(len(output) - 1):
        current, following = output[index], output[index + 1]
        if current.end > following.start - 0.02:
            output[index] = Cue(
                current.start,
                max(current.start + 0.3, following.start - 0.02),
                current.text,
            )
    return output


def clean_cues(cues: list[Cue], intro_skip: float = 0.0) -> list[Cue]:
    output: list[Cue] = []
    previous = None
    for cue in cues:
        text = cue.text.strip()
        if cue.start < intro_skip or not re.search(r"\w", text):
            continue
        normalized = text.lower().strip(" .,!?…")
        if normalized == previous:
            continue
        previous = normalized
        values = re.findall(r"\w+", text.lower())
        if values:
            _, count = Counter(values).most_common(1)[0]
            if count >= 4 and len(values) >= 5:
                continue
        if suspicious_repetition(text):
            continue
        output.append(Cue(cue.start, cue.end, text))
    return remove_overlaps(output)


def cue_stats(cues: list[Cue], max_cps: float = 17.0, max_cpl: int = 42) -> dict:
    cps_violations = 0
    cpl_violations = 0
    for cue in cues:
        duration = max(0.001, cue.end - cue.start)
        if len(cue.text.replace("\n", "")) / duration > max_cps:
            cps_violations += 1
        if any(len(line) > max_cpl for line in cue.text.splitlines()):
            cpl_violations += 1
    return {"cues": len(cues), "cps_violations": cps_violations, "cpl_violations": cpl_violations}
