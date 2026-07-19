from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def format_timestamp(value: float) -> str:
    total_ms = max(0, round(value * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt_text(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        match = _TIMESTAMP_RE.search(block)
        if not match:
            continue
        lines = block.splitlines()
        timing_line = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_line is None:
            continue
        cue_text = "\n".join(lines[timing_line + 1 :]).strip()
        cues.append(
            Cue(
                parse_timestamp(match.group("start")),
                parse_timestamp(match.group("end")),
                cue_text,
            )
        )
    return cues


def read_srt(path: Path) -> list[Cue]:
    return parse_srt_text(path.read_text(encoding="utf-8-sig"))


def render_srt(cues: Iterable[Cue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        if cue.end < cue.start:
            raise ValueError(f"cue {index} ends before it starts")
        blocks.append(
            f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(cues: Iterable[Cue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")
