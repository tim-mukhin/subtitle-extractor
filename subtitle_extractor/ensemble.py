from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

from .constants import BRANCH_FAMILIES
from .srt import Cue, read_srt
from .text import frequent_ngrams, ngrams, repetition_flags, serbian_cyrillic_to_latin, suspicious_repetition, tokens


def rescale_words(words: Iterable[dict], factor: float) -> list[dict]:
    if factor <= 0:
        raise ValueError("timestamp scale factor must be positive")
    output = []
    for word in words:
        scaled = dict(word)
        scaled["start"] = round(float(word["start"]) * factor, 3)
        scaled["end"] = round(float(word["end"]) * factor, 3)
        output.append(scaled)
    return output


def assign_words_to_windows(
    words: Iterable[dict],
    regions: list[tuple[float, float]],
    max_distance: float = 1.5,
) -> list[str]:
    assigned: list[list[str]] = [[] for _ in regions]
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2
        distances = [
            0.0 if start <= midpoint <= end else min(abs(midpoint - start), abs(midpoint - end))
            for start, end in regions
        ]
        if distances and min(distances) <= max_distance:
            assigned[distances.index(min(distances))].append(
                serbian_cyrillic_to_latin(str(word["word"]))
            )
    return [" ".join(parts).strip() for parts in assigned]


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalized_distance(left: str, right: str) -> float:
    left_tokens, right_tokens = tokens(left), tokens(right)
    return levenshtein(left_tokens, right_tokens) / max(1, len(left_tokens), len(right_tokens))


def available_families(sources: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    families = {
        family: [(key, sources[key]) for key in keys if sources.get(key, "").strip()]
        for family, keys in BRANCH_FAMILIES.items()
    }
    dynamic_resolve = [
        (key, text)
        for key, text in sources.items()
        if key.startswith("resolve21_vi") and key.endswith("_mlx") and text.strip()
    ]
    if dynamic_resolve:
        families["resolve"] = dynamic_resolve
    return families


def _source_has_repetition_flag(text: str, flags: Iterable[str]) -> bool:
    return bool(ngrams(text) & set(flags))


def branch_medoid(sources: dict[str, str], flag_map: dict[str, list[str]] | None = None) -> tuple[str, str]:
    flags = flag_map or {}
    candidates = [
        (key, text)
        for key, text in sources.items()
        if text.strip()
        and not suspicious_repetition(text)
        and not _source_has_repetition_flag(text, flags.get(key, ()))
    ]
    if not candidates:
        return "", ""

    families = available_families(sources)
    scored = []
    for key, text in candidates:
        score = 0.0
        participating = 0
        for members in families.values():
            if not members:
                continue
            participating += 1
            score += min(normalized_distance(text, member_text) for _, member_text in members)
        score /= max(1, participating)
        scored.append((score, -len(tokens(text)), key, text))
    _, _, key, text = min(scored)
    return key, text


def supporting_families(text: str, sources: dict[str, str], threshold: float = 0.55) -> set[str]:
    if not tokens(text):
        return set()
    return {
        family
        for family, members in available_families(sources).items()
        if any(1 - normalized_distance(text, candidate) >= threshold for _, candidate in members)
    }


def cross_branch_agreement(sources: dict[str, str], threshold: float = 0.55) -> bool:
    families = available_families(sources)
    populated = [(name, values) for name, values in families.items() if values]
    for index, (_, left_values) in enumerate(populated):
        for _, right_values in populated[index + 1 :]:
            if any(
                1 - normalized_distance(left, right) >= threshold
                for _, left in left_values
                for _, right in right_values
            ):
                return True
    return False


def exact_curated_override(window_id: int, corrections: Iterable[dict]) -> str | None:
    for correction in corrections:
        if correction.get("window_id") == window_id:
            value = correction.get("text", correction.get("suggested"))
            if value is None:
                raise ValueError(f"curated correction for window {window_id} has no text")
            return serbian_cyrillic_to_latin(str(value)).strip()
    return None


def _overlapping_cues(cues: list[Cue], start: float, end: float) -> list[dict]:
    return [
        {"start": cue.start, "end": cue.end, "text": cue.text}
        for cue in cues
        if cue.start < end and cue.end > start
    ]


def build_ensemble_windows(
    regions: list[tuple[float, float]],
    sources: dict[str, list[dict]],
    english_srt: Path | None = None,
    corrections_path: Path | None = None,
    max_assign_distance: float = 1.5,
) -> list[dict]:
    assigned = {
        key: assign_words_to_windows(words, regions, max_assign_distance)
        for key, words in sources.items()
    }
    frequent = {
        key: frequent_ngrams(texts, size=3, minimum_windows=4)
        for key, texts in assigned.items()
    }
    english = read_srt(english_srt) if english_srt and english_srt.exists() else []
    corrections = (
        json.loads(corrections_path.read_text(encoding="utf-8"))
        if corrections_path and corrections_path.exists()
        else []
    )
    if not isinstance(corrections, list):
        raise ValueError("corrections sidecar must contain a JSON array")

    valid_ids = set(range(1, len(regions) + 1))
    unknown = [
        correction.get("window_id")
        for correction in corrections
        if correction.get("window_id") not in valid_ids
    ]
    if unknown:
        print(
            f"warning: ignoring {len(unknown)} correction(s) with unknown/mistyped "
            f"window_id (valid range 1..{len(regions)}): {unknown}",
            file=sys.stderr,
        )

    windows = []
    for index, (start, end) in enumerate(regions, 1):
        source_texts = {key: texts[index - 1] for key, texts in assigned.items()}
        flags = {
            key: repetition_flags(source_texts[key], frequent[key])
            for key in source_texts
        }
        curated = exact_curated_override(index, corrections)
        windows.append(
            {
                "window_id": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "sources": source_texts,
                "repeated_ngram_flags": flags,
                "english": _overlapping_cues(english, start, end),
                "curated_override": curated,
            }
        )
    return windows


def write_agent_inputs(windows: list[dict], workdir: Path, agents: int = 4) -> list[dict]:
    if agents != 4:
        raise ValueError("the winning workflow uses exactly four fresh agents")
    input_dir = workdir / "resolver-input"
    result_dir = workdir / "resolver-results"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, math.ceil(len(windows) / agents))
    plan = []
    for part in range(1, agents + 1):
        subset = windows[(part - 1) * chunk_size : part * chunk_size]
        input_path = input_dir / f"part-{part}.json"
        output_path = result_dir / f"part-{part}.json"
        input_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        plan.append(
            {
                "part": part,
                "first_window": subset[0]["window_id"] if subset else None,
                "last_window": subset[-1]["window_id"] if subset else None,
                "count": len(subset),
                "input": str(input_path),
                "output": str(output_path),
            }
        )
    (workdir / "agent-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def merge_resolver_results(workdir: Path, windows: list[dict]) -> list[dict]:
    plan_path = workdir / "agent-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, list) or [entry.get("part") for entry in plan] != [1, 2, 3, 4]:
        raise ValueError("agent-plan.json must contain parts 1..4 in order")

    result_dir = (workdir / "resolver-results").resolve()
    expected_paths = {result_dir / f"part-{part}.json" for part in range(1, 5)}
    actual_paths = {path.resolve() for path in result_dir.glob("part-*.json")}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"resolver file mismatch: missing={missing}, extra={extra}")

    allowed_statuses = {"consensus", "curated", "unresolved", "no_speech"}
    allowed_families = {"original", "slowed", "resolve"}
    required_fields = {
        "window_id", "start", "end", "selected_text", "status", "confidence",
        "branch_support", "alternatives", "reason",
    }
    window_by_id = {window["window_id"]: window for window in windows}
    merged: list[dict] = []
    for entry in plan:
        part = entry["part"]
        input_path = Path(entry["input"])
        output_path = Path(entry["output"]).resolve()
        expected_output = result_dir / f"part-{part}.json"
        if output_path != expected_output:
            raise ValueError(f"agent plan output escapes its expected part: {output_path}")
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        expected_part_ids = [item["window_id"] for item in input_payload]
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"resolver result must be a JSON array: {output_path}")
        if [item.get("window_id") for item in payload if isinstance(item, dict)] != expected_part_ids:
            raise ValueError(f"resolver part {part} IDs/order do not match its input")

        for item in payload:
            if not isinstance(item, dict) or set(item) != required_fields:
                raise ValueError(f"resolver part {part} has an invalid object schema")
            window_id = item["window_id"]
            expected_window = window_by_id.get(window_id)
            if expected_window is None:
                raise ValueError(f"unknown resolver window_id: {window_id}")
            if item["start"] != expected_window["start"] or item["end"] != expected_window["end"]:
                raise ValueError(f"resolver changed boundaries for window {window_id}")
            if item["status"] not in allowed_statuses:
                raise ValueError(f"invalid resolver status for window {window_id}: {item['status']}")
            if not isinstance(item["selected_text"], str):
                raise ValueError(f"selected_text must be a string for window {window_id}")
            confidence = item["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"confidence must be 0..1 for window {window_id}")
            support = item["branch_support"]
            if not isinstance(support, list) or any(value not in allowed_families for value in support):
                raise ValueError(f"invalid branch_support for window {window_id}")
            alternatives = item["alternatives"]
            if not isinstance(alternatives, list) or any(not isinstance(value, str) for value in alternatives):
                raise ValueError(f"invalid alternatives for window {window_id}")
            if not isinstance(item["reason"], str):
                raise ValueError(f"reason must be a string for window {window_id}")
        merged.extend(payload)

    expected_ids = [window["window_id"] for window in windows]
    if [item["window_id"] for item in merged] != expected_ids:
        raise ValueError("resolver results do not cover all windows in order")
    return merged


def conservative_finalize(windows: list[dict], results: list[dict]) -> tuple[list[dict], list[dict]]:
    result_by_id = {item["window_id"]: item for item in results}
    selected: list[dict] = []
    unresolved: list[dict] = []
    for window in windows:
        result = result_by_id[window["window_id"]]
        sources = window["sources"]
        flags = window.get("repeated_ngram_flags", {})
        text = serbian_cyrillic_to_latin(result.get("selected_text", "")).strip()
        status = result["status"]
        source = "resolver"

        curated = window.get("curated_override")
        source_lengths = sorted(len(tokens(value)) for value in sources.values() if value.strip())
        median_length = source_lengths[len(source_lengths) // 2] if source_lengths else 0
        too_short = status == "consensus" and median_length >= 3 and len(tokens(text)) < 0.7 * median_length
        flagged_ngrams = {gram for source_flags in flags.values() for gram in source_flags}
        selected_has_flagged_repetition = bool(ngrams(text) & flagged_ngrams)
        unsupported_consensus = status == "consensus" and len(supporting_families(text, sources)) < 2
        false_curated = status == "curated" and curated is None

        if curated is not None:
            text, status, source = curated, "curated", "curated_override"
        elif (
            status == "unresolved"
            or false_curated
            or unsupported_consensus
            or suspicious_repetition(text)
            or selected_has_flagged_repetition
            or too_short
        ):
            source, text = branch_medoid(sources, flags)
            status = f"medoid_{status}"
        elif status == "no_speech" and cross_branch_agreement(sources):
            source, text = branch_medoid(sources, flags)
            status = "medoid_recovered_speech"
        elif status == "no_speech":
            text = ""

        record = {
            "window_id": window["window_id"],
            "start": window["start"],
            "end": window["end"],
            "text": text.strip(),
            "mode": status,
            "source": source,
            "original_status": result["status"],
            "confidence": result.get("confidence"),
            "reason": result.get("reason", ""),
        }
        selected.append(record)
        if result["status"] == "unresolved" or status.startswith("medoid_"):
            unresolved.append({**record, "alternatives": result.get("alternatives", []), "sources": sources})
    return selected, unresolved
