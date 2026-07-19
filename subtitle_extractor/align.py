from __future__ import annotations

from pathlib import Path

from .srt import Cue
from .text import tokens


def align_selected_windows(
    audio: Path,
    selected: list[dict],
    model_name: str,
    language: str,
    device: str = "cpu",
) -> tuple[list[dict], list[Cue], list[int]]:
    import whisperx

    active = [item for item in selected if item["text"].strip()]
    segments = [
        {"start": item["start"], "end": item["end"], "text": item["text"]}
        for item in active
    ]
    if not segments:
        return [], [], []

    model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=model_name,
    )
    decoded_audio = whisperx.load_audio(str(audio))
    result = whisperx.align(
        segments,
        model,
        metadata,
        decoded_audio,
        device,
        return_char_alignments=False,
        print_progress=True,
    )

    words: list[dict] = []
    fallback_cues: list[Cue] = []
    failed_windows: list[int] = []
    aligned_segments = result.get("segments", [])
    for index, item in enumerate(active):
        segment = aligned_segments[index] if index < len(aligned_segments) else {}
        segment_words = [
            word
            for word in segment.get("words", [])
            if word.get("start") is not None and word.get("end") is not None
        ]
        expected_count = len(tokens(item["text"]))
        aligned_count = sum(len(tokens(str(word.get("word", "")))) for word in segment_words)
        if expected_count == 0 or aligned_count < expected_count:
            failed_windows.append(item["window_id"])
            fallback_cues.append(Cue(item["start"], item["end"], item["text"]))
            continue
        for word in segment_words:
            words.append(
                {
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                    "word": str(word["word"]).strip(),
                    "prob": round(float(word.get("score", 0) or 0), 3),
                }
            )
    return words, fallback_cues, failed_windows
