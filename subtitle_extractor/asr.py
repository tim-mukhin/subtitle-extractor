from __future__ import annotations

import json
from pathlib import Path

from .srt import Cue, write_srt


def write_words(words: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")


def read_words(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def transcribe_mlx(audio: Path, repo: str, language: str) -> tuple[list[Cue], list[dict]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    cues: list[Cue] = []
    words: list[dict] = []
    for segment in result["segments"]:
        cues.append(Cue(float(segment["start"]), float(segment["end"]), segment["text"].strip()))
        for word in segment.get("words", []):
            if word.get("start") is None or word.get("end") is None:
                continue
            words.append(
                {
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                    "word": word["word"].strip(),
                    "prob": round(float(word.get("probability", 0) or 0), 3),
                }
            )
    return cues, words


def transcribe_faster_whisper(
    audio: Path,
    model_name: str,
    language: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> tuple[list[Cue], list[dict]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        str(audio),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 100},
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    cues: list[Cue] = []
    words: list[dict] = []
    for segment in segments:
        cues.append(Cue(float(segment.start), float(segment.end), segment.text.strip()))
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            words.append(
                {
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "word": word.word.strip(),
                    "prob": round(float(word.probability or 0), 3),
                }
            )
    return cues, words


def save_transcript(cues: list[Cue], words: list[dict], srt_path: Path, words_path: Path) -> None:
    write_srt(cues, srt_path)
    write_words(words, words_path)
