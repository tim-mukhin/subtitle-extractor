from __future__ import annotations

import subprocess
from pathlib import Path

from .audio import require_executable
from .srt import Cue


def load_audio(path: Path, sample_rate: int = 16000):
    import numpy as np

    ffmpeg = require_executable("ffmpeg")
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def speech_regions(path: Path, threshold: float = 0.25) -> list[tuple[float, float]]:
    import torch

    torch.set_num_threads(4)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]
    sample_rate = 16000
    audio = torch.from_numpy(load_audio(path, sample_rate))
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=250,
        min_speech_duration_ms=80,
        speech_pad_ms=180,
    )
    return [(item["start"] / sample_rate, item["end"] / sample_rate) for item in timestamps]


def clamp_cues(cues: list[Cue], speech: list[tuple[float, float]], pad: float = 0.1) -> list[Cue]:
    output: list[Cue] = []
    for cue in cues:
        overlaps = [
            (max(cue.start, start), min(cue.end, end))
            for start, end in speech
            if min(cue.end, end) > max(cue.start, start)
        ]
        if not overlaps:
            output.append(cue)
            continue
        start = max(cue.start, overlaps[0][0] - pad)
        end = min(cue.end, overlaps[-1][1] + pad)
        end = max(end, start + 0.3)
        output.append(Cue(start, end, cue.text))
    for index in range(len(output) - 1):
        current, following = output[index], output[index + 1]
        if current.end > following.start - 0.02:
            output[index] = Cue(current.start, max(current.start + 0.3, following.start - 0.02), current.text)
    return output
