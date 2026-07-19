from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"required executable not found on PATH: {name}")
    return executable


def extract_audio(video: Path, output: Path) -> None:
    ffmpeg = require_executable("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v", "error", "-y", "-i", str(video), "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(output),
        ],
        check=True,
    )


def slow_audio(source: Path, output: Path, speed: float) -> None:
    if not 0.5 <= speed <= 1.0:
        raise ValueError("ffmpeg atempo speed must be between 0.5 and 1.0")
    ffmpeg = require_executable("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v", "error", "-y", "-i", str(source),
            "-filter:a", f"atempo={speed}", "-ar", "16000", "-ac", "1", str(output),
        ],
        check=True,
    )
