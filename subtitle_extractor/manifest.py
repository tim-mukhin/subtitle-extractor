from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .constants import CLASSLA_MODEL, FASTER_WHISPER_MODEL, MLX_LARGE_REPO, MLX_TURBO_REPO, PIPELINE_VERSION

_HEAVY_MARKERS = (
    "ComfyUI/main.py",
    "comfyui",
    "ollama runner",
    "stable-diffusion",
    "flux",
    "DaVinci Resolve.app/Contents/MacOS/Resolve",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def detect_heavy_work(include_resolve: bool = True) -> list[str]:
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    markers = _HEAVY_MARKERS if include_resolve else tuple(
        marker for marker in _HEAVY_MARKERS if "DaVinci Resolve" not in marker
    )
    warnings = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in markers):
            warnings.append(line.strip())
    return warnings


def package_versions() -> dict[str, str | None]:
    names = ["mlx-whisper", "faster-whisper", "whisperx", "torch", "numpy", "transformers"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def base_manifest(video: Path, language: str, intro_skip: float, heavy_work: list[str]) -> dict:
    return {
        "pipeline": PIPELINE_VERSION,
        "package_version": __version__,
        "created_at": utc_now(),
        "input": input_fingerprint(video),
        "language": language,
        "intro_skip": intro_skip,
        "benchmark_eligible": not heavy_work,
        "concurrent_heavy_work_warning": heavy_work,
        "models": {
            "mlx_large": MLX_LARGE_REPO,
            "mlx_turbo": MLX_TURBO_REPO,
            "faster_whisper": FASTER_WHISPER_MODEL,
            "forced_aligner": CLASSLA_MODEL,
        },
        "packages": package_versions(),
        "stages": {},
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
