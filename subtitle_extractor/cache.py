from __future__ import annotations

import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Callable

from .constants import PIPELINE_VERSION

_CACHE_PACKAGES = ("mlx-whisper", "faster-whisper", "whisperx", "torch", "transformers")


def runtime_signature() -> dict[str, str | None]:
    versions = {}
    for package in _CACHE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"pipeline": PIPELINE_VERSION, "packages": versions}


class ArtifactCache:
    def __init__(self, workdir: Path, benchmark_eligible: bool):
        self.workdir = workdir
        self.benchmark_eligible = benchmark_eligible
        self.stage_dir = workdir / "stages"
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}

    def run(self, name: str, outputs: list[Path], parameters: dict, action: Callable[[], None]) -> bool:
        meta_path = self.stage_dir / f"{name}.json"
        expected = {
            "runtime": runtime_signature(),
            "parameters": parameters,
            "outputs": [str(path) for path in outputs],
        }
        if meta_path.exists() and all(path.exists() for path in outputs):
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if previous == expected:
                self.records[name] = {"status": "cached", "seconds": 0.0 if self.benchmark_eligible else None}
                return True

        existing = [path for path in outputs if path.exists()]
        if existing:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = self.workdir / "obsolete" / f"{stamp}-{name}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for path in existing:
                shutil.move(str(path), backup_dir / path.name)

        started = monotonic()
        action()
        missing = [path for path in outputs if not path.exists()]
        if missing:
            raise RuntimeError(f"stage {name} did not create: {', '.join(map(str, missing))}")
        seconds = round(monotonic() - started, 3) if self.benchmark_eligible else None
        meta_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
        self.records[name] = {"status": "completed", "seconds": seconds}
        return False
