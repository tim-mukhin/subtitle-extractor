from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .constants import DEFAULT_RESOLVE_APP, PIPELINE_VERSION

_RESOLVE_API = Path("/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting")
_RESOLVE_LIB = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so")


def _running_resolve_commands() -> list[str]:
    result = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if "DaVinci Resolve.app/Contents/MacOS/Resolve" in line]


def _connect_resolve():
    os.environ.setdefault("RESOLVE_SCRIPT_API", str(_RESOLVE_API))
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", str(_RESOLVE_LIB))
    module_path = str(_RESOLVE_API / "Modules")
    if module_path not in sys.path:
        sys.path.append(module_path)
    dvr = importlib.import_module("DaVinciResolveScript")
    for _ in range(30):
        resolve = dvr.scriptapp("Resolve")
        if resolve is not None:
            return resolve
        time.sleep(2)
    raise RuntimeError("could not connect to DaVinci Resolve scripting API")


def render_voice_isolation(
    source_wav: Path,
    output_wav: Path,
    amount: int = 50,
    app_path: Path = DEFAULT_RESOLVE_APP,
) -> str:
    if platform.system() != "Darwin":
        raise RuntimeError("DaVinci Resolve branch is currently supported only on macOS")
    if not 0 <= amount <= 100:
        raise ValueError("Resolve Voice Isolation amount must be 0..100")
    if not app_path.exists():
        raise RuntimeError(f"DaVinci Resolve not found: {app_path}")

    commands = _running_resolve_commands()
    if commands and any("-nogui" not in command for command in commands):
        raise RuntimeError("Resolve is already running without -nogui; close it before the optional headless branch")
    if not commands:
        log_path = output_wav.parent / "resolve-headless.log"
        log = log_path.open("ab")
        subprocess.Popen([str(app_path), "-nogui"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    resolve = _connect_resolve()
    manager = resolve.GetProjectManager()
    project_name = f"SubtitleEnsemble-{PIPELINE_VERSION}"
    project = manager.LoadProject(project_name) or manager.CreateProject(project_name)
    if project is None:
        raise RuntimeError(f"could not load or create Resolve project {project_name}")

    media_pool = project.GetMediaPool()
    items = media_pool.ImportMedia([str(source_wav.resolve())])
    if not items:
        raise RuntimeError(f"Resolve could not import {source_wav}")
    timeline_name = f"{source_wav.stem}-vi{amount}-{PIPELINE_VERSION}-{int(time.time())}"
    timeline = media_pool.CreateTimelineFromClips(timeline_name, [items[0]])
    if timeline is None:
        raise RuntimeError("Resolve could not create a timeline")
    project.SetCurrentTimeline(timeline)

    timeline_items = timeline.GetItemListInTrack("audio", 1)
    state = {"isEnabled": True, "amount": amount}
    enabled = False
    if timeline_items and hasattr(timeline_items[0], "SetVoiceIsolationState"):
        enabled = bool(timeline_items[0].SetVoiceIsolationState(state))
    elif hasattr(timeline, "SetVoiceIsolationState"):
        enabled = bool(timeline.SetVoiceIsolationState(1, state))
    if not enabled:
        raise RuntimeError("Resolve Voice Isolation could not be enabled")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    render_name = output_wav.stem
    if not project.LoadRenderPreset("Audio Only"):
        raise RuntimeError("Resolve render preset 'Audio Only' is unavailable")
    project.SetRenderSettings(
        {
            "TargetDir": str(output_wav.parent.resolve()),
            "CustomName": render_name,
            "ExportVideo": False,
            "ExportAudio": True,
        }
    )
    job_id = project.AddRenderJob()
    if not job_id:
        raise RuntimeError("Resolve AddRenderJob failed")
    project.StartRendering([job_id], False)
    while project.IsRenderingInProgress():
        time.sleep(1)
    status = project.GetRenderJobStatus(job_id)
    if status.get("JobStatus") != "Complete":
        raise RuntimeError(f"Resolve render failed: {status}")

    candidates = sorted(output_wav.parent.glob(f"{render_name}*.wav"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"Resolve completed but no WAV matched {render_name}*.wav")
    rendered = candidates[-1]
    if rendered != output_wav:
        rendered.rename(output_wav)
    return str(resolve.GetVersionString())
