from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .align import align_selected_windows
from .asr import read_words, save_transcript, transcribe_faster_whisper, transcribe_mlx, write_words
from .audio import extract_audio, require_executable, slow_audio
from .cache import ArtifactCache
from .constants import (
    CLASSLA_MODEL,
    DEFAULT_RESOLVE_APP,
    FASTER_WHISPER_MODEL,
    MLX_LARGE_REPO,
    MLX_TURBO_REPO,
    PIPELINE_VERSION,
    SUPPORTED_LANGUAGES,
    VIDEO_EXTENSIONS,
)
from .cues import build_cues, clean_cues, cue_stats, remove_overlaps
from .ensemble import (
    build_ensemble_windows,
    conservative_finalize,
    merge_resolver_results,
    rescale_words,
    tokens,
    write_agent_inputs,
)
from .manifest import base_manifest, detect_heavy_work, input_fingerprint, load_json, utc_now, write_json
from .resolve import render_voice_isolation
from .srt import Cue, write_srt
from .vad import clamp_cues, speech_regions

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunPaths:
    workdir: Path
    audio: Path
    asr: Path
    windows: Path
    selected: Path
    manifest: Path
    config: Path

    @classmethod
    def from_workdir(cls, workdir: Path) -> "RunPaths":
        return cls(
            workdir=workdir,
            audio=workdir / "audio",
            asr=workdir / "asr",
            windows=workdir / "ensemble-windows.json",
            selected=workdir / "ensemble-selected.json",
            manifest=workdir / "manifest.json",
            config=workdir / "run.json",
        )


def default_workdir(video: Path) -> Path:
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", video.stem).strip("-") or "video"
    import hashlib

    path_id = hashlib.sha256(str(video.resolve()).encode()).hexdigest()[:10]
    return ROOT / ".subtitle-ensemble" / f"{safe_stem}-{path_id}"


def default_output(video: Path) -> Path:
    return ROOT / "output" / f"{video.stem}.ensemble.srt"


def sidecars_for(video: Path) -> tuple[Path | None, Path | None]:
    english = video.with_suffix(".en.srt")
    corrections = video.with_suffix(".corrections.json")
    return (english if english.exists() else None, corrections if corrections.exists() else None)


def validate_video(video: Path) -> None:
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"input must be a video ({', '.join(sorted(VIDEO_EXTENSIONS))}): {video}")


def _transcribe_stage(
    cache: ArtifactCache,
    name: str,
    audio: Path,
    srt_path: Path,
    words_path: Path,
    parameters: dict,
    runner,
) -> None:
    cache.run(
        name,
        [srt_path, words_path],
        {**parameters, "audio": input_fingerprint(audio)},
        lambda: save_transcript(*runner(), srt_path, words_path),
    )


def prepare(
    video: Path,
    language: str = "sr",
    intro_skip: float = 0.0,
    workdir: Path | None = None,
    output: Path | None = None,
    use_resolve: bool = False,
    resolve_amount: int = 50,
    resolve_app: Path = DEFAULT_RESOLVE_APP,
    faster_device: str = "cpu",
    faster_compute: str = "int8",
    vad_threshold: float = 0.25,
    dry_run: bool = False,
) -> dict:
    video = video.expanduser().resolve()
    validate_video(video)
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"language {language!r} is not supported by the ensemble aligner "
            f"({CLASSLA_MODEL} is Serbian). Supported: {supported}. "
            f"For other languages use the portable extract.py instead."
        )
    require_executable("ffmpeg")
    workdir = (workdir or default_workdir(video)).expanduser().resolve()
    output = (output or default_output(video)).expanduser().resolve()
    english, corrections = sidecars_for(video)

    resolve_available = use_resolve and resolve_app.exists()
    plan = {
        "phase": "prepare",
        "video": str(video),
        "workdir": str(workdir),
        "output": str(output),
        "language": language,
        "intro_skip": intro_skip,
        "branches": [
            "original_mlx_large",
            "original_mlx_turbo",
            "original_faster_whisper",
            "slow70_mlx",
            "slow50_mlx",
        ] + (["resolve21_vi50_mlx"] if resolve_available else []),
        "resolve_requested": use_resolve,
        "resolve_available": resolve_available,
        "english_sidecar": str(english) if english else None,
        "corrections_sidecar": str(corrections) if corrections else None,
    }
    if dry_run:
        return plan

    workdir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths.from_workdir(workdir)
    paths.audio.mkdir(parents=True, exist_ok=True)
    paths.asr.mkdir(parents=True, exist_ok=True)

    heavy_work = detect_heavy_work()
    manifest = base_manifest(video, language, intro_skip, heavy_work)
    manifest["resolve"] = {
        "requested": use_resolve,
        "available": resolve_available,
        "amount": resolve_amount,
        "app": str(resolve_app),
    }
    manifest["warnings"] = []
    if use_resolve and not resolve_available:
        manifest["warnings"].append(f"Resolve branch skipped; app not found: {resolve_app}")
    cache = ArtifactCache(workdir, manifest["benchmark_eligible"])

    original_wav = paths.audio / "original-16k-mono.wav"
    cache.run(
        "extract_audio",
        [original_wav],
        {"input": input_fingerprint(video), "sample_rate": 16000, "channels": 1},
        lambda: extract_audio(video, original_wav),
    )

    slow_wavs = {0.7: paths.audio / "original-slow70.wav", 0.5: paths.audio / "original-slow50.wav"}
    for speed, slowed in slow_wavs.items():
        cache.run(
            f"slow_{int(speed * 100)}",
            [slowed],
            {"source": input_fingerprint(original_wav), "speed": speed},
            lambda speed=speed, slowed=slowed: slow_audio(original_wav, slowed, speed),
        )

    branch_paths: dict[str, Path] = {}

    def mlx_branch(key: str, audio: Path, repo: str) -> Path:
        srt_path = paths.asr / f"{key}.raw.srt"
        words_path = paths.asr / f"{key}.words.json"
        _transcribe_stage(
            cache,
            f"asr_{key}",
            audio,
            srt_path,
            words_path,
            {"engine": "mlx-whisper", "repo": repo, "language": language},
            lambda: transcribe_mlx(audio, repo, language),
        )
        branch_paths[key] = words_path
        return words_path

    mlx_branch("original_mlx_large", original_wav, MLX_LARGE_REPO)
    mlx_branch("original_mlx_turbo", original_wav, MLX_TURBO_REPO)

    fw_srt = paths.asr / "original_faster_whisper.raw.srt"
    fw_words = paths.asr / "original_faster_whisper.words.json"
    _transcribe_stage(
        cache,
        "asr_original_faster_whisper",
        original_wav,
        fw_srt,
        fw_words,
        {
            "engine": "faster-whisper",
            "model": FASTER_WHISPER_MODEL,
            "language": language,
            "device": faster_device,
            "compute": faster_compute,
        },
        lambda: transcribe_faster_whisper(
            original_wav,
            FASTER_WHISPER_MODEL,
            language,
            faster_device,
            faster_compute,
        ),
    )
    branch_paths["original_faster_whisper"] = fw_words

    for speed, key in ((0.7, "slow70_mlx"), (0.5, "slow50_mlx")):
        raw_words = mlx_branch(f"{key}_raw", slow_wavs[speed], MLX_LARGE_REPO)
        scaled_words = paths.asr / f"{key}.words.json"
        cache.run(
            f"rescale_{key}",
            [scaled_words],
            {"source": input_fingerprint(raw_words), "factor": speed},
            lambda raw_words=raw_words, scaled_words=scaled_words, speed=speed: write_words(
                rescale_words(read_words(raw_words), speed), scaled_words
            ),
        )
        branch_paths.pop(f"{key}_raw")
        branch_paths[key] = scaled_words

    if resolve_available:
        resolve_wav = paths.audio / f"resolve21-vi{resolve_amount}-{PIPELINE_VERSION}.wav"
        resolve_version = paths.audio / "resolve-version.txt"

        def run_resolve() -> None:
            version = render_voice_isolation(original_wav, resolve_wav, resolve_amount, resolve_app)
            resolve_version.write_text(version, encoding="utf-8")

        try:
            cache.run(
                "resolve_voice_isolation",
                [resolve_wav, resolve_version],
                {
                    "source": input_fingerprint(original_wav),
                    "amount": resolve_amount,
                    "pipeline": PIPELINE_VERSION,
                    "app": str(resolve_app),
                },
                run_resolve,
            )
            mlx_branch("resolve21_vi50_mlx", resolve_wav, MLX_LARGE_REPO)
            manifest["resolve"]["version"] = resolve_version.read_text(encoding="utf-8").strip()
        except Exception as error:
            manifest["warnings"].append(f"Resolve branch skipped after error: {error}")
            manifest["resolve"]["error"] = str(error)

    vad_path = workdir / "vad-regions.json"
    cache.run(
        "vad",
        [vad_path],
        {"source": input_fingerprint(original_wav), "threshold": vad_threshold},
        lambda: write_json(
            vad_path,
            [
                {"start": round(start, 3), "end": round(end, 3)}
                for start, end in speech_regions(original_wav, vad_threshold)
            ],
        ),
    )
    vad_payload = json.loads(vad_path.read_text(encoding="utf-8"))
    regions = [(item["start"], item["end"]) for item in vad_payload]
    sources = {key: read_words(path) for key, path in branch_paths.items()}

    source_fingerprints = {key: input_fingerprint(path) for key, path in branch_paths.items()}
    window_parameters = {
        "vad": input_fingerprint(vad_path),
        "sources": source_fingerprints,
        "english": input_fingerprint(english) if english else None,
        "corrections": input_fingerprint(corrections) if corrections else None,
        "max_assign_distance": 1.5,
    }
    cache.run(
        "build_ensemble_windows",
        [paths.windows],
        window_parameters,
        lambda: write_json(
            paths.windows,
            build_ensemble_windows(regions, sources, english, corrections),
        ),
    )
    windows = json.loads(paths.windows.read_text(encoding="utf-8"))
    agent_outputs = [workdir / "agent-plan.json"] + [
        workdir / "resolver-input" / f"part-{part}.json" for part in range(1, 5)
    ]
    cache.run(
        "write_agent_inputs",
        agent_outputs,
        {"windows": input_fingerprint(paths.windows), "agents": 4},
        lambda: write_agent_inputs(windows, workdir, 4),
    )
    existing_results = sorted((workdir / "resolver-results").glob("part-*.json"))
    if existing_results:
        backup_dir = workdir / "obsolete" / f"{time.time_ns()}-resolver-results"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for result_path in existing_results:
            shutil.move(str(result_path), backup_dir / result_path.name)

    run_config = {
        **plan,
        "resolve_used": "resolve21_vi50_mlx" in branch_paths,
        "resolver_outputs_archived": len(existing_results),
        "original_wav": str(original_wav),
        "vad_regions": str(vad_path),
        "ensemble_windows": str(paths.windows),
        "prepared_at": utc_now(),
    }
    write_json(paths.config, run_config)
    manifest["stages"].update(cache.records)
    manifest["prepared_at"] = utc_now()
    manifest["window_count"] = len(windows)
    manifest["branch_count"] = len(branch_paths)
    write_json(paths.manifest, manifest)
    return {**plan, "window_count": len(windows), "agent_plan": str(workdir / "agent-plan.json")}


def _unresolved_markdown(video: str, unresolved: list[dict], alignment_failures: list[int]) -> str:
    lines = [
        f"# Unresolved subtitle windows: {Path(video).name}",
        "",
        "These windows were not lexically resolved by independent branch consensus. The final SRT uses a branch-medoid fallback where possible.",
        "",
        f"- Unresolved/medoid windows: {len(unresolved)}",
        f"- CLASSLA windows with no aligned words: {len(alignment_failures)}",
        "",
    ]
    for item in unresolved:
        lines.extend(
            [
                f"## Window {item['window_id']} ({item['start']:.3f}-{item['end']:.3f}s)",
                "",
                f"- Final fallback: `{item['text']}`",
                f"- Mode: `{item['mode']}`",
                f"- Resolver reason: {item.get('reason') or 'not provided'}",
                "",
            ]
        )
    if alignment_failures:
        lines.extend(["## CLASSLA alignment failures", "", ", ".join(map(str, alignment_failures)), ""])
    return "\n".join(lines)


_ORIGINAL_AUDIO_BRANCHES = (
    "original_faster_whisper",
    "original_mlx_large",
    "original_mlx_turbo",
)


def _branch_word_fallback(
    paths: "RunPaths",
    windows: list[dict],
    selected: list[dict],
    failed_ids: list[int],
) -> list[dict]:
    """Timing for CLASSLA-failed windows. Keep the clean ensemble-selected text and
    spread its tokens across the speech span the closest original-audio ASR branch
    detected inside the window. This preserves ensemble text quality (no single-branch
    stutter) while giving sub-window timing instead of one coarse whole-window cue."""
    if not failed_ids:
        return []
    branch_words: dict[str, list[dict]] = {}
    for branch in _ORIGINAL_AUDIO_BRANCHES:
        path = paths.asr / f"{branch}.words.json"
        if path.exists():
            branch_words[branch] = json.loads(path.read_text(encoding="utf-8"))
    window_by_id = {window["window_id"]: window for window in windows}
    selected_by_id = {item["window_id"]: item for item in selected}

    def similarity(left: str, right: str) -> float:
        a, b = set(tokens(left)), set(tokens(right))
        return len(a & b) / len(a | b) if a and b else 0.0

    out: list[dict] = []
    for window_id in failed_ids:
        window = window_by_id.get(window_id)
        chosen = selected_by_id.get(window_id)
        if not window or not chosen:
            continue
        pieces = chosen.get("text", "").split()
        if not pieces:
            continue
        start, end = float(window["start"]), float(window["end"])
        span_start, span_end = start, end
        if branch_words:
            branch = max(
                branch_words,
                key=lambda name: similarity(window["sources"].get(name, ""), chosen["text"]),
            )
            inside = [
                w for w in branch_words[branch]
                if w.get("start") is not None and w.get("end") is not None
                and start <= (float(w["start"]) + float(w["end"])) / 2 <= end
            ]
            if inside:
                span_start = max(start, float(inside[0]["start"]))
                span_end = min(end, float(inside[-1]["end"]))
        step = max(0.12, (span_end - span_start) / len(pieces))
        for index, piece in enumerate(pieces):
            word_start = span_start + index * step
            out.append(
                {
                    "start": round(word_start, 3),
                    "end": round(word_start + step, 3),
                    "word": piece,
                    "prob": 0.0,
                }
            )
    return out


def _dedup_near_words(words: list[dict], window: float = 1.2) -> list[dict]:
    """Drop a word if the same normalized token was already kept within `window`
    seconds. Removes duplicates where a CLASSLA-aligned word and a synthetic
    fallback word (from overlapping ensemble windows) cover the same speech."""
    kept: list[dict] = []
    last_time: dict[str, float] = {}
    for word in sorted(words, key=lambda w: (w["start"], w["end"])):
        key = "".join(ch for ch in str(word["word"]).lower() if ch.isalnum())
        if key and key in last_time and word["start"] - last_time[key] <= window:
            continue
        kept.append(word)
        if key:
            last_time[key] = word["start"]
    return kept


def finalize(workdir: Path, dry_run: bool = False) -> dict:
    workdir = workdir.expanduser().resolve()
    paths = RunPaths.from_workdir(workdir)
    if not paths.config.exists() or not paths.windows.exists() or not paths.manifest.exists():
        raise FileNotFoundError(f"prepare artifacts are incomplete in {workdir}")
    config = load_json(paths.config)
    manifest = load_json(paths.manifest)
    windows = json.loads(paths.windows.read_text(encoding="utf-8"))
    missing_results = [
        str(workdir / "resolver-results" / f"part-{part}.json")
        for part in range(1, 5)
        if not (workdir / "resolver-results" / f"part-{part}.json").exists()
    ]
    plan = {
        "phase": "finalize",
        "workdir": str(workdir),
        "window_count": len(windows),
        "missing_resolver_results": missing_results,
        "output": config["output"],
    }
    if dry_run:
        return plan
    if missing_results:
        raise FileNotFoundError("missing resolver result files: " + ", ".join(missing_results))

    finalize_heavy_work = detect_heavy_work(include_resolve=False)
    if finalize_heavy_work:
        existing = manifest.get("concurrent_heavy_work_warning", [])
        manifest["concurrent_heavy_work_warning"] = list(dict.fromkeys(existing + finalize_heavy_work))
        manifest["benchmark_eligible"] = False
        for stage in manifest.get("stages", {}).values():
            stage["seconds"] = None
    if "finalized_at" not in manifest:
        prepared_at = datetime.fromisoformat(config["prepared_at"])
        resolver_seconds = round((datetime.now(timezone.utc) - prepared_at).total_seconds(), 3)
        manifest.setdefault("stages", {})["agent_resolution"] = {
            "status": "completed",
            "seconds": resolver_seconds if manifest.get("benchmark_eligible") else None,
            "agents": 4,
        }
    cache = ArtifactCache(workdir, bool(manifest.get("benchmark_eligible")))
    results = merge_resolver_results(workdir, windows)
    selected_path = paths.selected
    unresolved_work_path = workdir / "unresolved.json"

    def choose_text() -> None:
        selected, unresolved = conservative_finalize(windows, results)
        write_json(selected_path, selected)
        write_json(unresolved_work_path, unresolved)

    result_inputs = {
        f"part-{part}": input_fingerprint(workdir / "resolver-results" / f"part-{part}.json")
        for part in range(1, 5)
    }
    cache.run(
        "conservative_finalize",
        [selected_path, unresolved_work_path],
        {"windows": input_fingerprint(paths.windows), "results": result_inputs, "logic": "keep-nonempty-v2"},
        choose_text,
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    unresolved = json.loads(unresolved_work_path.read_text(encoding="utf-8"))

    aligned_words_path = workdir / "aligned.words.json"
    fallback_path = workdir / "alignment-fallback-cues.json"
    alignment_failures_path = workdir / "alignment-failures.json"
    original_wav = Path(config["original_wav"])

    def run_alignment() -> None:
        words, fallback_cues, failed = align_selected_windows(
            original_wav,
            selected,
            CLASSLA_MODEL,
            config["language"],
            "cpu",
        )
        write_words(words, aligned_words_path)
        write_json(
            fallback_path,
            [{"start": cue.start, "end": cue.end, "text": cue.text} for cue in fallback_cues],
        )
        write_json(alignment_failures_path, failed)

    cache.run(
        "classla_alignment",
        [aligned_words_path, fallback_path, alignment_failures_path],
        {
            "audio": input_fingerprint(original_wav),
            "selected": input_fingerprint(selected_path),
            "model": CLASSLA_MODEL,
            "language": config["language"],
        },
        run_alignment,
    )

    output_srt = Path(config["output"])
    unresolved_json = output_srt.with_suffix(".unresolved.json")
    unresolved_md = output_srt.with_suffix(".unresolved.md")
    output_manifest = output_srt.with_suffix(".manifest.json")
    vad_payload = json.loads(Path(config["vad_regions"]).read_text(encoding="utf-8"))
    speech = [(item["start"], item["end"]) for item in vad_payload]
    alignment_failures = json.loads(alignment_failures_path.read_text(encoding="utf-8"))

    # For windows CLASSLA could not align (fast/noisy speech), do not dump the whole
    # window as one cue. Instead reuse word timestamps from the original-audio ASR
    # branch whose text is closest to the selected text — real sub-window timing.
    fallback_words = _branch_word_fallback(paths, windows, selected, alignment_failures)

    def build_final_output() -> None:
        merged = sorted(read_words(aligned_words_path) + fallback_words,
                        key=lambda w: (w["start"], w["end"]))
        cues = build_cues(_dedup_near_words(merged))
        cues = clean_cues(cues, float(config["intro_skip"]))
        cues = clamp_cues(cues, speech)
        write_srt(cues, output_srt)
        unresolved_payload = {
            "video": config["video"],
            "count": len(unresolved),
            "alignment_failures": alignment_failures,
            "windows": unresolved,
        }
        write_json(unresolved_json, unresolved_payload)
        unresolved_md.parent.mkdir(parents=True, exist_ok=True)
        unresolved_md.write_text(
            _unresolved_markdown(config["video"], unresolved, alignment_failures),
            encoding="utf-8",
        )
        manifest["cue_stats"] = cue_stats(cues)

    cache.run(
        "cue_build_clean_vad_clamp",
        [output_srt, unresolved_json, unresolved_md],
        {
            "aligned_words": input_fingerprint(aligned_words_path),
            "windows": input_fingerprint(paths.windows),
            "alignment_failures": alignment_failures,
            "vad": input_fingerprint(Path(config["vad_regions"])),
            "intro_skip": config["intro_skip"],
            "profile": "speech-tight-42x2-17cps-branchfallback3-dedup",
        },
        build_final_output,
    )

    manifest["stages"].update(cache.records)
    manifest["finalized_at"] = utc_now()
    manifest["output"] = {
        "srt": str(output_srt),
        "unresolved_json": str(unresolved_json),
        "unresolved_markdown": str(unresolved_md),
        "manifest": str(output_manifest),
    }
    manifest["unresolved_count"] = len(unresolved)
    manifest["alignment_failure_count"] = len(alignment_failures)
    write_json(paths.manifest, manifest)
    write_json(output_manifest, manifest)
    return {
        **plan,
        "output": str(output_srt),
        "unresolved_json": str(unresolved_json),
        "unresolved_markdown": str(unresolved_md),
        "manifest": str(output_manifest),
        "unresolved_count": len(unresolved),
        "alignment_failure_count": len(alignment_failures),
    }
