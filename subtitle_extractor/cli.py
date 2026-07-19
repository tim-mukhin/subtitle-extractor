from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .constants import DEFAULT_RESOLVE_APP
from .pipeline import finalize, prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-ensemble",
        description="Cached local multi-branch subtitle ensemble (current profile: Serbian).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="extract audio, run cached ASR branches, VAD, and write four agent inputs",
    )
    prepare_parser.add_argument("video", type=Path, help="input video file")
    prepare_parser.add_argument("--lang", default="sr", help="language code (default: sr)")
    prepare_parser.add_argument("--intro-skip", type=float, default=0.0, help="drop cues starting before this second")
    prepare_parser.add_argument("--workdir", type=Path, help="artifact/cache directory")
    prepare_parser.add_argument("--output", type=Path, help="final SRT path (outside workdir)")
    prepare_parser.add_argument("--resolve", action="store_true", help="add optional Resolve 21 Voice Isolation branch")
    prepare_parser.add_argument("--resolve-amount", type=int, default=50, help="Voice Isolation amount (default: 50)")
    prepare_parser.add_argument("--resolve-app", type=Path, default=DEFAULT_RESOLVE_APP, help="path to Resolve executable")
    prepare_parser.add_argument("--faster-device", default="cpu", help="faster-whisper device (default: cpu)")
    prepare_parser.add_argument("--faster-compute", default="int8", help="faster-whisper compute type (default: int8)")
    prepare_parser.add_argument("--vad-threshold", type=float, default=0.25, help="permissive Silero threshold (default: 0.25)")
    prepare_parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without models")

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="merge four resolver outputs, align, build cues, clean, and clamp",
    )
    finalize_parser.add_argument("--workdir", type=Path, required=True, help="workdir produced by prepare")
    finalize_parser.add_argument("--dry-run", action="store_true", help="validate prepared/resolver artifacts without models")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "prepare":
        result = prepare(
            args.video,
            language=args.lang,
            intro_skip=args.intro_skip,
            workdir=args.workdir,
            output=args.output,
            use_resolve=args.resolve,
            resolve_amount=args.resolve_amount,
            resolve_app=args.resolve_app,
            faster_device=args.faster_device,
            faster_compute=args.faster_compute,
            vad_threshold=args.vad_threshold,
            dry_run=args.dry_run,
        )
    else:
        result = finalize(args.workdir, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
