#!/usr/bin/env python
"""
Optional timing evaluator — how well does a subtitle track line up with speech?

Ground truth = Silero VAD run on the audio: where speech actually is. We then
measure each .srt against it:

  in_speech_pct  share of subtitle-on-screen time that overlaps real speech (higher better)
  hang_sec       subtitle-on-screen time sitting over silence (lower better; the "hanging" bug)
  onset_mae      mean |cue start - nearest speech onset| (lower = tighter start)
  cover_pct      share of speech covered by subtitles (higher = fewer misses)

Usage:
    python evaluate_timing.py --audio input/movie.mp4 --srt output/movie.srt
    python evaluate_timing.py --audio a.wav --srt "old=a.old.srt" "new=a.new.srt"

Needs: pip install -r requirements-eval.txt  (torch, numpy) + ffmpeg
"""
import argparse
import re
import subprocess
from pathlib import Path


def load_audio(path, sr=16000):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def vad_speech(path):
    import torch
    torch.set_num_threads(4)
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]
    sr = 16000
    audio = torch.from_numpy(load_audio(path, sr))
    ts = get_speech_timestamps(audio, model, sampling_rate=sr,
                               min_silence_duration_ms=300, min_speech_duration_ms=120,
                               speech_pad_ms=60)
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def parse_srt(path):
    cues = []
    for block in re.split(r"\n\n+", Path(path).read_text(encoding="utf-8").strip()):
        m = re.search(r"(\d+:\d+:\d+[.,]\d+)\s*-->\s*(\d+:\d+:\d+[.,]\d+)", block)
        if not m:
            continue

        def p(ts):
            h, mnt, rest = ts.replace(",", ".").split(":")
            return int(h) * 3600 + int(mnt) * 60 + float(rest)
        cues.append((p(m.group(1)), p(m.group(2))))
    return cues


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def score(cues, speech):
    total_cue = sum(e - s for s, e in cues) or 1e-9
    total_speech = sum(e - s for s, e in speech) or 1e-9
    in_speech = sum(_ov(cs, ce, ss, se) for cs, ce in cues for ss, se in speech)
    starts = [s for s, _ in speech]
    onset = [min(abs(cs - ss) for ss in starts) for cs, _ in cues] if starts else []
    covered = sum(min(sum(_ov(ss, se, cs, ce) for cs, ce in cues), se - ss) for ss, se in speech)
    return {
        "cues": len(cues),
        "in_speech_pct": round(100 * in_speech / total_cue, 1),
        "hang_sec": round(total_cue - in_speech, 1),
        "onset_mae": round(sum(onset) / len(onset), 2) if onset else 0.0,
        "cover_pct": round(100 * covered / total_speech, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt", nargs="+", required=True, help="path or label=path")
    args = ap.parse_args()

    speech = vad_speech(args.audio)
    print(f"VAD: {len(speech)} speech segments, {sum(e - s for s, e in speech):.1f}s speech\n")
    hdr = f"{'track':<22}{'cues':>6}{'in_speech%':>12}{'hang_s':>8}{'onset_mae':>11}{'cover%':>8}"
    print(hdr); print("-" * len(hdr))
    for item in args.srt:
        label, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        r = score(parse_srt(path), speech)
        print(f"{label:<22}{r['cues']:>6}{r['in_speech_pct']:>12}{r['hang_sec']:>8}"
              f"{r['onset_mae']:>11}{r['cover_pct']:>8}")


if __name__ == "__main__":
    main()
