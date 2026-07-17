#!/usr/bin/env python
"""
subtitle-extractor — accurately-timed subtitles from any video or audio.

Pipeline:
    video/audio --(ffmpeg)--> 16k mono wav
                --(faster-whisper large-v3, word timestamps)--> words
                --(resegment by pause / punctuation / length)--> cues
                --(clean: drop intro, dedup, word-spam)--> .srt

Why this exists
---------------
Whisper's own segment timestamps span silences, so subtitles pop up before a line
is spoken and hang on screen through the pauses after it. The fix is NOT a better
model and NOT removing background music — it's throwing away Whisper's segmentation
and rebuilding cues from word-level timestamps, cutting on real pauses. A subtitle
then appears exactly when speech starts and disappears when it stops.

Usage
-----
    # drop videos into input/, then process all of them:
    python extract.py

    # or point at one file:
    python extract.py input/movie.mp4 --lang sr --model large-v3
    python extract.py input/episode.mkv --intro-skip 88   # drop an 88s title sequence

Audio extraction is internal (a temp wav that never touches input/ or output/).
Output goes to output/<name>.srt
"""
import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv"}


# ----------------------------------------------------------------------------- audio
def extract_audio(src: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)],
        check=True,
    )


# ------------------------------------------------------------------------- transcribe
def build_model(model_size: str, device: str, compute: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device=device, compute_type=compute)


def transcribe_words(model, wav: Path, lang):
    """faster-whisper with word-level timestamps. Returns [{start,end,word}]."""
    segments, info = model.transcribe(
        str(wav),
        language=lang,                     # None -> auto-detect
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=100),
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    print(f"[transcribe] language={info.language} ({info.language_probability:.0%}), "
          f"duration={info.duration:.0f}s", flush=True)

    words = []
    n = 0
    for seg in segments:
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
        n += 1
        if n % 100 == 0:
            print(f"  ...{n} segments, {len(words)} words", flush=True)
    print(f"[transcribe] {len(words)} words", flush=True)
    return words


# -------------------------------------------------------------------------- resegment
_PUNCT_NOSPACE = (",", ".", "!", "?", ":", ";")


def _join(words) -> str:
    return "".join((" " if not w["word"].startswith(_PUNCT_NOSPACE) else "") + w["word"]
                   for w in words).strip()


def resegment(words, max_gap=0.6, max_dur=6.0, max_chars=84, min_flush_dur=1.2,
              start_pad=0.03, end_pad=0.12):
    """Rebuild cues from word timestamps. Cut a cue when:
       - the pause before the next word > max_gap  (kills hanging subtitles),
       - the cue is longer than max_dur,
       - the line is longer than max_chars,
       - the last word ended a sentence and the cue is already long enough.
    Cue bounds = first word start .. last word end (padded, never overlapping next).
    """
    cues, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur = cur[-1]["end"] - cur[0]["start"]
            ends_sentence = bool(re.search(r"[.!?]$", cur[-1]["word"].strip()))
            if (gap > max_gap or dur >= max_dur
                    or (len(_join(cur)) >= max_chars and gap > 0.12)
                    or (ends_sentence and dur >= min_flush_dur)):
                cues.append(cur)
                cur = []
        cur.append(w)
    if cur:
        cues.append(cur)

    out = []
    for c in cues:
        out.append([c[0]["start"] - start_pad, c[-1]["end"] + end_pad, _join(c)])
    for i in range(len(out) - 1):                       # no overlap with next cue
        if out[i][1] > out[i + 1][0] - 0.02:
            out[i][1] = max(out[i][0] + 0.4, out[i + 1][0] - 0.02)
    return out


# ------------------------------------------------------------------------------ clean
def clean(cues, intro_skip=0.0):
    """Drop title-sequence junk, consecutive duplicates, and word-spam hallucinations."""
    out, prev = [], None
    for s, e, t in cues:
        if not re.search(r"\w", t):
            continue
        if s < intro_skip:
            continue
        norm = t.lower().strip(" .,!?…")
        if norm == prev:
            continue
        prev = norm
        toks = re.findall(r"\w+", t)
        if toks:
            _, top = Counter(w.lower() for w in toks).most_common(1)[0]
            if top >= 4 and len(toks) >= 5:             # one word repeated 4+ times
                continue
        out.append((s, e, t))
    return out


# -------------------------------------------------------------------------------- srt
def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def write_srt(cues, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{t}\n\n")


# ------------------------------------------------------------------------------- main
def process_one(model, src: Path, outdir: Path, lang, seg_opts, intro_skip):
    print(f"\n=== {src.name} ===", flush=True)
    with tempfile.TemporaryDirectory() as td:            # wav is internal, never leaks
        wav = Path(td) / f"{src.stem}.wav"
        print("[1/4] audio", flush=True)
        extract_audio(src, wav)
        print("[2/4] transcribe", flush=True)
        words = transcribe_words(model, wav, lang)
    print("[3/4] resegment + clean", flush=True)
    cues = clean(resegment(words, **seg_opts), intro_skip)
    srt = outdir / f"{src.stem}.srt"
    write_srt(cues, srt)
    print(f"[4/4] done -> {srt.name}  ({len(cues)} cues)", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Accurately-timed subtitles from video. Drop videos in input/, run, get output/*.srt.")
    ap.add_argument("input", nargs="?",
                    help="a video file. If omitted, processes every video in input/")
    ap.add_argument("--lang", default=None, help="ISO code (e.g. sr, en). Default: auto-detect")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model (default large-v3)")
    ap.add_argument("--output-dir", default=str(ROOT / "output"))
    ap.add_argument("--intro-skip", type=float, default=0.0,
                    help="seconds of title sequence to drop (default 0)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--compute", default="int8", help="int8 | float16 | float32")
    ap.add_argument("--max-gap", type=float, default=0.6, help="split cue on pause > this (s)")
    ap.add_argument("--max-dur", type=float, default=6.0, help="max cue duration (s)")
    ap.add_argument("--max-chars", type=int, default=84, help="max chars per cue")
    args = ap.parse_args()

    if args.input:
        sources = [Path(args.input).resolve()]
        if not sources[0].exists():
            sys.exit(f"not found: {sources[0]}")
    else:
        indir = ROOT / "input"
        sources = sorted(p for p in indir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not sources:
            sys.exit(f"drop a video into {indir}/ (or pass a path)")
        print(f"found {len(sources)} video(s) in input/", flush=True)

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    seg_opts = dict(max_gap=args.max_gap, max_dur=args.max_dur, max_chars=args.max_chars)

    print(f"loading model {args.model} ({args.device}/{args.compute})...", flush=True)
    model = build_model(args.model, args.device, args.compute)
    for src in sources:
        process_one(model, src, outdir, args.lang, seg_opts, args.intro_skip)


if __name__ == "__main__":
    main()
