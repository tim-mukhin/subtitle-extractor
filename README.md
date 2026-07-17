# subtitle-extractor

Generate **accurately-timed** subtitles from any video or audio, locally, with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

The point is the **timing**. Most Whisper-based subtitle tools produce lines that
pop up before a character speaks and hang on screen through the silence after
them — because Whisper's segment timestamps span pauses. This tool throws away
Whisper's segmentation and rebuilds each cue from **word-level** timestamps,
cutting on real pauses. A subtitle then appears exactly when speech starts and
disappears when it stops.

## Quick start

```bash
# 1. install (needs ffmpeg on PATH: brew install ffmpeg)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. drop a video in input/ and run
python extract.py input/movie.mp4

# 3. subtitle lands in output/movie.srt
```

Options:

```bash
python extract.py input/movie.mp4 --lang sr        # force a language (else auto-detect)
python extract.py input/movie.mp4 --model large-v3 # any faster-whisper model
python extract.py input/episode.mkv --intro-skip 88  # drop an 88s title sequence
python extract.py input/movie.mp4 --device cuda --compute float16  # GPU
```

First run downloads the model (`large-v3` ≈ 3 GB) into the faster-whisper cache.

## How it works

```
video/audio ──ffmpeg──► 16k mono wav
            ──faster-whisper large-v3, word_timestamps──► words
            ──resegment (pause / punctuation / length)──► cues
            ──clean (intro, dedup, word-spam)──► .srt
```

The whole thing is deterministic and lives in one file, [`extract.py`](extract.py).
Tune the cue splitting with `--max-gap` (pause that ends a cue), `--max-dur`, and
`--max-chars`.

## Measure it

`evaluate_timing.py` scores a subtitle track against speech detected by Silero VAD
— useful for comparing "before/after" or tuning parameters:

```bash
pip install -r requirements-eval.txt
python evaluate_timing.py --audio input/movie.mp4 \
    --srt "old=old.srt" "new=output/movie.srt"
```

It reports `hang_sec` (subtitle time sitting over silence — the bug this tool
fixes), `in_speech_pct`, `onset_mae`, and `cover_pct`.

## Design notes (what we tested)

- **Removing background music does NOT fix timing.** We compared Demucs, BS-Roformer,
  and DaVinci Resolve Voice Isolation as a pre-step. None improved subtitle timing;
  on dialogue-heavy audio aggressive separation *hurt* (it clipped quiet speech).
  The hanging-subtitle problem is a segmentation problem, not an audio-cleanliness
  problem. Vocal isolation is left out on purpose.
- **The engine barely matters for text.** faster-whisper, WhisperX and stable-ts all
  run the same Whisper model underneath, so the transcript is ~identical; they differ
  only in segmentation. That is exactly what this tool replaces.
- **Word timestamps + pause-based re-segmentation** is the lever. It cut on-screen-in-
  silence time by 3–8× versus raw Whisper segments in our tests.

---

A tiny wrapper around open-source tools ([faster-whisper](https://github.com/SYSTRAN/faster-whisper),
ffmpeg, [Silero VAD](https://github.com/snakers4/silero-vad)). Use it however you like.

