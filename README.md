# subtitle-extractor

Generate tightly timed subtitles from video. The repository now has two workflows:

1. `extract.py` - the original portable faster-whisper script. It stays small, deterministic, and independent of the ensemble package.
2. `/subtitle-ensemble` - a cached Claude Code workflow that combines several local acoustic branches, resolves neutral VAD windows with four fresh Agents, then aligns the selected Serbian text to the original audio with CLASSLA.

The current calibrated ensemble profile is **Serbian Latin subtitles on Apple Silicon**. The underlying separation of transcript, forced alignment, and cue building is extensible, but other language profiles need their own aligner and evaluation before being described as supported.

## Portable quick start

```bash
# ffmpeg must be on PATH: brew install ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python extract.py input/movie.mp4 --lang sr
python extract.py input/episode.mkv --intro-skip 88
```

Output lands in `output/<name>.srt`. `extract.py` uses faster-whisper word timestamps, rebuilds cues around pauses, and removes intro/duplicate/repetition junk. Its CLI and dependency file remain unchanged.

Runtime for the portable path is dominated by faster-whisper. On CPU it runs at roughly real time: about **40 minutes for a 48-minute episode** on an Apple M4 Max (`--compute int8` needs ~2-3 GB RAM; `--compute float32` needs ~9-13 GB). With an NVIDIA GPU (`--device cuda --compute float16`) the same episode takes a few minutes. The first run also downloads the `large-v3` model (~3 GB) into the faster-whisper cache.

## Ensemble quick start

Open this repository in Claude Code, install the optional dependencies, and invoke the project skill:

```bash
pip install -r requirements-ensemble.txt

/subtitle-ensemble input/video.mp4 --lang sr
/subtitle-ensemble input/video.mp4 --lang sr --intro-skip 88
```

The skill performs one automatic workflow:

1. `prepare` extracts a workdir-only WAV and runs cached full branches:
   - MLX Whisper large-v3 on original audio;
   - MLX Whisper large-v3-turbo on original audio;
   - faster-whisper large-v3 on original audio;
   - MLX large on 70% and 50% slowed audio, then rescales timestamps;
   - optional DaVinci Resolve 21 Voice Isolation 50 -> MLX large.
2. Permissive Silero VAD creates neutral original-audio windows. Rough ASR timestamps only assign text to windows; they never become final cue boundaries.
3. Serbian Cyrillic is transliterated to Latin and each source gets corpus-level repetition flags.
4. Four fresh Claude Code Agents resolve disjoint ranges. No source is primary, correlated decoder families are capped, English is semantic-only, and exact user corrections win.
5. `finalize` validates full window coverage, applies conservative branch-medoid fallback to unresolved windows, filters suspicious repetitions, and only recovers `no_speech` when independent branch families agree.
6. CLASSLA `classla/wav2vec2-xls-r-juznevesti-sr` aligns selected text against original audio. A deterministic speech-tight cue builder, intro cleanup, and final VAD clamp produce the SRT.

Outputs are separate from the workdir:

```text
output/video.ensemble.srt
output/video.ensemble.unresolved.json
output/video.ensemble.unresolved.md
output/video.ensemble.manifest.json
```

The manifest records model/package versions, cache status, branch/window counts, and stage timings. If ComfyUI, an Ollama runner, Resolve, or another known heavy process is already active, the run is marked `benchmark_eligible: false` and durations are not recorded as benchmark measurements.

### Validate without running models

```bash
python subtitle_ensemble.py --help
python subtitle_ensemble.py prepare input/video.mp4 --lang sr --intro-skip 88 --dry-run
```

The explicit phases used by the skill are:

```bash
python subtitle_ensemble.py prepare input/video.mp4 --lang sr
python subtitle_ensemble.py finalize --workdir .subtitle-ensemble/<run-id>
```

`prepare` writes `agent-plan.json` plus four `resolver-input/part-N.json` files. `finalize` requires four matching `resolver-results/part-N.json` files and rejects missing, duplicate, or extra window IDs.

## Cache and sidecars

By default, generated audio, transcripts, VAD regions, agent inputs/results, and alignment artifacts live under `.subtitle-ensemble/<video-id>/`. A repeated run reuses acoustic/alignment stages whose input fingerprints and parameters match. Because each invocation deliberately uses four fresh semantic resolvers, prior resolver outputs are moved to that run's `obsolete/` directory before new Agents start. Other stale generated artifacts are preserved there too rather than deleted.

Optional sidecars live next to the video:

- `video.en.srt` - English semantic context. It never votes for exact Serbian wording.
- `video.corrections.json` - exact curated overrides:

```json
[
  {"window_id": 23, "text": "A? A gde ide?"},
  {"window_id": 34, "text": ""}
]
```

Overrides match `window_id` exactly. There is no fuzzy cue/text matching.

## Optional Resolve branch

Resolve is off by default. Enable it explicitly:

```bash
/subtitle-ensemble input/video.mp4 --lang sr --resolve
```

Requirements and safety behavior:

- macOS and DaVinci Resolve Studio 21;
- launch is always `Resolve -nogui` - never GUI mode;
- default Voice Isolation amount is 50;
- rendered WAV names include the pipeline version;
- the pipeline does not delete Resolve projects, timelines, media, or render history;
- if Resolve is absent or the optional branch fails, the other five branches continue;
- if Resolve is already running without `-nogui`, the branch refuses to attach.

Resolve is secondary evidence, not a replacement for original audio. The final forced alignment always uses the original WAV.

## Dependencies and resources

`requirements.txt` is only for portable `extract.py`. `requirements-ensemble.txt` adds MLX Whisper, PyTorch/Silero VAD, WhisperX, Transformers, and CLASSLA model support.

First use downloads model weights into the normal local caches. Allow roughly:

- 8-12 GB of model cache across both MLX models, faster-whisper large-v3, and CLASSLA;
- 0.5-1.5 GB of workdir storage for a roughly 48-minute episode, depending on Resolve and retained artifacts;
- about 8-13 GB peak RAM for the measured individual branches on an M4 Max. Resolve needs additional memory. More headroom is recommended.

All ASR, VAD, slowdown, Voice Isolation, and forced alignment run locally. The repository does not call cloud ASR/LLM APIs and does not require API keys. The semantic resolver is deliberately a Claude Code skill: the four Agents run inside the user's existing Claude Code session and receive transcript-window JSON, not audio or video. This is not a standalone offline resolver.

## Reference result: Serbian Episode 01

The bake-off used one 48-minute episode of *Jutro će promeniti sve*. Silero VAD is a consistent proxy, not human gold.

| Track | Cues | In speech | Hang | Onset proxy | Coverage |
|---|---:|---:|---:|---:|---:|
| MLX + VAD + CLASSLA + clamp | 567 | 91.5% | 87.0s | 1.65s | 86.1% |
| Filtered full ensemble | 572 | **91.9%** | **83.7s** | **1.60s** | **87.4%** |

The ensemble removed known corpus hallucinations such as repeated `women`, `KOLA KOLA`, and `Hvala što pratite`, while preserving confirmed corrections. Ninety-three windows still required unresolved branch-medoid fallback in the reference run. The result was judged useful, not perfect, and it is not a human-gold transcript.

Preliminary cached-model end-to-end estimates for the reference machine were about 55-70 minutes with reasonable CPU/GPU overlap. The public workflow prioritizes reproducibility and cache safety over that estimate, so hardware, cache state, branch scheduling, and concurrent heavy work can make a run slower.

## Timing evaluator

`evaluate_timing.py` compares SRT tracks against Silero VAD:

```bash
pip install -r requirements-eval.txt
python evaluate_timing.py --audio input/movie.mp4 \
  --srt "old=old.srt" "new=output/movie.srt"
```

It reports subtitle time over speech, hang over silence, onset distance, and speech coverage. These metrics are useful for relative comparisons; they are not millisecond-accurate human ground truth.
