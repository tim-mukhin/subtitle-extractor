---
name: subtitle-ensemble
description: Run the cached local subtitle ensemble pipeline for one video, resolve every neutral VAD window with four fresh Claude Code agents, then produce final SRT and unresolved reports. Use when the user invokes /subtitle-ensemble.
argument-hint: "<video> --lang sr [--intro-skip 88] [--resolve]"
allowed-tools: Bash, Read, Agent
---

# Subtitle ensemble

Run one complete prepare -> four-agent resolution -> finalize workflow. The Python stages use local ASR/alignment models only. Do not add or call cloud ASR/LLM APIs, MCP servers, or the Anthropic SDK. The four Agent calls are part of the current Claude Code session and communicate only through local files.

## 1. Parse and prepare

Treat `$ARGUMENTS` as CLI arguments. A video path is required. Preserve quoted paths and pass optional flags through. The current Serbian profile defaults to `--lang sr`; do not silently claim another language profile is calibrated.

Find the repository root with `git rev-parse --show-toplevel`, then run:

```bash
python3 "$ROOT/subtitle_ensemble.py" prepare <video-and-options>
```

This is a long local-model process. Run it in the background and wait for the completion notification; do not poll. If it fails, report the failing stage and stop. Do not delete or overwrite workdir artifacts manually. Cached stages are reused, and stale generated artifacts are moved under the run's `obsolete/` directory.

Read the returned `agent_plan` JSON. It must contain exactly four entries and cover all windows. Each entry names one input JSON and one distinct output JSON.

## 2. Resolve with exactly four fresh Agents

Launch exactly four new Agent tool calls in one parallel tool invocation, one per plan entry. Use `subagent_type: general-purpose`, `run_in_background: true`, and no worktree isolation. Do not reuse an earlier agent with SendMessage. Wait for all four completion notifications; do not poll files while they run.

Use this prompt for each agent, substituting only its part number and input/output paths:

```text
Resolve subtitle ensemble part N/4. This is a fresh independent pass.

Read INPUT_JSON. It is a JSON array of neutral original-audio VAD windows. Write OUTPUT_JSON as one JSON array with exactly one result per input window, preserving every window_id exactly once and in order. Do not edit any other file. Do not call web tools, cloud APIs, MCP, or other agents.

Evidence rules:
- No source is primary. Never copy one transcript wholesale merely because it is MLX large or faster-whisper.
- Cap correlated evidence by family: A/original = original_mlx_large + original_mlx_turbo + original_faster_whisper; B/slowed = slow70_mlx + slow50_mlx; C/Resolve = resolve21_vi50_mlx when present. Three similar A decoders do not outvote B and C by raw count.
- repeated_ngram_flags are source-specific hallucination evidence. Reject flagged distant repetitions, consecutive token runs, and one-token dominance.
- If curated_override is not null, copy it exactly into selected_text and use status curated. Do not rewrite it.
- English cues are semantic-only context. They may disambiguate broad meaning or grammatical person, but never vote for exact Serbian words, add missing words, or override acoustic Serbian evidence.
- Use status consensus only for text supported by independent acoustic families without a material contradiction. Use unresolved when the lexical form is uncertain; do not invent a fluent reconstruction or shorten the evidence to make it look resolved.
- Use no_speech only when no stable Serbian speech has cross-family support. A single ASR phrase in a VAD window is not enough.
- Preserve Serbian Latin script. Transliterate any remaining Serbian Cyrillic, but do not translate.

Each output object must contain:
{
  "window_id": integer,
  "start": number,
  "end": number,
  "selected_text": string,
  "status": "consensus" | "curated" | "unresolved" | "no_speech",
  "confidence": number from 0 to 1,
  "branch_support": array containing zero or more of "original", "slowed", "resolve",
  "alternatives": array of strings,
  "reason": short string
}

Before finishing, validate that output length equals input length, IDs are unique, and first/last IDs match this part. Write valid UTF-8 JSON directly to OUTPUT_JSON, with no Markdown wrapper.
```

If one agent fails to create valid output, launch one new replacement Agent for only that same part. Never let another part rewrite it.

## 3. Finalize

After all four outputs exist, run the exact workdir returned by prepare:

```bash
python3 "$ROOT/subtitle_ensemble.py" finalize --workdir "$WORKDIR"
```

This is also a long local-model stage because it runs CLASSLA alignment. Run it in the background and wait for the completion notification; do not poll.

Finalization must remain deterministic: curated/consensus text is accepted, unresolved text falls back to a correlation-capped branch medoid, suspicious repetitions are filtered, and no-speech is recovered only with cross-family agreement. CLASSLA aligns against original audio, then cue building, intro cleanup, and final VAD clamp produce the SRT.

Report the returned paths for:
- final SRT;
- unresolved JSON;
- unresolved Markdown;
- manifest.

Also report whether `benchmark_eligible` is false because prepare detected concurrent heavy work. Do not describe such a run's stage durations as a benchmark.
