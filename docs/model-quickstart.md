# Model Quickstart

This page is the copy-paste path for getting speech working quickly.

Use `swift run --package-path apps/ValarCLI valartts ...` if you have not installed `valartts` on your `PATH` yet. If you already have the executable installed, you can replace that prefix with `valartts ...`.

## Two-Command Newcomer Path

For the smallest first success in a clean clone:

```bash
make quickstart
make first-clip
```

That writes a WAV under your macOS temp directory by default, usually at `$TMPDIR/valar-first-clip.wav`. Override it with `VALAR_FIRST_CLIP_OUTPUT=/absolute/path.wav`.

## Native Prerequisites

The CLI and daemon require the Metal toolchain to build `mlx.metallib` for local inference.

```bash
bash tools/bootstrap.sh native
```

If `bash scripts/build_metallib.sh` fails because `metal` or `metallib` is missing, fix Xcode first:

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

If you already have an `mlx.metallib` from another MLX checkout or Python install, you can point Valar at it directly:

```bash
VALARTTS_METALLIB_FALLBACK_PATH=/absolute/path/to/mlx.metallib bash scripts/build_metallib.sh
```

If you also use another local Valar checkout on the same Mac, isolate this repo while testing:

```bash
export VALARTTS_HOME="$PWD/.valartts-public-home"
```

That repo-local state directory is gitignored in this public repo.

## Common Readiness Check

```bash
make quickstart
swift run --package-path apps/ValarCLI valartts models list
swift run --package-path apps/ValarCLI valartts models status
swift run --package-path apps/ValarCLI valartts doctor --json
```

Model state terms:

- `supported`: the model is in Valar's public catalog, but no local artifacts are registered yet.
- `cached`: a reusable upstream cache appears to be present, but Valar still needs `models install <id>` to register a model pack.
- `installed`: Valar has a registered local model pack, or the entry is a built-in system backup such as `apple/system-tts` or `apple/system-asr`.
- `resident`: the daemon has the model loaded or warming in the current runtime.

Useful cleanup commands:

```bash
swift run --package-path apps/ValarCLI valartts models cleanup --dry-run
swift run --package-path apps/ValarCLI valartts models cleanup --apply
swift run --package-path apps/ValarCLI valartts models remove <id>
swift run --package-path apps/ValarCLI valartts models purge-cache <id>
```

Use `cleanup --dry-run` before deleting anything. `models remove` removes Valar's installed model pack for one ID. `models purge-cache` removes shared Hugging Face cache entries for that ID and does not replace `models remove`.

## Project Import

Create a `.valarproject` bundle from a local TXT, Markdown, or simple dialogue script file:

```bash
swift run --package-path apps/ValarCLI valartts projects import ./script.md --split-mode markdown-headings
swift run --package-path apps/ValarCLI valartts projects lint
swift run --package-path apps/ValarCLI valartts projects info
swift run --package-path apps/ValarCLI valartts projects save
```

Supported split modes are `markdown-headings`, `paragraphs`, `lines`, `dialogue`, and `whole-document`. Dialogue mode understands simple speaker prefixes such as `[Narrator] text` and `Narrator: text`.

Use `projects lint` before long renders to check script markup, speaker labels, voice profile coverage, and selected model fit:

```bash
swift run --package-path apps/ValarCLI valartts projects lint --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
```

After creating chapter exports, write a portable manifest for review, handoff, or publishing workflows:

```bash
swift run --package-path apps/ValarCLI valartts projects export-pack
```

The export pack writes `valar-export-pack.json` under the project bundle by default. Artifact paths inside the manifest are relative to the `.valarproject` bundle root.

## One-Command Smoke Tests

These `make` shortcuts wrap the public validation script:

```bash
make validate-live
make validate-live-blessed
```

If you also want the bridge typecheck and Bun install path in the same pass:

```bash
make validate-bridge-live
make validate-bridge-live-blessed
```

## Soprano: Smallest First Run

```bash
make first-clip
```

Equivalent direct CLI commands:

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Soprano-1.1-80M-bf16 --allow-download
swift run --package-path apps/ValarCLI valartts speak \
  --model mlx-community/Soprano-1.1-80M-bf16 \
  --text "Hello from Soprano." \
  --output "${TMPDIR:-/tmp}/soprano.wav"
```

## Built-In Apple Backups

Valar exposes two no-download system backup IDs:

```bash
swift run --package-path apps/ValarCLI valartts speak \
  --model apple/system-tts \
  --text "Hello from the macOS system voice." \
  --output "${TMPDIR:-/tmp}/apple-system.wav"
```

`apple/system-tts` uses macOS `AVSpeechSynthesizer` voices. `apple/system-asr` uses on-device macOS Speech recognition and is intended for app-hosted workflows that can request Speech permission. For CLI and daemon transcription, install `mlx-community/Qwen3-ASR-0.6B-8bit`.

## Qwen: Main Lane

Qwen has three public TTS checkpoints:

- `Base`: main narrator lane, stable voices, and reference-audio/stabilized voice workflows.
- `VoiceDesign`: text-described voice creation for expressive short and medium clips.
- `CustomVoice`: official Qwen named-speaker lane.

For consistent long narration, prefer `VoiceDesign` first, then `voices stabilize`, then speak with the stabilized voice on the `Base` checkpoint. Let `voiceBehavior` stay on its default `auto` behavior unless you have a specific reason to force `expressive` or `stableNarrator`.

### Qwen TTS narrator

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --allow-download
swift run --package-path apps/ValarCLI valartts speak \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 \
  --text "Hello from the Qwen main lane." \
  --output "${TMPDIR:-/tmp}/qwen.wav"
```

### Qwen designed voice (text-only)

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 --allow-download
swift run --package-path apps/ValarCLI valartts voices design \
  --name "Warm Narrator" \
  --description "Warm, calm narrator with measured pacing."
swift run --package-path apps/ValarCLI valartts voices list
swift run --package-path apps/ValarCLI valartts speak \
  --voice <voice-uuid-from-voices-list> \
  --text "Hello from my designed Qwen voice." \
  --output "${TMPDIR:-/tmp}/qwen-designed.wav"
```

Use the UUID printed by `voices design` or shown in `voices list`.

### Qwen stable narrator from a designed voice

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --allow-download
swift run --package-path apps/ValarCLI valartts voices stabilize <voice-uuid-from-voices-list> --name "Warm Narrator Stable"
swift run --package-path apps/ValarCLI valartts voices list
swift run --package-path apps/ValarCLI valartts speak \
  --voice <stable-voice-uuid-from-voices-list> \
  --text "Hello from my stable Qwen narrator voice." \
  --output "${TMPDIR:-/tmp}/qwen-stable.wav"
```

Qwen `CustomVoice` remains the named-speaker lane, but the simplest public voice-creation path is `VoiceDesign` first and optional `stabilize` second.

### Qwen ASR

Replace `sample.wav` with a real local WAV or M4A clip from your Desktop, Downloads, Documents, Music, or `/tmp`.

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Qwen3-ASR-0.6B-8bit --allow-download
swift run --package-path apps/ValarCLI valartts transcribe \
  sample.wav \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --output "${TMPDIR:-/tmp}/transcript.txt"
```

### Qwen alignment

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/Qwen3-ForcedAligner-0.6B-8bit --allow-download
swift run --package-path apps/ValarCLI valartts align \
  sample.wav \
  --transcript @"${TMPDIR:-/tmp}/transcript.txt" \
  --model mlx-community/Qwen3-ForcedAligner-0.6B-8bit \
  --output "${TMPDIR:-/tmp}/alignment.json"
```

## VibeVoice: Preview Preset Voices

```bash
swift run --package-path apps/ValarCLI valartts models install mlx-community/VibeVoice-Realtime-0.5B-4bit --allow-download
swift run --package-path apps/ValarCLI valartts speak \
  --model mlx-community/VibeVoice-Realtime-0.5B-4bit \
  --voice random \
  --language en \
  --text "Hello from VibeVoice." \
  --output "${TMPDIR:-/tmp}/vibevoice.wav"
```

VibeVoice should be presented as preset-only, English-first, and preview-only. It is not the default narrator lane.
During install, Valar materializes the companion tokenizer from `Qwen/Qwen2.5-0.5B` automatically when the MLX VibeVoice pack does not ship it directly.

VibeVoice does not support local voice cloning, voice design, or reference-audio conditioning. Use Qwen for those workflows.

Preset voice names are:

```text
en-Carter_man, en-Davis_man, en-Emma_woman, en-Frank_man, en-Grace_woman, en-Mike_man
de-Spk0_man, de-Spk1_woman
fr-Spk0_man, fr-Spk1_woman
in-Samuel_man
it-Spk0_woman, it-Spk1_man
jp-Spk0_man, jp-Spk1_woman
kr-Spk0_woman, kr-Spk1_man
nl-Spk0_man, nl-Spk1_woman
pl-Spk0_man, pl-Spk1_woman
pt-Spk0_woman, pt-Spk1_man
sp-Spk0_woman, sp-Spk1_man
```

`random` selects a preset at synthesis time. If you pass `language` without `voice`, Valar chooses a deterministic default for that language. Hindi remains exploratory and is not part of the release-facing default quality set.

## Voxtral: Explicit Non-Commercial Opt-In

```bash
export VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1
swift run --package-path apps/ValarCLI valartts models install mlx-community/Voxtral-4B-TTS-2603-mlx-4bit --allow-download
swift run --package-path apps/ValarCLI valartts speak \
  --model mlx-community/Voxtral-4B-TTS-2603-mlx-4bit \
  --voice emma \
  --text "Hello from Voxtral." \
  --output "${TMPDIR:-/tmp}/voxtral.wav"
```

Voxtral is preset-only in this local public surface. Do not route reference-audio cloning, saved voice design, or reusable custom voices to Voxtral. Use it only when you intentionally opt into non-commercial software and want a multilingual preset voice.

## Daemon Checks

Start the daemon in a separate terminal:

```bash
make quickstart
make build-daemon
swift run --package-path apps/ValarDaemon valarttsd
```

If the daemon is already running locally, these endpoints should answer on localhost:

```bash
curl http://127.0.0.1:8787/v1/health
curl http://127.0.0.1:8787/v1/capabilities
curl http://127.0.0.1:8787/v1/models
curl http://127.0.0.1:8787/v1/runtime
```

For a broader public smoke test after the quickstart is working:

```bash
make validate-live-blessed
```

That covers `Soprano`, `Qwen`, and `VibeVoice`. If you also want `Voxtral`, opt in first:

```bash
export VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1
make validate-live-blessed
```
