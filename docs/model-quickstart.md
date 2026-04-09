# Model Quickstart

This page is the copy-paste path for getting speech working quickly.

Use `swift run --package-path apps/ValarCLI valartts ...` if you have not installed `valartts` on your `PATH` yet. If you already have the executable installed, you can replace that prefix with `valartts ...`.

## Two-Command Newcomer Path

For the fastest first success in a clean clone:

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

## Common Readiness Check

```bash
make quickstart
swift run --package-path apps/ValarCLI valartts models list
```

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

## Soprano: Fastest First Run

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

## Qwen: Main Lane

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

Qwen `CustomVoice` remains the named-speaker lane, but the fastest public voice-creation path is `VoiceDesign` first and optional `stabilize` second.

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
