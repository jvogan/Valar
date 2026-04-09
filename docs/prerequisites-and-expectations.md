# Prerequisites and Expectations

This page answers the practical public-repo question first: what do you need on your Mac, and what should you install first?

## Supported Machine Profile

Valar is a local speech stack for macOS on Apple Silicon. It requires **macOS 14 (Sonoma) or later** and is validated on macOS 15 (Sequoia).

Public newcomer assumptions:

- macOS 14+ on Apple Silicon (M1 or later)
- Xcode command-line tools installed
- the full Metal toolchain available so `mlx.metallib` can be built for local inference
- `jq` and `ripgrep` available for validation helpers
- Bun is optional unless you want the MCP bridge

If `bash scripts/build_metallib.sh` complains that `metal` or `metallib` is missing, fix Xcode first:

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

## Fastest Working Path

Valar is intentionally source-first and CLI-first:

```bash
make quickstart
make first-clip
```

That path:

1. resolves Swift dependencies
2. builds the public CLI
3. builds `mlx.metallib`
4. runs `valartts doctor`
5. installs `Soprano`
6. writes a first WAV clip under your macOS temp directory, usually at `$TMPDIR/valar-first-clip.wav`

If you already use another local Valar checkout on the same Mac, isolate this repo while testing:

```bash
export VALARTTS_HOME="$PWD/.valartts-public-home"
```

That repo-local state directory is gitignored in this public repo.

## What Works Today

The public v1 working set is intentionally narrow.

| Family | Public posture | Rough local footprint | Best first use |
| --- | --- | --- | --- |
| `Soprano` | Supported | About `285 MB` | Fastest first success |
| `Qwen` | Supported | About `1.0 GB` to `4.2 GB`, depending on the lane | Main TTS, ASR, and alignment lane |
| `VibeVoice` | Compatibility preview | About `700 MB` | Preset-voice realtime TTS, English-first |
| `Voxtral` | Preview, explicit non-commercial opt-in | About `2.4 GB` | Optional preset-voice multilingual lane |

Footprint values are rough planning numbers and can drift as upstream packs change.

## What To Install First

- Install `Soprano` first if you want the fastest proof that your machine is working.
- Install `Qwen Base` first if your real target is long-form narration or the main public TTS lane.
- Install `Qwen ASR` or `Qwen ForcedAligner` only when you need transcription or timestamps.
- Install `VibeVoice` only when you specifically want preset voices and accept preview-quality multilingual behavior.
- Install `Voxtral` only when you intentionally opt into non-commercial models.

## Bridge And App Expectations

- The MCP bridge is included, but it is an advanced path and requires Bun.
- The macOS app source is included, but it is not the first-run path.
- Get the CLI working first, then move to the app from Xcode if needed: [App from Xcode](./app-from-xcode.md)

## License Boundary

- Repo code: MIT
- Model weights: upstream model-specific licenses
- Non-commercial models remain opt-in and are never the default newcomer recommendation
