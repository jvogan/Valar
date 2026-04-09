# FAQ

## What works today?

Valar's public v1 working set is intentionally narrow:

- `Soprano` for the fastest first clip
- `Qwen` for the main TTS, ASR, and alignment lane
- `VibeVoice` for preset-only preview speech
- `Voxtral` for explicit non-commercial opt-in

See [working-models.md](./working-models.md) for the exact install IDs.

## Which model should I install first?

Install `Soprano` first if you want the fastest proof that your machine and toolchain are working. Install `Qwen Base` first if your real target is long-form narration or the main supported TTS lane.

## Does Valar run fully local?

Yes for inference. Speech generation, transcription, and alignment run locally on your Mac. Network access is only needed when you explicitly download models or other dependencies from upstream hosts.

## Do I need Bun?

No. Bun is only required if you want the MCP bridge in `bridge/`. The default newcomer path is native Swift only.

## How much disk and memory should I expect?

Rough local footprint for the public working set:

- `Soprano`: about `285 MB`
- `Qwen`: about `1.0 GB` to `4.2 GB`, depending on the lane
- `VibeVoice`: about `700 MB`
- `Voxtral`: about `2.4 GB`

Runtime memory usage varies by model family and request size. `Soprano` is the easiest first install when you want the smallest local footprint.

## Which models are commercial-safe?

Valar code is MIT, but model weights keep their upstream licenses.

- `Soprano` uses Apache 2.0
- `VibeVoice` uses MIT
- `Qwen` uses the Qwen License Agreement
- `Voxtral` is non-commercial only

Check the exact model card and license before commercial deployment. Valar does not provide legal advice.

## When should I use the app vs CLI vs MCP?

- Use the **CLI** for the fastest first success and most direct automation scripts
- Use the **daemon** when you want a local HTTP surface on `127.0.0.1:8787`
- Use the **MCP bridge** when you want a local agent or tool-calling workflow
- Use the **macOS app** after the CLI path works and you want a desktop workflow

## Why does the first run take a while?

The first run resolves Swift dependencies, builds the CLI, and makes sure MLX Metal shaders are available for local inference. After that, repeated runs are much faster.

## What if `make quickstart` says the Metal toolchain is missing?

Install the Metal toolchain and complete first-launch setup:

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

If you already have an `mlx.metallib` from another MLX setup, reuse it with `VALARTTS_METALLIB_FALLBACK_PATH`.
