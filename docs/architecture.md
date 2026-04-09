# Architecture

Valar ships three public frontend surfaces in this repo:

- `apps/ValarCLI`: the `valartts` CLI
- `apps/ValarDaemon`: the local `valarttsd` HTTP daemon
- `apps/ValarTTSMac`: the macOS app source

For new users, the reliable path is still CLI first, daemon second, app third.

## Shared Runtime

The shared runtime lives in `Packages/ValarCore`:

- `ValarRuntime` is the common composition root for CLI and daemon
- shared DTOs live in `Packages/ValarCore/Sources/ValarCore/DTOs.swift`
- model and catalog contracts live in `Packages/ValarModelKit`
- MLX-backed loading and inference live in `Packages/ValarMLX`
- audio decode, playback, export, and resampling live in `Packages/ValarAudio`
- persistence and project storage live in `Packages/ValarPersistence`

## Public Entry Points

### CLI

`apps/ValarCLI` is the fastest path to a first working clip:

- `valartts doctor`
- `valartts models list`
- `valartts models install`
- `valartts speak`
- `valartts transcribe`
- `valartts align`

### Daemon

`apps/ValarDaemon` exposes a local HTTP API on `127.0.0.1:8787` by default:

- `/v1/health`
- `/v1/capabilities`
- `/v1/models`
- `/v1/runtime`
- `/v1/audio/speech`
- `/v1/audio/transcriptions`
- `/v1/alignments`

### App

`apps/ValarTTSMac` contains the macOS app source. It shares the same runtime packages, but it is not required for first success in this public repo.

## Public Model Surface

The blessed public families are intentionally narrow:

- `Soprano`
- `Qwen`
- `VibeVoice`
- `Voxtral`

Use the public catalog and docs for the compatibility contract:

- [Working Models](./working-models.md)
- [Model Quickstart](./model-quickstart.md)

## Integrations

The public bridge in `bridge/` is a generic MCP surface for local agents and automation. Channel-specific adapters are intentionally out of scope for the public beginner path.
