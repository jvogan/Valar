# Privacy

Valar is a local speech stack for macOS and Apple Silicon. This document describes the default data model and network posture of the public source release.

## Summary

- no telemetry or analytics are sent anywhere
- synthesis, transcription, and alignment run locally
- model downloads happen only when the user explicitly installs a model
- the daemon listens on loopback by default
- Valar does not install background services for you by default

## Local Data

Valar stores local state on your Mac for things like:

- downloaded model metadata and installed model packs
- saved voices and voice metadata
- project or document state
- generated outputs that you choose to write to disk

Saved voice material and other local state remain on your machine unless you choose to export or share those files yourself.

If you use the MCP bridge in `bridge/`, you may choose to read or write local media under `~/Library/Application Support/Valar/bridge-storage`. The public bridge does not persist channel identifiers, sender metadata, or transcript/reply sidecars by default. Override this location with `VALARTTS_BRIDGE_STORAGE_ROOT` if needed.

## Network Access

The normal workflow uses the network only for model downloads from upstream hosts such as Hugging Face. Valar does not require a cloud inference backend.

The public daemon binds to `127.0.0.1:8787` only. It is loopback-only in this repo and is not intended to be exposed as a network service.

## Model Licenses

The repo code is MIT, but model weights are governed by their upstream licenses.

Important public examples:

- `Soprano`: Apache 2.0
- `Qwen`: Qwen License Agreement
- `VibeVoice`: MIT
- `Voxtral`: CC BY-NC 4.0 (non-commercial only)

Always check the specific model card before deployment or redistribution.
