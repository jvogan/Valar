# Privacy

Valar is a local speech stack for macOS and Apple Silicon. This document describes the public repo's default data model and network posture.

## Summary

- no telemetry or analytics are sent anywhere
- synthesis, transcription, and alignment run locally
- model downloads happen only when the user explicitly installs a model
- the daemon listens on loopback by default
- the public repo does not install background services for you by default

## Local Data

Valar stores local state on your Mac for things like:

- downloaded model metadata and installed model packs
- saved voices and voice metadata
- project or document state
- generated outputs that you choose to write to disk

Saved voice material and other local state remain on your machine unless you choose to export or share those files yourself.

If you use the public MCP bridge in `bridge/`, you may choose to read or write local media under `~/Library/Application Support/Valar/bridge-storage`. The public bridge does not persist channel identifiers, sender metadata, or transcript/reply sidecars by default. Override this location with `VALARTTS_BRIDGE_STORAGE_ROOT` if needed.

## Network Access

The normal public workflow uses the network only for model downloads from upstream hosts such as Hugging Face. Valar does not require a cloud inference backend.

The daemon defaults to `127.0.0.1:8787` and is not network-accessible unless you explicitly opt into a non-loopback bind.

## Model Licenses

The repo code is MIT, but model weights are governed by their upstream licenses.

Important public examples:

- `Soprano`: model-specific upstream license
- `Qwen`: model-specific upstream license
- `VibeVoice`: MIT
- `Voxtral`: CC BY-NC 4.0 and non-commercial only

Always check the specific model card before deployment or redistribution.
