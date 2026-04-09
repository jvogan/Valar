# Examples

This directory contains small public examples for Valar outside of the macOS app.

## What belongs here

| Example | Description |
|---|---|
| `headless-synthesis.swift` | Programmatic synthesis sketch for the public Swift runtime |
| *(more coming)* | Small, public-safe examples aligned to the blessed model families |

## Intended use cases

- **Headless synthesis** — sketch how to drive the runtime without opening the app.
- **Swift package integration** — show how Valar packages fit together in code.
- **Batch workflows** — document lightweight public patterns that build on the CLI or runtime.

## Requirements

All Swift examples assume:

- macOS 14+
- Apple Silicon for MLX inference
- Swift 6 / Xcode 15+
- this repo checked out locally so the example can import local packages

## Running an example

```bash
swift examples/headless-synthesis.swift
```

`headless-synthesis.swift` is a guided sketch today, not a standalone package target. Running it prints the expected public runtime flow and the next commands to use.

These examples are intentionally small and public-safe. They should stay aligned to the blessed public families: `Soprano`, `Qwen`, `VibeVoice`, and `Voxtral`.
