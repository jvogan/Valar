# Examples

Small public examples for Valar outside of the macOS app.

## Working Examples

| Example | Description | Prerequisites |
|---|---|---|
| [`cli-quickstart.sh`](./cli-quickstart.sh) | Generate speech from text via the CLI | `make quickstart` |
| [`daemon-api.sh`](./daemon-api.sh) | Generate speech via the local HTTP daemon | `make quickstart` + `make start-daemon` |
| [`headless-synthesis.swift`](./headless-synthesis.swift) | Programmatic synthesis sketch for the public Swift runtime | `make quickstart` |

## Quick Start

```bash
# CLI path (simplest)
bash examples/cli-quickstart.sh "Hello from Valar."

# Daemon path (HTTP API)
make start-daemon  # in another terminal
bash examples/daemon-api.sh "Hello from the daemon."

# Custom model
VALAR_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 bash examples/cli-quickstart.sh "Hello with Qwen."
```

## Requirements

- macOS 14+ on Apple Silicon
- `make quickstart` completed successfully
- Bun 1.2+ (only if using the MCP bridge)

## Intended Use Cases

- **CLI synthesis** — generate audio files from text in one command
- **Daemon API** — use the local HTTP API for integrations
- **Headless synthesis** — sketch for driving the Swift runtime directly
- **Batch workflows** — build on these patterns for bulk audio generation

These examples are intentionally small and public-safe. They use the blessed public families: `Soprano`, `Qwen`, `VibeVoice`, and `Voxtral`.
