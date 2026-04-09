# Upstream References And Credits

Valar builds on open-source MLX and speech tooling. This page points at the main upstream references behind the public repo.

## Runtime And Backend References

- `mlx-audio-swift-valar` is Valar's MLX Swift backend fork
- Upstream Swift MLX reference: [Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)
- Historical Python reference: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- Apple MLX project: [ml-explore/mlx](https://github.com/ml-explore/mlx)

## Model Family References

- `Soprano`: public small-model first-run lane
- `Qwen`: main TTS, ASR, and alignment lane
- `VibeVoice`: preset-only preview speech lane
- `Voxtral`: explicit non-commercial opt-in lane

Use [working-models.md](./working-models.md) for the exact public install IDs and current support posture.

## How To Cite Valar

- cite the public repo for the CLI, daemon, bridge, and app source
- cite the relevant upstream model card for model behavior, license, and constraints
- cite `mlx-audio-swift-valar` and its upstream references for backend provenance

## Historical Notes

The older Python parity harness remains a useful reference point, but the maintained public workflow is the current Swift CLI + daemon + MCP bridge stack in this repo.
