# Lineage / Upstream References

Valar grew out of an earlier private development tree and is now published here as a cleaned public source repo.

## Repo Lineage

- Earlier internal development tree: `ValarTTS`
- Public source repo: `Valar`
- Legacy parity harness: `mlx_audio`
- This repo is the public source base for current app, CLI, daemon, and bridge work

## Swift / MLX Lineage

- `mlx-audio-swift-valar` is the Valar fork of `mlx-audio-swift`
- Upstream Swift MLX reference: [Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)
- Legacy Python reference: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- The local Swift runtime intentionally moved beyond the legacy Python parity harness; public docs should teach the current Swift CLI + daemon workflow

## What This Means For Public Users

- The public repo is the maintained source base for the app, CLI, daemon, and integrations.
- The legacy Python parity harness stays as a historical reference point only.
- The public repo should teach the current CLI and daemon workflow, not the old private operator workflow.

## What To Cite In The Public Docs

- For current model behavior, cite the public working-models page and the live `valartts models info` output.
- For backend provenance, cite `mlx-audio-swift-valar` and its upstream reference.
- For historical context, mention `mlx_audio` as the original parity harness, but do not teach it as the beginner path.
