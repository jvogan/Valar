# Valar

Valar is a local speech stack for macOS and Apple Silicon.

This repo is the public, MIT-licensed source tree for Valar. The fastest path for new users and agents is **CLI + daemon first**. The macOS app source stays in the repo, but it is not the primary onboarding path.

If you already use another local Valar checkout on the same Mac, isolate this public repo while testing by setting `VALARTTS_HOME` to a clean directory before running `valartts`.

## Start Here

1. Read [docs/prerequisites-and-expectations.md](./docs/prerequisites-and-expectations.md).
2. Run `make quickstart`.
3. Run `make first-clip`.
4. Read [docs/working-models.md](./docs/working-models.md) for the exact supported public model IDs.
5. Read [docs/model-quickstart.md](./docs/model-quickstart.md) for copy-paste install and first-run commands.
6. Read [docs/integrations.md](./docs/integrations.md) if you want MCP or advanced integrations.
7. Read [docs/lineage-upstream-references.md](./docs/lineage-upstream-references.md) for repo lineage and upstream references.

## Blessed Public Families

The newcomer path is intentionally narrow:

- `Soprano`: smallest and easiest first clip
- `Qwen`: main TTS, ASR, and forced-alignment lane
- `VibeVoice`: compatibility preview, preset-only, English-first
- `Voxtral`: optional non-commercial preset-voice lane

Other model families may be promoted later, but they are not part of the initial public beginner path.

## Quick Start

```bash
make quickstart
make first-clip
```

That writes a starter WAV to `/tmp/valar-first-clip.wav` by default. Override it with `VALAR_FIRST_CLIP_OUTPUT=/absolute/path.wav`.

If `make quickstart` reports a missing Metal toolchain, fix Xcode first:

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

If you already have `mlx.metallib` from another MLX checkout or Python install, you can reuse it instead of rebuilding:

```bash
VALARTTS_METALLIB_FALLBACK_PATH=/absolute/path/to/mlx.metallib bash scripts/build_metallib.sh
```

To start the local daemon:

```bash
make build-daemon
make build-metallib
swift run --package-path apps/ValarDaemon valarttsd
```

If `valartts` is already on your `PATH`, replace `swift run --package-path apps/ValarCLI valartts ...` with `valartts ...`.

For a broader public smoke test, use `make validate-live-blessed`. That adds Qwen and VibeVoice to the live check, and includes Voxtral too if you enable `VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1`.

For the full public pre-push path, including the Bun bridge typecheck:

```bash
make bootstrap-bridge
make validate-bridge
```

The macOS app source is included, but it is intentionally a secondary path. Build it from Xcode after the CLI path works:

- [docs/app-from-xcode.md](./docs/app-from-xcode.md)
- [docs/xcode-troubleshooting.md](./docs/xcode-troubleshooting.md)

## Support And Contributions

- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [SUPPORT.md](./SUPPORT.md)
- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- [SECURITY.md](./SECURITY.md)
- [PRIVACY.md](./PRIVACY.md)

## License Boundary

- Repo code: MIT
- Model weights: model-specific upstream licenses
- Non-commercial models are opt-in and clearly labeled

## Public Scope

- CLI + daemon are the default onboarding path
- macOS app source is included, but not required for first success
- integrations are available behind an advanced boundary and require Bun only if you opt into the MCP bridge
- no private machine paths, personal automation, or workstation-specific operator flows are part of this repo

## Maintainers

This public repo is maintained from a canonical source tree and exported with fresh public history. The maintainer flow is documented here:

- [docs/release-maintainers.md](./docs/release-maintainers.md)
- [docs/github-repo-settings.md](./docs/github-repo-settings.md)
