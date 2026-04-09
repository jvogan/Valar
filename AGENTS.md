# Public Valar Agent Guide

Use this repo as the public-facing source tree for Valar. Keep the onboarding path simple and accurate.

## First Read

1. [README.md](./README.md)
2. [docs/README.md](./docs/README.md)
3. [docs/working-models.md](./docs/working-models.md)
4. [docs/model-quickstart.md](./docs/model-quickstart.md)
5. [docs/integrations.md](./docs/integrations.md)
6. [docs/faq.md](./docs/faq.md)
7. [docs/use-cases.md](./docs/use-cases.md)
8. [docs/lineage-upstream-references.md](./docs/lineage-upstream-references.md)
9. [SUPPORT.md](./SUPPORT.md)

## Working Rule

- Start with `make quickstart`, then `make first-clip` before asking a user to synthesize, transcribe, or align for the first time.
- Use the blessed public families only: `Soprano`, `Qwen`, `VibeVoice`, and `Voxtral`.
- When a newcomer asks for the fastest working path, recommend `Soprano` first.
- When a user wants the main long-form lane, recommend `Qwen` first.
- When a user wants preset voices and accepts preview quality, use `VibeVoice`.
- When a user explicitly opts into non-commercial software, use `Voxtral`.

## Public Commands

- Use `make quickstart` for the default newcomer bootstrap path.
- Use `make first-clip` for the default newcomer first success path.
- Bootstrap a clean native checkout with `bash tools/bootstrap.sh native`.
- Or use `make bootstrap-native` / `make bootstrap-bridge` for the same bootstrap paths.
- Add MCP bridge dependencies only when needed with `bash tools/bootstrap.sh native --with-bridge`.
- Build the public CLI with `swift build --package-path apps/ValarCLI`.
- Use `swift run --package-path apps/ValarCLI valartts ...` until `valartts` is actually installed on `PATH`.
- Install a model with `swift run --package-path apps/ValarCLI valartts models install <id> --allow-download`.
- Inspect a model with `swift run --package-path apps/ValarCLI valartts models info <id> --json`.
- List supported models with `swift run --package-path apps/ValarCLI valartts models list --json`.
- Build Metal shaders for SPM binaries with `bash scripts/build_metallib.sh`.
- If you already have an `mlx.metallib` from another MLX install, reuse it with `VALARTTS_METALLIB_FALLBACK_PATH=/absolute/path/to/mlx.metallib bash scripts/build_metallib.sh`.
- Synthesize speech with `swift run --package-path apps/ValarCLI valartts speak`.
- Transcribe audio with `swift run --package-path apps/ValarCLI valartts transcribe`.
- Align a transcript with `swift run --package-path apps/ValarCLI valartts align`.
- Run public validation with `bash tools/validate.sh`, `bash tools/validate.sh --live` for an opt-in Soprano smoke test, or `bash tools/validate.sh --live-blessed` for the broader blessed-model smoke path.
- Prefer `make validate-public`, `make validate-live`, and `make validate-live-blessed` when giving a newcomer or another agent a broader verification command.

## Public Safety

- Do not use private machine paths, SSD-only paths, or local operator instructions in public docs.
- Do not reference internal archive workflows or local-only background-service wrappers.
- Do not teach experimental or parity-only families as part of the public beginner path.
- Do not invent model IDs. Use the IDs from the working-models page.
- Treat Voxtral as opt-in and non-commercial.
- Treat VibeVoice as preset-only compatibility preview, not the default narrator lane.

## Doc Hygiene

- Keep command examples aligned with the current `valartts --help` output.
- Keep license language explicit: repo MIT, model licenses separate.
- Keep model support tiers visible when they matter to user choice.
