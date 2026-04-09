# Contributing to Valar

Valar is a local speech stack for macOS and Apple Silicon. The public repo is source-first: the fastest reliable path is still CLI + daemon first, with the macOS app source included for Xcode users.

## Start Here

1. Read [README.md](./README.md).
2. Read [docs/working-models.md](./docs/working-models.md).
3. Read [docs/model-quickstart.md](./docs/model-quickstart.md).
4. Use [AGENTS.md](./AGENTS.md) if you are driving the repo through an agent.

## Build And Validation

For the default native path:

```bash
brew install jq ripgrep
make quickstart
make first-clip
make validate-native
```

If you are working on the MCP bridge too:

```bash
brew install jq ripgrep
make bootstrap-bridge
make validate-bridge
```

`validate-native` is the baseline public pre-push check. `validate-bridge` adds Bun-backed bridge validation on top of the native Swift checks.
`validate-live` adds an isolated first-run smoke test, and `validate-live-blessed` extends that to `Qwen` and `VibeVoice`.

For the public release gates:

```bash
make audit-public
```

Maintainers currently port accepted public changes back into the canonical source tree before the next export/sync. See [docs/release-maintainers.md](./docs/release-maintainers.md).
Use [SUPPORT.md](./SUPPORT.md) for routine help channels and [SECURITY.md](./SECURITY.md) for vulnerability reporting.

## Public Scope

The public beginner path is intentionally narrow:

- `Soprano`
- `Qwen`
- `VibeVoice`
- `Voxtral`

Do not expand newcomer docs or agent guidance to experimental families unless they are intentionally promoted into the public working-model set.

## Contributor Rules

- Keep PRs focused on one concern.
- Keep docs aligned with the actual CLI surface.
- Do not add private workstation paths, launchd helpers, SSD-only paths, or local operator playbooks to the public repo.
- Keep model-license language explicit: repo code is MIT, model weights keep their upstream licenses.
- Treat Voxtral as explicit non-commercial opt-in.
- Treat VibeVoice as preset-only compatibility preview, not the default narrator lane.
