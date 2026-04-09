# GitHub Repo Settings

This checklist is for maintainers opening the public `Valar` repository on GitHub for the first time.

## Core Settings

- set the default branch to `main`
- require pull requests before merging to `main`
- require status checks to pass before merging
- disable force-pushes to `main`
- disable branch deletion for `main`

## Required Checks

Require these checks before merge:

- `validate-public`
- `audit-and-secret-scan`

If check names change, update this doc and [docs/release-maintainers.md](./release-maintainers.md) in the same export wave.

## Security Features

Enable these GitHub features before accepting outside contributions:

- GitHub Security Advisories
- secret scanning
- push protection
- Dependabot alerts
- Dependabot security updates

Optional once the repo is live and stable:

- CodeQL for Swift and TypeScript

## About Metadata

Use this public description:

`Local speech stack for Apple Silicon: TTS, ASR, forced alignment, voices, daemon, and MCP bridge.`

Recommended topics:

- `text-to-speech`
- `speech-to-text`
- `forced-alignment`
- `apple-silicon`
- `macos`
- `swift`
- `mlx`
- `mcp`
- `local-ai`
- `speech-synthesis`
- `asr`
- `voice-cloning`

Use `assets/media/social-preview.png` as the social preview image when the GitHub repo goes live.

## Issue And PR Hygiene

- keep blank issues disabled
- keep the security contact link pointed at the repo advisory form
- keep `SUPPORT.md`, `SECURITY.md`, and issue templates aligned
- make sure the PR template still references `make audit-and-secret-scan` and `make validate-public`
- leave funding links disabled until there is a real public funding handle

## Release Boundary

The public repo should keep fresh public history only. Do not mirror the canonical private git history into the public repository.
