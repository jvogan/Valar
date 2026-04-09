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

## Issue And PR Hygiene

- keep blank issues disabled
- keep the security contact link pointed at the repo advisory form
- keep `SUPPORT.md`, `SECURITY.md`, and issue templates aligned
- make sure the PR template still references `make audit-public`

## Release Boundary

The public repo should keep fresh public history only. Do not mirror the canonical private git history into the public repository.
