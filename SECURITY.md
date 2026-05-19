# Security Policy

Valar runs locally on Apple Silicon. Audio, text, model execution, and project data stay on-device unless you explicitly download models from an upstream host or connect Valar to another tool.

## Default Security Posture

- synthesis, transcription, and alignment run locally
- the daemon binds to loopback by default
- there is no telemetry or analytics pipeline in the repo
- model downloads are user-initiated and fetched over HTTPS
- the public MCP bridge depends on the same local loopback daemon rather than a hosted inference service
- the public validation gate runs a public-content audit, secret scan, and git-history scan in CI

Do not expose `valarttsd` outside loopback without adding your own authentication, authorization, TLS, logging, and rate controls. The public daemon is intended as a local developer and automation surface, not a hardened internet service.

## Reporting A Vulnerability

If you discover a security issue in Valar, report it privately rather than opening a public issue with exploit details.

Preferred path:

- use GitHub Security Advisories on the public `Valar` repo
- if advisories are unavailable, open a minimal public issue that says you need private security contact without exploit details
- do not publish exploit details in public issues while the report is untriaged

Please include:

- what is vulnerable
- how it can be reproduced
- expected impact
- the commit or release you tested

Maintainers should acknowledge new reports within 5 business days and keep the reporter updated when a fix, mitigation, or non-issue determination is ready.

## Repo Scope

This public repo focuses on:

- `apps/ValarCLI`
- `apps/ValarDaemon`
- `apps/ValarTTSMac`
- `Packages/*`
- the public MCP bridge under `bridge/`

Advanced integrations are included as source, but they are not the default onboarding path.
