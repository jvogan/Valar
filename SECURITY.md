# Security Policy

Valar runs locally on Apple Silicon. Audio, text, model execution, and project data stay on-device unless you explicitly download models from an upstream host or connect Valar to another tool.

## Default Security Posture

- synthesis, transcription, and alignment run locally
- the daemon binds to loopback by default
- there is no telemetry or analytics pipeline in the repo
- model downloads are user-initiated and fetched over HTTPS
- the public MCP bridge depends on the same local loopback daemon rather than a hosted inference service

## Reporting A Vulnerability

If you discover a security issue in Valar, report it privately rather than opening a public issue with exploit details.

Preferred path:

- use GitHub Security Advisories on the public `Valar` repo
- do not publish exploit details in public issues while the report is untriaged

Please include:

- what is vulnerable
- how it can be reproduced
- expected impact
- the commit or release you tested

## Repo Scope

This public repo focuses on:

- `apps/ValarCLI`
- `apps/ValarDaemon`
- `apps/ValarTTSMac`
- `Packages/*`
- the public MCP bridge under `bridge/`

Advanced integrations are included as source, but they are not the default onboarding path.
