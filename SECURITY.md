# Security Policy

## Privacy and Data Model

Valar runs locally on Apple Silicon. Audio, text, model execution, and project data stay on-device unless you explicitly download models from an upstream host or wire Valar into an external integration.

Core security posture:

- synthesis, transcription, and alignment run locally
- the daemon binds to loopback by default
- there is no telemetry or analytics pipeline in the public source tree
- model downloads are user-initiated and fetched over HTTPS
- the public repo should keep both the public audit and tracked-file secret scan green in CI

## Reporting a Vulnerability

If you discover a security issue in Valar, report it privately rather than opening a public issue with exploit details.

Preferred path:

- use GitHub Security Advisories on the public `Valar` repo
- do not publish exploit details in public issues while the report is untriaged

Release requirement for the public repo:

- do not publish `Valar` until GitHub Security Advisories are enabled for the repo
- treat that advisory flow as the supported disclosure channel in the first public release
- enable secret scanning and push protection before accepting public contributions

Please include:

- what is vulnerable
- how it can be reproduced
- expected impact
- the commit or release you tested

## Public Scope

This public repo focuses on:

- `apps/ValarCLI`
- `apps/ValarDaemon`
- `apps/ValarTTSMac`
- `Packages/*`
- the public MCP bridge under `bridge/`

Advanced integrations are included as source, but they are not the primary onboarding path.
