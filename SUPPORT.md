# Support

## Before Opening an Issue

Start with the public newcomer path:

```bash
make quickstart
make first-clip
```

Then gather the basics:

```bash
swift run --package-path apps/ValarCLI valartts doctor --json
swift run --package-path apps/ValarCLI valartts models list --json
```

If you are working on the MCP bridge too:

```bash
make bootstrap-bridge
make validate-public
```

## Where To Ask

- bug reports: use the GitHub bug template
- feature requests: use the GitHub feature template
- security reports: follow [SECURITY.md](./SECURITY.md) and do not post exploit details publicly

## What To Include

- macOS version and Apple Silicon model
- whether you used `make quickstart`
- the exact model ID you installed
- the command you ran
- the relevant CLI or daemon output

Keep reports focused on the public repo surface. Redact secrets and personal file paths before posting logs.
