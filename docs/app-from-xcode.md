# App from Xcode

The public repo includes the macOS app source, but the supported newcomer path is still CLI + daemon first.

## Recommended Order

1. Run `make quickstart`
2. Run `make first-clip`
3. Optionally start the daemon and verify `/v1/health`
4. Only then build the app from Xcode

That order keeps first-run problems narrow. If the CLI cannot synthesize locally, the app will not fix the underlying toolchain or model issue.

## What The App Is For

Use the app when you want:

- a native macOS interface for generation
- local document/project workflows
- model browsing and voice management in a GUI

Do not treat the app as the simplest way to prove the repo works. The CLI is the public smoke path.

## Xcode Notes

- Build the app from Xcode, not `swift run`
- If local inference fails, revisit:
  - [Prerequisites and Expectations](./prerequisites-and-expectations.md)
  - [Model Quickstart](./model-quickstart.md)
  - [Xcode Troubleshooting](./xcode-troubleshooting.md)
