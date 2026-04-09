# Xcode Build Troubleshooting for Valar Swift Builds

## Metal Toolchain Error

When building the CLI or daemon from source, you may see:

```text
error: cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain
```

This means Xcode is installed, but the Metal toolchain component needed for
`mlx.metallib` is still missing.

## Fix

```bash
xcodebuild -downloadComponent MetalToolchain
xcodebuild -runFirstLaunch
```

If you use a beta Xcode, point `DEVELOPER_DIR` at that install explicitly:

```bash
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcodebuild -downloadComponent MetalToolchain
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcodebuild -runFirstLaunch
```

## Verify

```bash
xcrun --find metal
xcrun --find metallib
```

Both commands should print a path.

## Retry Valar Build

```bash
swift build --package-path apps/ValarCLI
bash scripts/build_metallib.sh
swift run --package-path apps/ValarCLI valartts doctor
```

If you are only trying to get a first clip working, return to
[docs/model-quickstart.md](./model-quickstart.md) after the Metal toolchain is
installed.
