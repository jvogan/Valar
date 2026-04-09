#!/usr/bin/env bash
set -euo pipefail

mode="native"
with_bridge=0

usage() {
  cat <<EOF
Usage: $0 [native] [--with-bridge]

native         Resolve Swift dependencies for the public CLI, daemon, and packages.
--with-bridge  Also install bridge dependencies with Bun.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    native)
      mode="native"
      shift
      ;;
    --with-bridge)
      with_bridge=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$mode" != "native" ]]; then
  usage >&2
  exit 1
fi

for dir in \
  "Packages/ValarCore" \
  "Packages/ValarAudio" \
  "Packages/ValarPersistence" \
  "Packages/ValarModelKit" \
  "Packages/ValarMLX" \
  "apps/ValarCLI" \
  "apps/ValarDaemon"
do
  if [[ -d "$dir" ]]; then
    echo "Resolving Swift package in $dir"
    swift package resolve --package-path "$dir"
  fi
done

if ! xcrun --find metal >/dev/null 2>&1 || ! xcrun --find metallib >/dev/null 2>&1; then
  echo "Note: local CLI/daemon inference needs the full Metal toolchain."
  echo "If bash scripts/build_metallib.sh fails later, run:"
  echo "  xcodebuild -downloadComponent MetalToolchain"
  echo "  xcodebuild -runFirstLaunch"
  echo "If you already have mlx.metallib from another MLX checkout or Python install, you can also reuse it with:"
  echo "  VALARTTS_METALLIB_FALLBACK_PATH=/absolute/path/to/mlx.metallib bash scripts/build_metallib.sh"
fi

if ! command -v jq >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1; then
  echo "Note: public validation also expects jq and ripgrep (rg)."
  echo "Install them before running bash tools/validate.sh, for example:"
  echo "  brew install jq ripgrep"
fi

if [[ "$with_bridge" == "1" && -f "bridge/package.json" ]]; then
  if ! command -v bun >/dev/null 2>&1; then
    echo "Error: --with-bridge requires Bun 1.2 or newer on PATH." >&2
    exit 1
  fi
  echo "Installing bridge dependencies"
  (
    cd bridge
    bun install
  )
fi
