#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Bootstrapping native dependencies"
bash ./tools/bootstrap.sh native

echo "Building Valar CLI"
swift build --package-path apps/ValarCLI

echo "Building MLX metallib"
bash ./scripts/build_metallib.sh

cli_bin="$repo_root/apps/ValarCLI/.build/arm64-apple-macosx/debug/valartts"
if [[ ! -x "$cli_bin" ]]; then
  echo "Error: expected CLI binary at $cli_bin" >&2
  exit 1
fi

doctor_json="$(mktemp)"
doctor_err="$(mktemp)"
cleanup() {
  rm -f "$doctor_json" "$doctor_err"
}
trap cleanup EXIT

echo "Checking local readiness"
set +e
"$cli_bin" doctor --json >"$doctor_json" 2>"$doctor_err"
doctor_exit=$?
set -e

if command -v jq >/dev/null 2>&1; then
  jq -sr '
    map(select(type == "object" and .command == "valartts doctor" and .data != null and .ok == true))
    | .[0]
    | if . == null then
        "Quickstart note: doctor did not return a machine-readable readiness report."
      elif .data.localInferenceAssetsReady == true then
        "Quickstart ready: local inference assets are available."
      else
        "Quickstart note: doctor reported missing local inference prerequisites."
      end
  ' "$doctor_json"
fi

if [[ "$doctor_exit" -ne 0 ]]; then
  echo "Doctor returned a non-zero exit code. This is common on a fresh checkout before any models are installed." >&2
  cat "$doctor_err" >&2 || true
fi

echo "Next step: make first-clip"
