#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_path="${VALAR_FIRST_CLIP_OUTPUT:-${TMPDIR:-/tmp}/valar-first-clip.wav}"
clip_text="${VALAR_FIRST_CLIP_TEXT:-Hello from Valar.}"
model_id="mlx-community/Soprano-1.1-80M-bf16"

cli_bin="$repo_root/apps/ValarCLI/.build/arm64-apple-macosx/debug/valartts"
metallib_path="$repo_root/apps/ValarCLI/.build/arm64-apple-macosx/debug/mlx.metallib"
default_metallib_path="$repo_root/apps/ValarCLI/.build/arm64-apple-macosx/debug/default.metallib"

if [[ ! -x "$cli_bin" || ( ! -f "$metallib_path" && ! -f "$default_metallib_path" ) ]]; then
  bash ./tools/quickstart.sh
fi

if [[ ! -x "$cli_bin" ]]; then
  echo "Error: expected CLI binary at $cli_bin after quickstart" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_path")"
rm -f "$output_path"

echo "Installing $model_id"
"$cli_bin" models install "$model_id" --allow-download

echo "Synthesizing first clip"
"$cli_bin" speak \
  --model "$model_id" \
  --text "$clip_text" \
  --output "$output_path"

if [[ ! -s "$output_path" ]]; then
  echo "Error: first clip was not written to $output_path" >&2
  exit 1
fi

echo "Wrote first clip to: $output_path"
