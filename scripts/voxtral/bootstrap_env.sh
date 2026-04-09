#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLCHAIN_SCRIPT="$ROOT_DIR/scripts/voxtral/ensure_toolchain.sh"

if [[ ! -x "$TOOLCHAIN_SCRIPT" ]]; then
  echo "Missing Voxtral toolchain helper at $TOOLCHAIN_SCRIPT" >&2
  exit 1
fi

python_bin="$("$TOOLCHAIN_SCRIPT" --print-python)"
torch_version="$("$python_bin" -c 'import torch; print(torch.__version__)')"
printf '[valartts] managed Voxtral tooling ready via %s (torch %s)\n' "$python_bin" "$torch_version" >&2
