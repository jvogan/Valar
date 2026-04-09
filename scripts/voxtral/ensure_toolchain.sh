#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VOXTRAL_DIR="$ROOT_DIR/scripts/voxtral"
VENV_DIR="${VALARTTS_VOXTRAL_VENV_DIR:-$VOXTRAL_DIR/.venv}"
REQUIREMENTS_FILE="$VOXTRAL_DIR/requirements.txt"
CONVERTER_SCRIPT="$VOXTRAL_DIR/convert_voice_embeddings.py"
BOOTSTRAP_PYTHON="${VALARTTS_VOXTRAL_BOOTSTRAP_PYTHON:-python3}"
STAMP_FILE="$VENV_DIR/.requirements.sha256"

usage() {
  cat <<EOF
Usage: scripts/voxtral/ensure_toolchain.sh [--print-python | --check | --run-converter INPUT --output DIR]

Options:
  --print-python          Ensure the managed Voxtral venv exists, then print its Python path.
  --check                 Exit 0 only if the managed Voxtral venv and torch are already available.
  --run-converter INPUT   Ensure the managed Voxtral venv exists, then run convert_voice_embeddings.py.
  --output DIR            Output directory used with --run-converter.

Environment overrides:
  VALARTTS_VOXTRAL_VENV_DIR
  VALARTTS_VOXTRAL_BOOTSTRAP_PYTHON
EOF
}

require_bootstrap_python() {
  if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
    echo "Missing bootstrap Python interpreter '$BOOTSTRAP_PYTHON'. Install Python 3, then rerun 'bash scripts/voxtral/bootstrap_env.sh' or 'make bootstrap-native'." >&2
    exit 1
  fi
}

managed_python() {
  printf '%s\n' "$VENV_DIR/bin/python"
}

toolchain_ready() {
  local python_bin
  python_bin="$(managed_python)"
  [[ -x "$python_bin" ]] || return 1
  "$python_bin" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
}

ensure_toolchain() {
  require_bootstrap_python

  local python_bin
  python_bin="$(managed_python)"

  if [[ ! -x "$python_bin" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
  fi

  local desired_hash existing_hash="" needs_install=0
  desired_hash="$(shasum -a 256 "$REQUIREMENTS_FILE" | awk '{print $1}')"
  if [[ -f "$STAMP_FILE" ]]; then
    existing_hash="$(<"$STAMP_FILE")"
  fi

  if ! toolchain_ready; then
    needs_install=1
  fi
  if [[ "$desired_hash" != "$existing_hash" ]]; then
    needs_install=1
  fi

  if [[ "$needs_install" -eq 0 ]]; then
    return 0
  fi

  "$python_bin" -m pip install --upgrade pip >/dev/null
  "$python_bin" -m pip install -r "$REQUIREMENTS_FILE" >/dev/null
  printf '%s\n' "$desired_hash" > "$STAMP_FILE"

  if ! toolchain_ready; then
    echo "Managed Voxtral toolchain setup did not produce a working torch install in '$VENV_DIR'." >&2
    exit 1
  fi
}

run_converter() {
  local input_path="$1"
  local output_dir="$2"
  ensure_toolchain
  exec "$(managed_python)" "$CONVERTER_SCRIPT" "$input_path" --output "$output_dir"
}

main() {
  if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
  fi

  case "$1" in
    --print-python)
      ensure_toolchain
      managed_python
      ;;
    --check)
      toolchain_ready
      ;;
    --run-converter)
      if [[ $# -ne 4 || "$3" != "--output" ]]; then
        usage >&2
        exit 2
      fi
      run_converter "$2" "$4"
      ;;
    --help|-h)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
