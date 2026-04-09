#!/usr/bin/env bash
set -euo pipefail

with_bridge=0
run_live=0
run_live_blessed=0

usage() {
  cat <<EOF
Usage: $0 [--with-bridge] [--live] [--live-blessed]

--with-bridge  Typecheck the public MCP bridge (requires Bun).
--live         Install Soprano into an isolated VALARTTS_HOME and synthesize one smoke clip.
--live-blessed Install and smoke-test Soprano, Qwen, and VibeVoice.
               Voxtral is also tested when VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-bridge)
      with_bridge=1
      shift
      ;;
    --live)
      run_live=1
      shift
      ;;
    --live-blessed)
      run_live=1
      run_live_blessed=1
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

require_cmd() {
  local cmd="$1"
  local hint="${2:-}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command not found on PATH: $cmd" >&2
    if [[ -n "$hint" ]]; then
      echo "$hint" >&2
    fi
    exit 1
  fi
}

require_cmd swift
require_cmd jq "Install jq before running public validation."
require_cmd rg "Install ripgrep before running public validation."

if [[ "$with_bridge" == "1" ]]; then
  require_cmd bun "Install Bun 1.2 or newer before running public bridge validation."
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/valar-public-validate.XXXXXX")"
cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

export VALARTTS_HOME="$tmp_root/valartts-home"
mkdir -p "$VALARTTS_HOME"

run_speak_smoke() {
  local label="$1"
  local model_id="$2"
  local text="$3"
  local output_path="$4"
  shift 4

  echo "Running live public smoke for $label"
  "$cli_bin" models install "$model_id" --allow-download
  "$cli_bin" speak \
    --model "$model_id" \
    --text "$text" \
    --output "$output_path" \
    "$@"

  [[ -s "$output_path" ]] || {
    echo "Error: $label live smoke did not create output audio." >&2
    exit 1
  }
}

[[ -f "tools/public_repo_audit.sh" ]] || {
  echo "Error: expected public audit script at tools/public_repo_audit.sh" >&2
  exit 1
}
[[ -f "tools/public_repo_secret_scan.sh" ]] || {
  echo "Error: expected public secret scan script at tools/public_repo_secret_scan.sh" >&2
  exit 1
}
[[ -f "tools/public_repo_history_scan.sh" ]] || {
  echo "Error: expected public history scan script at tools/public_repo_history_scan.sh" >&2
  exit 1
}
bash tools/public_repo_audit.sh --root "$repo_root"
bash tools/public_repo_secret_scan.sh --root "$repo_root"
bash tools/public_repo_history_scan.sh --root "$repo_root"

build_targets=(
  "apps/ValarCLI"
  "apps/ValarDaemon"
)

for dir in "${build_targets[@]}"; do
  [[ -d "$dir" ]] || continue
  echo "Building $dir"
  swift build --package-path "$dir"
done

echo "Building MLX metallib for SPM outputs"
bash scripts/build_metallib.sh

if [[ "$with_bridge" == "1" && -f "bridge/package.json" ]]; then
  echo "Typechecking MCP bridge"
  (
    cd bridge
    bun install
    bun run typecheck
  )
fi

cli_bin="$repo_root/apps/ValarCLI/.build/arm64-apple-macosx/debug/valartts"
[[ -x "$cli_bin" ]] || {
  echo "Error: expected CLI binary at $cli_bin" >&2
  exit 1
}

echo "Checking public model surface"
models_json="$tmp_root/models.json"
soprano_json="$tmp_root/soprano.json"
qwen_json="$tmp_root/qwen.json"
vibevoice_json="$tmp_root/vibevoice.json"
voxtral_json="$tmp_root/voxtral.json"
doctor_json="$tmp_root/doctor.json"
doctor_report_json="$tmp_root/doctor-report.json"
doctor_err="$tmp_root/doctor.err"
actual_ids="$tmp_root/actual-model-ids.txt"
expected_ids="$tmp_root/expected-model-ids.txt"

"$cli_bin" models list --json >"$models_json"
"$cli_bin" models info mlx-community/Soprano-1.1-80M-bf16 --json >"$soprano_json"
"$cli_bin" models info mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --json >"$qwen_json"
"$cli_bin" models info mlx-community/VibeVoice-Realtime-0.5B-4bit --json >"$vibevoice_json"
VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1 "$cli_bin" models info mlx-community/Voxtral-4B-TTS-2603-mlx-4bit --json >"$voxtral_json"

set +e
"$cli_bin" doctor --json >"$doctor_json" 2>"$doctor_err"
doctor_exit=$?
set -e

jq -r '.data.models[].id' "$models_json" | LC_ALL=C sort >"$actual_ids"
cat >"$expected_ids" <<'EOF'
mlx-community/Qwen3-ASR-0.6B-8bit
mlx-community/Qwen3-ForcedAligner-0.6B-8bit
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
mlx-community/Soprano-1.1-80M-bf16
mlx-community/VibeVoice-Realtime-0.5B-4bit
EOF

if [[ "${VALARTTS_ENABLE_NONCOMMERCIAL_MODELS:-0}" == "1" ]]; then
  echo "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit" >>"$expected_ids"
fi

diff -u "$expected_ids" "$actual_ids"

jq -e '.ok == true and .data.model.id == "mlx-community/Soprano-1.1-80M-bf16" and .data.model.supportTier == "supported" and .data.model.distributionTier == "bundledFirstRun" and .data.model.releaseEligible == true' "$soprano_json" >/dev/null
jq -e '.ok == true and .data.model.id == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" and .data.model.supportTier == "supported" and .data.model.releaseEligible == true' "$qwen_json" >/dev/null
jq -e '.ok == true and .data.model.id == "mlx-community/VibeVoice-Realtime-0.5B-4bit" and .data.model.supportTier == "preview" and .data.model.distributionTier == "compatibilityPreview" and .data.model.qualityTierByLanguage.en == "supported" and .data.model.qualityTierByLanguage.hi == "experimental"' "$vibevoice_json" >/dev/null
jq -e '.ok == true and .data.model.id == "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit" and .data.model.supportTier == "preview" and .data.model.releaseEligible == false and .data.model.licenseName == "CC BY-NC 4.0"' "$voxtral_json" >/dev/null
jq -s '
  map(select(.command == "valartts doctor" and .data != null and .ok == true))
  | .[0]
' "$doctor_json" >"$doctor_report_json"
jq -e '.ok == true and .data.localInferenceAssetsReady == true' "$doctor_report_json" >/dev/null

rg -q "mlx-community/Soprano-1.1-80M-bf16" README.md docs/working-models.md docs/model-quickstart.md
rg -q "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" docs/working-models.md docs/model-quickstart.md
rg -q "mlx-community/VibeVoice-Realtime-0.5B-4bit" docs/working-models.md docs/model-quickstart.md
rg -q "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit" docs/working-models.md docs/model-quickstart.md
rg -q "make quickstart" README.md docs/model-quickstart.md docs/prerequisites-and-expectations.md AGENTS.md
rg -q "make first-clip" README.md docs/model-quickstart.md AGENTS.md
[[ -f "docs/prerequisites-and-expectations.md" ]]
[[ -f "docs/app-from-xcode.md" ]]
[[ -f "docs/release-maintainers.md" ]]
[[ ! -e "examples/bible-audiobook" ]] || {
  echo "Error: experimental example subtree examples/bible-audiobook should not be in the public repo." >&2
  exit 1
}
! rg -n "ValarTTS" README.md docs/README.md docs/working-models.md docs/model-quickstart.md docs/prerequisites-and-expectations.md docs/app-from-xcode.md docs/release-maintainers.md docs/integrations.md examples/README.md examples/headless-synthesis.swift >/dev/null

if [[ "$run_live" == "1" ]]; then
  run_speak_smoke \
    "Soprano" \
    "mlx-community/Soprano-1.1-80M-bf16" \
    "Hello from Valar public validation." \
    "$tmp_root/soprano-smoke.wav"

  if [[ "$run_live_blessed" == "1" ]]; then
    run_speak_smoke \
      "Qwen" \
      "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16" \
      "Hello from the Valar public Qwen smoke test." \
      "$tmp_root/qwen-smoke.wav"

    run_speak_smoke \
      "VibeVoice" \
      "mlx-community/VibeVoice-Realtime-0.5B-4bit" \
      "Hello from the Valar public VibeVoice smoke test." \
      "$tmp_root/vibevoice-smoke.wav" \
      --voice random \
      --language en

    if [[ "${VALARTTS_ENABLE_NONCOMMERCIAL_MODELS:-0}" == "1" ]]; then
      run_speak_smoke \
        "Voxtral" \
        "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit" \
        "Hello from the Valar public Voxtral smoke test." \
        "$tmp_root/voxtral-smoke.wav" \
        --voice emma
    else
      echo "Skipping Voxtral live smoke because VALARTTS_ENABLE_NONCOMMERCIAL_MODELS is not enabled."
    fi
  fi
fi

if [[ "$doctor_exit" -ne 0 ]]; then
  echo "Note: doctor exited non-zero in the clean validation home, which is expected until a user installs models." >&2
  cat "$doctor_err" >&2 || true
fi

echo "Public validation passed."
