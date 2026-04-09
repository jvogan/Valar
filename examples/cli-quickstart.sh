#!/usr/bin/env bash
# cli-quickstart.sh — Generate speech from text in one command
#
# Prerequisites:
#   make quickstart   (builds CLI + installs Metal shaders)
#
# Usage:
#   bash examples/cli-quickstart.sh
#   bash examples/cli-quickstart.sh "Your custom text here"
#   VALAR_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 bash examples/cli-quickstart.sh

set -euo pipefail

TEXT="${1:-Hello from Valar. This is local speech synthesis on Apple Silicon.}"
MODEL="${VALAR_MODEL:-mlx-community/Soprano-1.1-80M-bf16}"
OUTPUT="${VALAR_OUTPUT:-/tmp/valar-example.wav}"

CLI="swift run --package-path apps/ValarCLI valartts"

echo "Model:  $MODEL"
echo "Text:   $TEXT"
echo "Output: $OUTPUT"
echo ""

# Install the model if not already present
$CLI models install "$MODEL" --allow-download

# Synthesize speech
$CLI speak --model "$MODEL" --text "$TEXT" --output "$OUTPUT"

echo ""
echo "Done. Play it with:"
echo "  afplay $OUTPUT"
