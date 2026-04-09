#!/usr/bin/env bash
# daemon-api.sh — Generate speech via the local HTTP daemon
#
# Prerequisites:
#   make quickstart
#   make start-daemon   (in another terminal)
#
# Usage:
#   bash examples/daemon-api.sh
#   bash examples/daemon-api.sh "Your custom text here"

set -euo pipefail

DAEMON="${VALAR_DAEMON_URL:-http://127.0.0.1:8787}"
TEXT="${1:-Hello from the Valar daemon. This is the local HTTP API.}"
MODEL="${VALAR_MODEL:-mlx-community/Soprano-1.1-80M-bf16}"
OUTPUT="${VALAR_OUTPUT:-/tmp/valar-daemon-example.wav}"

echo "Daemon: $DAEMON"
echo "Model:  $MODEL"
echo "Text:   $TEXT"
echo "Output: $OUTPUT"
echo ""

# Check daemon health
if ! curl -sf "$DAEMON/v1/health" > /dev/null 2>&1; then
  echo "Error: Daemon not running at $DAEMON"
  echo "Start it with: make start-daemon"
  exit 1
fi

# Synthesize via the OpenAI-compatible endpoint
curl -fSs -X POST "$DAEMON/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"input\": \"$TEXT\"}" \
  -o "$OUTPUT"

echo "Done. Play it with:"
echo "  afplay $OUTPUT"
