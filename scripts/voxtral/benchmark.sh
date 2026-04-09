#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS_ROOT="${VOXTRAL_ARTIFACTS_DIR:-$ROOT_DIR/artifacts/voxtral-benchmarks}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ARTIFACTS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

MODEL_ID="${VOXTRAL_MODEL_ID:-mlx-community/Voxtral-4B-TTS-2603-mlx-4bit}"
VOICE_ID="${VOXTRAL_VOICE:-${VOXTRAL_VOICE_ID:-neutral_female}}"
LANGUAGE="${VOXTRAL_LANGUAGE:-en}"
CLI_BIN="${VOXTRAL_CLI_BIN:-${VALARTTS_BIN:-}}"
DAEMON_URL="${VOXTRAL_DAEMON_URL:-http://127.0.0.1:8787/v1}"
SHORT_PROMPT="${VOXTRAL_SHORT_PROMPT:-Hello from Voxtral. This benchmark measures the fast path.}"
LONG_PROMPT="${VOXTRAL_LONG_PROMPT:-Hello from Voxtral. This benchmark uses a longer paragraph so the daemon has time to stream multiple chunks while we measure first audio, total render time, and chunk cadence. The goal is reproducibility on Apple Silicon rather than a fixed score, so every run saves raw logs and machine details as artifacts.}"

usage() {
  cat <<EOF
Usage: scripts/voxtral/benchmark.sh [options]

Options:
  --cli-bin PATH        Path to the built valartts CLI binary.
  --daemon-url URL      Base daemon URL. Default: $DAEMON_URL
  --model ID            Model identifier. Default: $MODEL_ID
  --voice ID            Voxtral preset voice. Accepts canonical names, aliases, or random. Default: $VOICE_ID
  --language CODE       Language hint. Default: $LANGUAGE
  --artifacts-dir PATH  Artifact root directory.
  --help                Show this help text.

Environment fallbacks:
  VOXTRAL_CLI_BIN / VALARTTS_BIN
  VOXTRAL_DAEMON_URL
  VOXTRAL_MODEL_ID
  VOXTRAL_VOICE / VOXTRAL_VOICE_ID
  VOXTRAL_LANGUAGE
  VOXTRAL_ARTIFACTS_DIR

Artifacts written to:
  $RUN_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli-bin)
      CLI_BIN="$2"
      shift 2
      ;;
    --daemon-url)
      DAEMON_URL="$2"
      shift 2
      ;;
    --model)
      MODEL_ID="$2"
      shift 2
      ;;
    --voice|--voice-id)
      VOICE_ID="$2"
      shift 2
      ;;
    --language)
      LANGUAGE="$2"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_ROOT="$2"
      RUN_DIR="$ARTIFACTS_ROOT/$RUN_ID"
      mkdir -p "$RUN_DIR"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CLI_BIN" ]]; then
  echo "Missing CLI binary. Pass --cli-bin or set VOXTRAL_CLI_BIN / VALARTTS_BIN." >&2
  exit 2
fi

if [[ ! -x "$CLI_BIN" ]]; then
  echo "CLI binary is not executable: $CLI_BIN" >&2
  exit 2
fi

trim_slash() {
  local value="$1"
  while [[ "$value" == */ ]]; do
    value="${value%/}"
  done
  printf '%s\n' "$value"
}

json_escape() {
  python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

write_machine_metadata() {
  python3 - "$RUN_DIR/machine.json" "$ROOT_DIR" <<'PY'
import json
import os
import platform
import subprocess
import sys

out_path = sys.argv[1]
root_dir = sys.argv[2]

def run(*args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return None

payload = {
    "timestamp_utc": run("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": run("sysctl", "-n", "machdep.cpu.brand_string"),
    "cpu_count": run("sysctl", "-n", "hw.ncpu"),
    "memory_bytes": run("sysctl", "-n", "hw.memsize"),
    "product_name": run("sw_vers", "-productName"),
    "product_version": run("sw_vers", "-productVersion"),
    "build_version": run("sw_vers", "-buildVersion"),
    "git_head": run("git", "-C", root_dir, "rev-parse", "HEAD"),
}

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

run_timed_cli() {
  local label="$1"
  local prompt="$2"
  local output_path="$3"
  local stdout_path="$RUN_DIR/${label}.stdout.log"
  local stderr_path="$RUN_DIR/${label}.stderr.log"
  local stats_path="$RUN_DIR/${label}.json"

  python3 - "$CLI_BIN" "$MODEL_ID" "$VOICE_ID" "$LANGUAGE" "$prompt" "$output_path" "$stdout_path" "$stderr_path" "$stats_path" <<'PY'
import json
import os
import re
import subprocess
import sys
import time

(
    cli_bin,
    model_id,
    voice_id,
    language,
    prompt,
    output_path,
    stdout_path,
    stderr_path,
    stats_path,
) = sys.argv[1:]

cmd = [
    "/usr/bin/time",
    "-l",
    cli_bin,
    "speak",
    "--model",
    model_id,
    "--voice",
    voice_id,
    "--language",
    language,
    "--text",
    prompt,
    "--output",
    output_path,
]

start = time.perf_counter()
with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(stderr_path, "w", encoding="utf-8") as stderr_handle:
    process = subprocess.run(cmd, stdout=stdout_handle, stderr=stderr_handle, text=True)
elapsed = time.perf_counter() - start

stderr_text = open(stderr_path, "r", encoding="utf-8").read()
peak_rss_kb = None
for line in stderr_text.splitlines():
    if "maximum resident set size" in line:
        match = re.search(r"(\d+)", line)
        if match:
            peak_rss_kb = int(match.group(1))
            break

payload = {
    "command": cmd,
    "elapsed_seconds": round(elapsed, 6),
    "exit_code": process.returncode,
    "output_path": output_path,
    "output_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
    "peak_rss_kb": peak_rss_kb,
}

with open(stats_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)

if process.returncode != 0:
    raise SystemExit(process.returncode)
PY
}

daemon_port() {
  python3 - "$DAEMON_URL" <<'PY'
from urllib.parse import urlparse
import sys
parsed = urlparse(sys.argv[1])
print(parsed.port or (443 if parsed.scheme == "https" else 80))
PY
}

daemon_pid() {
  local port
  port="$(daemon_port)"
  lsof -n -P -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true
}

sample_daemon_memory() {
  local pid="$1"
  local stop_file="$2"
  local sample_file="$3"
  : > "$sample_file"
  while kill -0 "$pid" 2>/dev/null && [[ ! -f "$stop_file" ]]; do
    ps -o rss= -p "$pid" | awk '{$1=$1;print $1}' >> "$sample_file" || true
    sleep 0.1
  done
}

stream_metrics() {
  local raw_sse="$RUN_DIR/stream.sse"
  local events_json="$RUN_DIR/stream-events.json"
  local summary_json="$RUN_DIR/stream-summary.json"
  local stop_file="$RUN_DIR/.stream-mem.stop"
  local sample_file="$RUN_DIR/daemon-rss-kb.txt"
  local pid=""
  pid="$(daemon_pid)"

  rm -f "$stop_file"
  if [[ -n "$pid" ]]; then
    sample_daemon_memory "$pid" "$stop_file" "$sample_file" &
    local sampler_pid="$!"
  else
    local sampler_pid=""
  fi

  local body
  body="$(python3 - "$MODEL_ID" "$VOICE_ID" "$LANGUAGE" "$SHORT_PROMPT" <<'PY'
import json
import sys
payload = {
    "input": sys.argv[4],
    "model": sys.argv[1],
    "voice": sys.argv[2],
    "language": sys.argv[3],
    "response_format": "pcm_f32le",
}
print(json.dumps(payload))
PY
)"

  curl -N -sS \
    -H "Content-Type: application/json" \
    -d "$body" \
    "$(trim_slash "$DAEMON_URL")/audio/speech/stream" \
    | tee "$raw_sse" \
    | python3 - "$events_json" "$summary_json" <<'PY'
import json
import statistics
import sys
import time

events_path = sys.argv[1]
summary_path = sys.argv[2]

events = []
current = {}
start = time.perf_counter()

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")
    now_ms = (time.perf_counter() - start) * 1000.0
    if line.startswith("event: "):
        current["event"] = line[7:]
    elif line.startswith("data: "):
        payload = line[6:]
        try:
            current["data"] = json.loads(payload)
        except json.JSONDecodeError:
            current["data"] = payload
    elif line == "" and current:
        current["received_ms"] = round(now_ms, 3)
        events.append(current)
        current = {}

with open(events_path, "w", encoding="utf-8") as handle:
    json.dump(events, handle, indent=2, sort_keys=True)

chunk_times = [event["received_ms"] for event in events if event.get("event") == "chunk"]
cadence = [round(b - a, 3) for a, b in zip(chunk_times, chunk_times[1:])]

summary = {
    "event_count": len(events),
    "chunk_count": len(chunk_times),
    "ttfa_ms": chunk_times[0] if chunk_times else None,
    "chunk_cadence_ms": cadence,
    "chunk_cadence_mean_ms": round(statistics.fmean(cadence), 3) if cadence else None,
    "chunk_cadence_median_ms": round(statistics.median(cadence), 3) if cadence else None,
}

with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
PY

  touch "$stop_file"
  if [[ -n "${sampler_pid:-}" ]]; then
    wait "$sampler_pid" || true
  fi

  python3 - "$sample_file" "$summary_json" <<'PY'
import json
import os
import sys

sample_path = sys.argv[1]
summary_path = sys.argv[2]

with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

if os.path.exists(sample_path):
    samples = []
    with open(sample_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw.isdigit():
                samples.append(int(raw))
    summary["daemon_peak_rss_kb"] = max(samples) if samples else None
    summary["daemon_rss_samples"] = len(samples)

with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
PY
}

write_machine_metadata

run_timed_cli "cold-short" "$SHORT_PROMPT" "$RUN_DIR/cold-short.wav"
run_timed_cli "warm-short" "$SHORT_PROMPT" "$RUN_DIR/warm-short.wav"
run_timed_cli "warm-long" "$LONG_PROMPT" "$RUN_DIR/warm-long.wav"
stream_metrics

python3 - "$RUN_DIR" "$MODEL_ID" "$VOICE_ID" "$LANGUAGE" <<'PY'
import json
import os
import sys

run_dir, model_id, voice_id, language = sys.argv[1:]

def load(name):
    path = os.path.join(run_dir, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

cold = load("cold-short.json")
warm = load("warm-short.json")
long_run = load("warm-long.json")
stream = load("stream-summary.json")
machine = load("machine.json")

results = {
    "model": model_id,
    "voice_id": voice_id,
    "language": language,
    "machine": machine,
    "metrics": {
        "cold_load_time_seconds": cold["elapsed_seconds"],
        "warm_load_time_seconds": warm["elapsed_seconds"],
        "full_render_seconds": {
            "short_prompt": warm["elapsed_seconds"],
            "long_prompt": long_run["elapsed_seconds"],
        },
        "peak_memory_kb": {
            "cold_cli": cold["peak_rss_kb"],
            "warm_cli": warm["peak_rss_kb"],
            "long_cli": long_run["peak_rss_kb"],
            "daemon_stream": stream.get("daemon_peak_rss_kb"),
        },
        "ttfa_ms": stream.get("ttfa_ms"),
        "stream_chunk_cadence_ms": {
            "samples": stream.get("chunk_cadence_ms", []),
            "mean": stream.get("chunk_cadence_mean_ms"),
            "median": stream.get("chunk_cadence_median_ms"),
        },
        "output_bytes": {
            "short_prompt": warm["output_bytes"],
            "long_prompt": long_run["output_bytes"],
        },
    },
    "artifacts": {
        "machine": "machine.json",
        "cold_short": "cold-short.json",
        "warm_short": "warm-short.json",
        "warm_long": "warm-long.json",
        "stream_events": "stream-events.json",
        "stream_summary": "stream-summary.json",
        "raw_stream": "stream.sse",
    },
}

with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, sort_keys=True)
PY

echo "Saved Voxtral benchmark artifacts to $RUN_DIR"
