#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 - "$ROOT_DIR" "$@" <<'PY'
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(sys.argv[1])
argv = sys.argv[2:]
DEFAULT_DAEMON_URL = "http://127.0.0.1:8787"
DEFAULT_EXPECTED_MODELS = [
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
]


def resolve_daemon_bin(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    if resolved := shutil.which("valarttsd"):
        return Path(resolved)
    return ROOT_DIR / "apps" / "ValarDaemon" / ".build" / "debug" / "valarttsd"


def fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def daemon_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/qwen/check_daemon_warm_start.sh",
        description=(
            "Fail fast if the daemon default warm set reaches ASR residency but leaves "
            "Qwen Base TTS out of the resident warm set."
        ),
    )
    parser.add_argument("--daemon-url", default=os.environ.get("VALAR_DAEMON_URL", DEFAULT_DAEMON_URL))
    parser.add_argument("--daemon-bin", default=os.environ.get("VALAR_DAEMON_BIN"))
    parser.add_argument("--timeout", type=float, default=90.0, help="Seconds to wait for resident warm start readiness.")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds.")
    parser.add_argument("--keep-daemon", action="store_true", help="Leave the daemon running if this script starts it.")
    parser.add_argument(
        "--expect-model",
        action="append",
        dest="expected_models",
        help="Warm-start model ID that must appear in the effective and resident warm set. Defaults to Qwen Base + Qwen ASR.",
    )
    return parser.parse_args(argv)


def summarize_state(ready: dict | None, runtime: dict | None) -> dict:
    runtime = runtime or {}
    ready = ready or {}
    resident_ids = sorted(
        entry.get("id")
        for entry in (runtime.get("residentModels") or [])
        if isinstance(entry, dict) and entry.get("id")
    )
    return {
        "residentTTSReady": bool(ready.get("residentTTSReady")),
        "residentASRReady": bool(ready.get("residentASRReady")),
        "effectiveWarmStartModels": sorted(runtime.get("effectiveWarmStartModels") or []),
        "prewarmedModels": sorted(runtime.get("prewarmedModels") or []),
        "warmingModels": sorted(runtime.get("warmingModels") or []),
        "residentModels": resident_ids,
        "activeSynthesisCount": int(runtime.get("activeSynthesisCount") or 0),
    }


def main() -> int:
    args = parse_args()
    daemon_url = args.daemon_url.rstrip("/")
    expected_models = sorted(set(args.expected_models or DEFAULT_EXPECTED_MODELS))
    daemon_bin = resolve_daemon_bin(args.daemon_bin)
    daemon_process: subprocess.Popen | None = None
    daemon_log_handle = None
    daemon_was_running = daemon_reachable(daemon_url)

    try:
        if not daemon_was_running:
            if not daemon_bin.exists():
                raise SystemExit(f"Daemon binary not found: {daemon_bin}")
            log_path = ROOT_DIR / "artifacts" / "qwen-warm-start.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            daemon_log_handle = open(log_path, "ab")
            daemon_process = subprocess.Popen(
                [str(daemon_bin)],
                stdout=daemon_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        deadline = time.time() + max(args.timeout, args.interval)
        last_ready = None
        last_runtime = None

        while time.time() < deadline:
            if not daemon_reachable(daemon_url):
                time.sleep(args.interval)
                continue

            last_ready = fetch_json(daemon_url + "/v1/ready")
            last_runtime = fetch_json(daemon_url + "/v1/runtime")
            summary = summarize_state(last_ready, last_runtime)
            resident_ids = set(summary["residentModels"])
            effective_warm = set(summary["effectiveWarmStartModels"])
            active_synthesis_count = int(summary["activeSynthesisCount"])

            if (
                set(expected_models).issubset(effective_warm)
                and set(expected_models).issubset(resident_ids)
                and summary["residentTTSReady"]
                and summary["residentASRReady"]
                and active_synthesis_count == 0
            ):
                print(
                    "Warm-start check passed: resident warm set includes "
                    + ", ".join(expected_models)
                )
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0

            time.sleep(args.interval)

        summary = summarize_state(last_ready, last_runtime)
        missing_effective = sorted(set(expected_models) - set(summary["effectiveWarmStartModels"]))
        missing_resident = sorted(set(expected_models) - set(summary["residentModels"]))
        failures: list[str] = []
        if missing_effective:
            failures.append("missing from effective warm set: " + ", ".join(missing_effective))
        if missing_resident:
            failures.append("missing from resident warm set: " + ", ".join(missing_resident))
        if not summary["residentTTSReady"]:
            failures.append("ready endpoint still reports residentTTSReady=false")
        if not summary["residentASRReady"]:
            failures.append("ready endpoint still reports residentASRReady=false")
        if int(summary["activeSynthesisCount"]) != 0:
            failures.append("daemon still reports active synthesis during warm-start check")
        failure_text = "; ".join(failures) if failures else "daemon did not reach warm-start success criteria before timeout"
        print(f"Warm-start check failed after {args.timeout:.1f}s: {failure_text}", file=sys.stderr)
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        if daemon_log_handle is not None:
            daemon_log_handle.close()
        if daemon_process is not None and not args.keep_daemon:
            try:
                os.killpg(daemon_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                daemon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(daemon_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                daemon_process.wait(timeout=5)


raise SystemExit(main())
PY
