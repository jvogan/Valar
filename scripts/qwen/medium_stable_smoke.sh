#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 - "$ROOT_DIR" "$@" <<'PY'
import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT_DIR = Path(sys.argv[1])
argv = sys.argv[2:]
DEFAULT_DAEMON_SECONDS = 267.0
DEFAULT_DAEMON_RTF = 2.0
DEFAULT_STREAM_SECONDS = 269.0
DEFAULT_STREAM_RTF = 2.0
DEFAULT_DAEMON_URL = "http://127.0.0.1:8787/v1"
DEFAULT_EXPECTED_MODELS = [
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/qwen/medium_stable_smoke.sh",
        description=(
            "Run the canonical medium stable-narrator daemon and stream matrix and "
            "fail if either surface drifts outside the current in-band smoke window."
        ),
    )
    parser.add_argument("--cli-bin", default=os.environ.get("QWEN_BENCH_CLI_BIN"))
    parser.add_argument("--daemon-url", default=os.environ.get("QWEN_BENCH_DAEMON_URL", DEFAULT_DAEMON_URL))
    parser.add_argument("--daemon-bin", default=os.environ.get("VALAR_DAEMON_BIN"))
    parser.add_argument("--stable-voice", default=os.environ.get("QWEN_BENCH_STABLE_VOICE"))
    parser.add_argument("--artifacts-dir", help="Optional benchmark artifacts root. Defaults to a temporary directory.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep artifacts even on success.")
    parser.add_argument("--keep-daemon", action="store_true", help="Leave the daemon running if this script starts it.")
    parser.add_argument("--daemon-start-timeout", type=float, default=45.0)
    parser.add_argument(
        "--expect-model",
        action="append",
        dest="expected_models",
        help="Warm-start model ID that must be resident before the smoke run. Defaults to Qwen Base + Qwen ASR.",
    )
    parser.add_argument("--min-audio-seconds", type=float, default=135.0, help="Minimum acceptable medium stable narrator audio duration per surface.")
    parser.add_argument("--seconds-tolerance", type=float, default=1.0, help="Extra wall-time tolerance for scheduler and warmup jitter.")
    parser.add_argument("--max-daemon-seconds", type=float, default=DEFAULT_DAEMON_SECONDS)
    parser.add_argument("--max-daemon-rtf", type=float, default=DEFAULT_DAEMON_RTF)
    parser.add_argument("--max-stream-seconds", type=float, default=DEFAULT_STREAM_SECONDS)
    parser.add_argument("--max-stream-rtf", type=float, default=DEFAULT_STREAM_RTF)
    return parser.parse_args(argv)


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


def daemon_reachable(daemon_url: str) -> bool:
    return fetch_json(daemon_url.rstrip("/") + "/health") is not None


def wait_for_ready_daemon(daemon_url: str, timeout_seconds: float, expected_models: list[str]) -> None:
    runtime_url = daemon_url.rstrip("/") + "/runtime"
    ready_url = daemon_url.rstrip("/") + "/ready"
    deadline = time.time() + timeout_seconds
    expected_set = set(expected_models)
    last_runtime = None
    last_ready = None
    while time.time() < deadline:
        last_ready = fetch_json(ready_url)
        last_runtime = fetch_json(runtime_url)
        if last_runtime is None or last_ready is None:
            time.sleep(1.0)
            continue

        resident_ids = {
            entry.get("id")
            for entry in (last_runtime.get("residentModels") or [])
            if isinstance(entry, dict) and entry.get("id")
        }
        effective_warm = set(last_runtime.get("effectiveWarmStartModels") or [])
        if (
            int(last_runtime.get("activeSynthesisCount") or 0) == 0
            and int(last_runtime.get("stalledSynthesisCount") or 0) == 0
            and bool(last_ready.get("residentTTSReady"))
            and bool(last_ready.get("residentASRReady"))
            and expected_set.issubset(effective_warm)
            and expected_set.issubset(resident_ids)
        ):
            return
        time.sleep(1.0)
    raise SystemExit(
        "Daemon did not reach the expected warm-ready state before the smoke run:\n"
        + json.dumps(
            {
                "ready": last_ready or {},
                "runtime": last_runtime or {},
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    benchmark_script = ROOT_DIR / "scripts" / "qwen" / "benchmark.sh"
    if not benchmark_script.exists():
        raise SystemExit(f"Benchmark script not found: {benchmark_script}")
    expected_models = sorted(set(args.expected_models or DEFAULT_EXPECTED_MODELS))

    temp_artifacts_dir: Path | None = None
    if args.artifacts_dir:
        artifacts_root = Path(args.artifacts_dir).expanduser()
        artifacts_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_artifacts_dir = Path(tempfile.mkdtemp(prefix="valar-qwen-medium-smoke-"))
        artifacts_root = temp_artifacts_dir

    daemon_url = args.daemon_url.rstrip("/")
    daemon_process: subprocess.Popen | None = None
    daemon_log_handle = None

    cmd = [
        str(benchmark_script),
        "--cases",
        "medium",
        "--lanes",
        "stableNarrator",
        "--surfaces",
        "daemon,stream",
        "--artifacts-dir",
        str(artifacts_root),
    ]
    if args.cli_bin:
        cmd.extend(["--cli-bin", args.cli_bin])
    if daemon_url:
        cmd.extend(["--daemon-url", daemon_url])
    if args.stable_voice:
        cmd.extend(["--stable-voice", args.stable_voice])

    try:
        if not daemon_reachable(daemon_url):
            daemon_bin = resolve_daemon_bin(args.daemon_bin)
            if not daemon_bin.exists():
                raise SystemExit(f"Daemon binary not found: {daemon_bin}")
            daemon_log_path = artifacts_root / "medium-stable-smoke-daemon.log"
            daemon_log_handle = open(daemon_log_path, "ab")
            daemon_process = subprocess.Popen(
                [str(daemon_bin)],
                stdout=daemon_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        wait_for_ready_daemon(daemon_url, args.daemon_start_timeout, expected_models)

        completed = subprocess.run(cmd, capture_output=True, text=True)
        stdout = completed.stdout
        stderr = completed.stderr
        summary_match = re.search(r"Summary JSON:\s+(.+)", stdout)
        if not summary_match:
            print(stdout, file=sys.stderr, end="" if stdout.endswith("\n") else "\n")
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
            raise SystemExit("Could not locate Summary JSON path in benchmark output.")

        summary_path = Path(summary_match.group(1).strip())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results = {
            result["surface"]: result
            for result in summary.get("results", [])
            if result.get("case") == "medium" and result.get("voiceBehavior") == "stableNarrator"
        }

        failures: list[str] = []
        advisories: list[str] = []
        expectations = {
            "daemon": (args.max_daemon_seconds, args.max_daemon_rtf),
            "stream": (args.max_stream_seconds, args.max_stream_rtf),
        }

        for surface, (max_seconds, max_rtf) in expectations.items():
            result = results.get(surface)
            if result is None:
                failures.append(f"missing {surface} result in summary")
                continue
            if result.get("exitCode", 1) != 0:
                failures.append(f"{surface} run exited with {result.get('exitCode')}")
            if result.get("validationIssues"):
                failures.append(f"{surface} validation failed: {'; '.join(result['validationIssues'])}")
            elapsed = float(result.get("elapsedSeconds") or 0.0)
            audio_seconds = float(result.get("audioSeconds") or 0.0)
            rtf = float(result.get("rtf") or 0.0)
            if audio_seconds < args.min_audio_seconds:
                failures.append(f"{surface} audio {audio_seconds:.3f}s fell below {args.min_audio_seconds:.3f}s")
            elapsed_limit = max_seconds + args.seconds_tolerance
            if elapsed > elapsed_limit:
                advisories.append(
                    f"{surface} elapsed {elapsed:.3f}s exceeds {max_seconds:.3f}s nominal limit "
                    f"(+{args.seconds_tolerance:.3f}s tolerance)"
                )
            if rtf > max_rtf:
                failures.append(f"{surface} RTF {rtf:.3f}x exceeds {max_rtf:.3f}x")

        if failures:
            print("Medium stable smoke failed:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            print(f"Summary JSON: {summary_path}", file=sys.stderr)
            if stdout:
                print(stdout, file=sys.stderr, end="" if stdout.endswith("\n") else "\n")
            if stderr:
                print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
            return 1

        print("Medium stable smoke passed.")
        for surface in ("daemon", "stream"):
            result = results[surface]
            print(
                f"{surface}: {float(result['elapsedSeconds']):.3f}s synth / "
                f"{float(result['audioSeconds']):.3f}s audio / {float(result['rtf']):.3f}x"
            )
        for advisory in advisories:
            print(f"Advisory: {advisory}")
        print(f"Summary JSON: {summary_path}")

        if temp_artifacts_dir is not None and not args.keep_artifacts:
            shutil.rmtree(temp_artifacts_dir, ignore_errors=True)
        return 0
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
