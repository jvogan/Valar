#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$ROOT_DIR" "$@" <<'PY'
import argparse
import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


ROOT_DIR = Path(sys.argv[1])
argv = sys.argv[2:]
DEFAULT_CLI_BIN = ROOT_DIR / "apps" / "ValarCLI" / ".build" / "debug" / "valartts"
DEFAULT_DAEMON_URL = "http://127.0.0.1:8787/v1"
DEFAULT_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "voxtral-validation"
PRESET_FIXTURES = {
    "neutral_female": ("emma", "en", "Hello from Voxtral. The archive is ready."),
    "neutral_male": ("alex", "en", "Hello from Voxtral. The harbor is calm."),
    "cheerful_female": ("lily", "en", "Hello from Voxtral. The morning briefing is underway."),
    "fr_female": ("claire", "fr", "Bonjour. La salle de controle est prete."),
    "es_male": ("carlos", "es", "Hola. El puerto esta listo para partir."),
    "de_female": ("lena", "de", "Guten Tag. Das Signal bleibt klar und ruhig."),
    "pt_male": ("pedro", "pt", "Ola. O canal esta estavel nesta manha."),
    "hi_female": ("priya", "hi", "Namaste. Niyantran kaksh taiyar hai."),
    "ar_male": ("omar", "ar", "Marhaban. Ghurfat al tahakkum jahiza lil amaliyat."),
}
DEFAULT_REPRESENTATIVE_PRESETS = [
    "neutral_female",
    "neutral_male",
    "cheerful_female",
    "fr_female",
    "es_male",
    "de_female",
    "pt_male",
    "hi_female",
]
DEFAULT_IDENTIFIER_MODES = ["canonical", "alias"]
DEFAULT_EXCLUDED_RANDOM_PRESETS = ["neutral_male", "ar_male"]
DEFAULT_RANDOM_LANGUAGE = "en"
DEFAULT_RANDOM_PROMPT = "Hello from Voxtral. Choose any supported preset except the excluded random voices."
DAEMON_IDLE_TIMEOUT_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/voxtral/validate_presets.sh",
        description="Validate the representative Voxtral preset matrix across CLI, daemon, and streaming surfaces.",
    )
    parser.add_argument("--cli-bin", default=os.environ.get("VOXTRAL_VALIDATE_CLI_BIN", str(DEFAULT_CLI_BIN)))
    parser.add_argument("--daemon-url", default=os.environ.get("VOXTRAL_VALIDATE_DAEMON_URL", DEFAULT_DAEMON_URL))
    parser.add_argument("--model", default=os.environ.get("VOXTRAL_VALIDATE_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument(
        "--surfaces",
        default=os.environ.get("VOXTRAL_VALIDATE_SURFACES", "cli,daemon,stream"),
        help="Comma-separated list of surfaces: cli,daemon,stream",
    )
    parser.add_argument(
        "--presets",
        default=os.environ.get("VOXTRAL_VALIDATE_PRESETS", ",".join(DEFAULT_REPRESENTATIVE_PRESETS)),
        help="Comma-separated list of canonical preset IDs to validate",
    )
    parser.add_argument(
        "--identifier-modes",
        default=os.environ.get("VOXTRAL_VALIDATE_IDENTIFIER_MODES", ",".join(DEFAULT_IDENTIFIER_MODES)),
        help="Comma-separated identifier modes for representative preset validation: canonical,alias",
    )
    parser.add_argument(
        "--excluded-random-presets",
        default=os.environ.get(
            "VOXTRAL_VALIDATE_EXCLUDED_RANDOM_PRESETS",
            ",".join(DEFAULT_EXCLUDED_RANDOM_PRESETS),
        ),
        help="Comma-separated canonical preset IDs that stay excluded from random but should still be validated directly.",
    )
    parser.add_argument(
        "--random-runs",
        type=int,
        default=int(os.environ.get("VOXTRAL_VALIDATE_RANDOM_RUNS", "1")),
        help="How many live random-smoke runs to execute per surface. Set to 0 to disable.",
    )
    parser.add_argument("--artifacts-dir", default=os.environ.get("VOXTRAL_VALIDATE_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    parser.add_argument("--dry-run", action="store_true", help="Print the planned validation matrix and exit.")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def wav_duration_seconds(path: Path) -> float:
    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0.0
    sample_rate = None
    block_align = None
    data_size = None
    cursor = 12
    while cursor + 8 <= len(data):
        chunk_id = data[cursor:cursor + 4]
        chunk_size = int.from_bytes(data[cursor + 4:cursor + 8], "little")
        start = cursor + 8
        end = start + chunk_size
        if end > len(data):
            break
        if chunk_id == b"fmt " and chunk_size >= 16:
            sample_rate = int.from_bytes(data[start + 4:start + 8], "little")
            block_align = int.from_bytes(data[start + 12:start + 14], "little")
        elif chunk_id == b"data":
            data_size = chunk_size
        cursor = end + (chunk_size % 2)
    if not sample_rate or not block_align or data_size is None or block_align == 0:
        return 0.0
    return (data_size / block_align) / float(sample_rate)


def write_float_stream_wav(pcm_f32le: bytes, sample_rate: int, out_path: Path) -> int:
    samples = len(pcm_f32le) // 4
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for (sample,) in struct.iter_unpack("<f", pcm_f32le):
            clipped = max(-1.0, min(1.0, float(sample)))
            frames.extend(struct.pack("<h", int(clipped * 32767.0)))
        wav_file.writeframes(bytes(frames))
    return samples


def sha256_hex(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preset_catalog() -> dict[str, tuple[str, str, str]]:
    return dict(PRESET_FIXTURES)


def runtime_is_idle(runtime: dict | None) -> bool:
    if not runtime:
        return False
    if int(runtime.get("activeSynthesisCount") or 0) != 0:
        return False
    if runtime.get("activeSynthesisRequests"):
        return False
    if int(runtime.get("stalledSynthesisCount") or 0) != 0:
        return False
    for model in runtime.get("residentModels") or []:
        if int(model.get("activeSessionCount") or 0) != 0:
            return False
    return True


def runtime_idle_snapshot(runtime: dict | None) -> dict:
    runtime = runtime or {}
    resident_models = runtime.get("residentModels") or []
    return {
        "activeSynthesisCount": int(runtime.get("activeSynthesisCount") or 0),
        "stalledSynthesisCount": int(runtime.get("stalledSynthesisCount") or 0),
        "activeRequestCount": len(runtime.get("activeSynthesisRequests") or []),
        "residentActiveSessionCount": sum(int(model.get("activeSessionCount") or 0) for model in resident_models),
        "lastSynthesisCompletionReason": runtime.get("lastSynthesisCompletionReason"),
    }


def fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def wait_for_idle_runtime(daemon_url: str, timeout_seconds: float = DAEMON_IDLE_TIMEOUT_SECONDS) -> dict:
    deadline = time.time() + timeout_seconds
    last_runtime = {}
    while time.time() < deadline:
        last_runtime = fetch_json(daemon_url.rstrip("/") + "/runtime") or {}
        if runtime_is_idle(last_runtime):
            return runtime_idle_snapshot(last_runtime)
        time.sleep(0.5)
    return runtime_idle_snapshot(last_runtime)


def daemon_request_json(model_id: str, voice: str, language: str, prompt_text: str, response_format: str) -> bytes:
    return json.dumps(
        {
            "model": model_id,
            "input": prompt_text,
            "voice": voice,
            "language": language,
            "response_format": response_format,
        }
    ).encode("utf-8")


def run_cli_case(
    cli_bin: Path,
    model_id: str,
    canonical: str,
    voice_input: str,
    alias: str,
    language: str,
    prompt_text: str,
    run_dir: Path,
    case_label: str,
    case_kind: str,
    identifier_mode: str | None,
) -> dict:
    output_path = run_dir / f"{case_label}-cli.wav"
    cmd = [
        str(cli_bin),
        "speak",
        "--model", model_id,
        "--voice", voice_input,
        "--language", language,
        "--format", "wav",
        "--output", str(output_path),
        "--text", prompt_text,
    ]
    env = os.environ.copy()
    env["VALARTTS_ENABLE_NONCOMMERCIAL_MODELS"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    audio_seconds = wav_duration_seconds(output_path) if completed.returncode == 0 and output_path.exists() else 0.0
    return {
        "surface": "cli",
        "caseKind": case_kind,
        "identifierMode": identifier_mode,
        "canonicalPreset": canonical,
        "alias": alias,
        "voiceInput": voice_input,
        "language": language,
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(output_path),
        "sha256": sha256_hex(output_path),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "exitCode": completed.returncode,
    }


def run_daemon_case(
    daemon_url: str,
    model_id: str,
    canonical: str,
    voice_input: str,
    alias: str,
    language: str,
    prompt_text: str,
    run_dir: Path,
    case_label: str,
    case_kind: str,
    identifier_mode: str | None,
) -> dict:
    output_path = run_dir / f"{case_label}-daemon.wav"
    pre_runtime = wait_for_idle_runtime(daemon_url)
    request = urllib.request.Request(
        daemon_url.rstrip("/") + "/audio/speech",
        data=daemon_request_json(model_id, voice_input, language, prompt_text, "wav"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = None
    error_message = None
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            status = response.status
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        status = error.code
        error_message = error.read().decode("utf-8", errors="replace")
    except Exception as error:
        error_message = str(error)
    elapsed = time.perf_counter() - started
    audio_seconds = wav_duration_seconds(output_path) if status == 200 and output_path.exists() else 0.0
    post_runtime = wait_for_idle_runtime(daemon_url)
    return {
        "surface": "daemon",
        "caseKind": case_kind,
        "identifierMode": identifier_mode,
        "canonicalPreset": canonical,
        "alias": alias,
        "voiceInput": voice_input,
        "language": language,
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(output_path),
        "sha256": sha256_hex(output_path),
        "httpStatus": status,
        "error": error_message,
        "preRunIdle": runtime_is_idle(pre_runtime),
        "postRunRuntime": post_runtime,
        "completionReason": post_runtime.get("lastSynthesisCompletionReason"),
        "exitCode": 0 if status == 200 else 1,
    }


def parse_sse_event_stream(response) -> list[tuple[str, str]]:
    events = []
    current_event = "message"
    data_lines = []
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        if not line:
            if data_lines:
                events.append((current_event, "\n".join(data_lines)))
            current_event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        events.append((current_event, "\n".join(data_lines)))
    return events


def run_stream_case(
    daemon_url: str,
    model_id: str,
    canonical: str,
    voice_input: str,
    alias: str,
    language: str,
    prompt_text: str,
    run_dir: Path,
    case_label: str,
    case_kind: str,
    identifier_mode: str | None,
) -> dict:
    raw_sse_path = run_dir / f"{case_label}-stream.sse.log"
    pcm_path = run_dir / f"{case_label}-stream.pcm_f32le"
    wav_path = run_dir / f"{case_label}-stream.wav"
    pre_runtime = wait_for_idle_runtime(daemon_url)
    request = urllib.request.Request(
        daemon_url.rstrip("/") + "/audio/speech/stream",
        data=daemon_request_json(model_id, voice_input, language, prompt_text, "pcm_f32le"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = None
    error_message = None
    pcm_chunks = bytearray()
    chunk_count = 0
    sample_rate = 24_000
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            status = response.status
            events = parse_sse_event_stream(response)
            raw_sse_path.write_text(
                "\n".join(f"{event}\t{payload}" for event, payload in events) + ("\n" if events else ""),
                encoding="utf-8",
            )
            for event_name, payload in events:
                if event_name == "chunk":
                    event = json.loads(payload)
                    chunk_count += 1
                    sample_rate = int(event.get("sampleRate") or sample_rate)
                    pcm_chunks.extend(base64.b64decode(event["data"]))
                elif event_name == "error":
                    error_message = payload
    except urllib.error.HTTPError as error:
        status = error.code
        error_message = error.read().decode("utf-8", errors="replace")
    except Exception as error:
        error_message = str(error)
    elapsed = time.perf_counter() - started
    pcm_path.write_bytes(bytes(pcm_chunks))
    total_samples = write_float_stream_wav(bytes(pcm_chunks), sample_rate, wav_path) if pcm_chunks else 0
    audio_seconds = 0.0 if sample_rate == 0 else total_samples / sample_rate
    post_runtime = wait_for_idle_runtime(daemon_url)
    return {
        "surface": "stream",
        "caseKind": case_kind,
        "identifierMode": identifier_mode,
        "canonicalPreset": canonical,
        "alias": alias,
        "voiceInput": voice_input,
        "language": language,
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(wav_path),
        "sha256": sha256_hex(wav_path),
        "streamChunkCount": chunk_count,
        "httpStatus": status,
        "error": error_message,
        "preRunIdle": runtime_is_idle(pre_runtime),
        "postRunRuntime": post_runtime,
        "completionReason": post_runtime.get("lastSynthesisCompletionReason"),
        "exitCode": 0 if status == 200 and error_message is None else 1,
    }


def validate_result(result: dict) -> list[str]:
    issues: list[str] = []
    if result.get("exitCode", 1) != 0:
        issues.append(f"{result['surface']} transport/process failed")
        return issues

    if float(result.get("audioSeconds") or 0.0) <= 0.0:
        issues.append("audio duration was zero")

    if result["surface"] in {"daemon", "stream"}:
        if not result.get("preRunIdle", False):
            issues.append("daemon was not idle before the case started")
        if result.get("completionReason") != "completed":
            issues.append(f"daemon completion reason was '{result.get('completionReason')}'")
        post_runtime = result.get("postRunRuntime") or {}
        if not runtime_is_idle(post_runtime):
            issues.append(
                "daemon did not return to idle state after the run "
                f"(activeSynthesis={post_runtime.get('activeSynthesisCount')}, "
                f"activeRequests={post_runtime.get('activeRequestCount')}, "
                f"residentActiveSessions={post_runtime.get('residentActiveSessionCount')}, "
                f"stalled={post_runtime.get('stalledSynthesisCount')})"
            )
    return issues


def markdown_summary(results: list[dict]) -> str:
    lines = [
        "# Voxtral Representative Preset Validation",
        "",
        "| Case | Mode | Preset | Voice Input | Alias | Surface | Language | Elapsed | Audio | RTF | Exit | SHA-256 | Validation | Notes |",
        "|------|------|--------|-------------|-------|---------|----------|---------|-------|-----|------|---------|------------|-------|",
    ]
    for result in results:
        elapsed = f"{result.get('elapsedSeconds', 0):.3f}s"
        audio = f"{result.get('audioSeconds', 0):.3f}s"
        rtf_value = result.get("rtf")
        rtf = f"{rtf_value:.3f}x" if isinstance(rtf_value, (int, float)) else "-"
        sha = result.get("sha256") or "-"
        note = result.get("error") or ""
        if result.get("surface") == "stream" and result.get("streamChunkCount") is not None:
            note = (note + f" chunks={result['streamChunkCount']}").strip()
        validation = "pass" if not result.get("validationIssues") else "; ".join(result["validationIssues"])
        mode = result.get("identifierMode") or result.get("caseKind") or "-"
        lines.append(
            f"| {result.get('caseKind', '-') } | {mode} | {result['canonicalPreset']} | {result.get('voiceInput', '-') } | {result['alias']} | {result['surface']} | {result['language']} | "
            f"{elapsed} | {audio} | {rtf} | {result.get('exitCode', 1)} | {sha} | {validation} | {note or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cli_bin = Path(args.cli_bin).expanduser()
    daemon_url = args.daemon_url.rstrip("/")
    artifacts_root = Path(args.artifacts_dir).expanduser()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    presets_index = preset_catalog()
    presets = split_csv(args.presets)
    surfaces = split_csv(args.surfaces)
    identifier_modes = split_csv(args.identifier_modes)
    excluded_random_presets = split_csv(args.excluded_random_presets)
    valid_surfaces = {"cli", "daemon", "stream"}
    valid_identifier_modes = {"canonical", "alias"}
    for preset in presets:
        if preset not in presets_index:
            raise SystemExit(f"Unknown preset '{preset}'.")
    for preset in excluded_random_presets:
        if preset not in presets_index:
            raise SystemExit(f"Unknown excluded-random preset '{preset}'.")
    for surface in surfaces:
        if surface not in valid_surfaces:
            raise SystemExit(f"Unknown surface '{surface}'. Valid surfaces: {sorted(valid_surfaces)}")
    for mode in identifier_modes:
        if mode not in valid_identifier_modes:
            raise SystemExit(
                f"Unknown identifier mode '{mode}'. Valid identifier modes: {sorted(valid_identifier_modes)}"
            )
    if args.random_runs < 0:
        raise SystemExit("--random-runs must be >= 0")

    plan = {
        "runID": run_id,
        "artifactsDir": str(run_dir),
        "modelID": args.model,
        "daemonURL": daemon_url,
        "cliBin": str(cli_bin),
        "presets": presets,
        "identifierModes": identifier_modes,
        "excludedRandomPresets": excluded_random_presets,
        "randomRuns": args.random_runs,
        "surfaces": surfaces,
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if "cli" in surfaces and not cli_bin.exists():
        raise SystemExit(f"CLI binary not found: {cli_bin}")

    results = []
    failures = 0
    validation_failures = 0
    for preset in presets:
        alias, language, prompt_text = presets_index[preset]
        for identifier_mode in identifier_modes:
            voice_input = preset if identifier_mode == "canonical" else alias
            case_label = f"{preset}-{identifier_mode}"
            for surface in surfaces:
                if surface == "cli":
                    result = run_cli_case(
                        cli_bin,
                        args.model,
                        preset,
                        voice_input,
                        alias,
                        language,
                        prompt_text,
                        run_dir,
                        case_label,
                        "representative",
                        identifier_mode,
                    )
                elif surface == "daemon":
                    result = run_daemon_case(
                        daemon_url,
                        args.model,
                        preset,
                        voice_input,
                        alias,
                        language,
                        prompt_text,
                        run_dir,
                        case_label,
                        "representative",
                        identifier_mode,
                    )
                else:
                    result = run_stream_case(
                        daemon_url,
                        args.model,
                        preset,
                        voice_input,
                        alias,
                        language,
                        prompt_text,
                        run_dir,
                        case_label,
                        "representative",
                        identifier_mode,
                    )
                result["validationIssues"] = validate_result(result)
                results.append(result)
                if result.get("exitCode", 1) != 0:
                    failures += 1
                if result["validationIssues"]:
                    validation_failures += 1

    for preset in excluded_random_presets:
        alias, language, prompt_text = presets_index[preset]
        case_label = f"{preset}-excluded-direct"
        for surface in surfaces:
            if surface == "cli":
                result = run_cli_case(
                    cli_bin,
                    args.model,
                    preset,
                    preset,
                    alias,
                    language,
                    prompt_text,
                    run_dir,
                    case_label,
                    "excludedRandomDirect",
                    "canonical",
                )
            elif surface == "daemon":
                result = run_daemon_case(
                    daemon_url,
                    args.model,
                    preset,
                    preset,
                    alias,
                    language,
                    prompt_text,
                    run_dir,
                    case_label,
                    "excludedRandomDirect",
                    "canonical",
                )
            else:
                result = run_stream_case(
                    daemon_url,
                    args.model,
                    preset,
                    preset,
                    alias,
                    language,
                    prompt_text,
                    run_dir,
                    case_label,
                    "excludedRandomDirect",
                    "canonical",
                )
            result["validationIssues"] = validate_result(result)
            results.append(result)
            if result.get("exitCode", 1) != 0:
                failures += 1
            if result["validationIssues"]:
                validation_failures += 1

    for run_index in range(args.random_runs):
        case_label = f"random-smoke-{run_index + 1}"
        for surface in surfaces:
            if surface == "cli":
                result = run_cli_case(
                    cli_bin,
                    args.model,
                    "random",
                    "random",
                    "random",
                    DEFAULT_RANDOM_LANGUAGE,
                    DEFAULT_RANDOM_PROMPT,
                    run_dir,
                    case_label,
                    "random",
                    None,
                )
            elif surface == "daemon":
                result = run_daemon_case(
                    daemon_url,
                    args.model,
                    "random",
                    "random",
                    "random",
                    DEFAULT_RANDOM_LANGUAGE,
                    DEFAULT_RANDOM_PROMPT,
                    run_dir,
                    case_label,
                    "random",
                    None,
                )
            else:
                result = run_stream_case(
                    daemon_url,
                    args.model,
                    "random",
                    "random",
                    "random",
                    DEFAULT_RANDOM_LANGUAGE,
                    DEFAULT_RANDOM_PROMPT,
                    run_dir,
                    case_label,
                    "random",
                    None,
                )
            result["validationIssues"] = validate_result(result)
            results.append(result)
            if result.get("exitCode", 1) != 0:
                failures += 1
            if result["validationIssues"]:
                validation_failures += 1

    summary = {
        "runID": run_id,
        "artifactsDir": str(run_dir),
        "modelID": args.model,
        "daemonURL": daemon_url,
        "identifierModes": identifier_modes,
        "excludedRandomPresets": excluded_random_presets,
        "randomRuns": args.random_runs,
        "results": results,
        "failureCount": failures,
        "validationFailureCount": validation_failures,
        "distinctSHA256Count": len({result["sha256"] for result in results if result.get("sha256")}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown_summary(results), encoding="utf-8")

    print(f"Saved Voxtral preset validation artifacts to {run_dir}")
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Summary Markdown: {run_dir / 'summary.md'}")
    return 1 if failures or validation_failures else 0


raise SystemExit(main())
PY
