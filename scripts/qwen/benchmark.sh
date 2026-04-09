#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 - "$ROOT_DIR" "$@" <<'PY'
import argparse
import base64
import json
import math
import os
import platform
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


ROOT_DIR = Path(sys.argv[1])
argv = sys.argv[2:]
PROMPTS_DIR = ROOT_DIR / "scripts" / "qwen" / "prompts"
DEFAULT_CLI_BIN = ROOT_DIR / "apps" / "ValarCLI" / ".build" / "debug" / "valartts"
DEFAULT_DAEMON_URL = "http://127.0.0.1:8787/v1"
DEFAULT_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
DEFAULT_EXPRESSIVE_VOICE = "The Architect Dark"
DEFAULT_STABLE_VOICE = "The Architect Dark Stable"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "qwen-benchmarks"
CASE_FILES = {
    "short": PROMPTS_DIR / "01-short.txt",
    "medium": PROMPTS_DIR / "02-medium.txt",
    "long": PROMPTS_DIR / "03-long.txt",
}
CASE_EXPECTATIONS = {
    ("short", "expressive"): {"segmented": False},
    ("short", "stableNarrator"): {"segmented": False},
    ("medium", "expressive"): {"segmented": True},
    ("medium", "stableNarrator"): {"segmented": True},
    ("long", "expressive"): {"segmented": True},
    ("long", "stableNarrator"): {"segmented": True, "minAudioSeconds": 240.0},
}
DAEMON_IDLE_TIMEOUT_SECONDS = 30.0
ACTIVE_CHILD_PROCESSES: set[subprocess.Popen] = set()


def terminate_active_children() -> None:
    active = [process for process in list(ACTIVE_CHILD_PROCESSES) if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 2.0
    for process in active:
        remaining = max(0.0, deadline - time.time())
        if remaining <= 0:
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    for process in active:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def handle_termination_signal(signum, _frame) -> None:
    terminate_active_children()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGINT, handle_termination_signal)
signal.signal(signal.SIGTERM, handle_termination_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/qwen/benchmark.sh",
        description=(
            "Benchmark Qwen expressive and stableNarrator lanes across CLI, "
            "daemon non-stream, and daemon stream surfaces."
        ),
    )
    parser.add_argument("--cli-bin", default=os.environ.get("QWEN_BENCH_CLI_BIN", str(DEFAULT_CLI_BIN)))
    parser.add_argument("--daemon-url", default=os.environ.get("QWEN_BENCH_DAEMON_URL", DEFAULT_DAEMON_URL))
    parser.add_argument("--model", default=os.environ.get("QWEN_BENCH_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--expressive-voice", default=os.environ.get("QWEN_BENCH_EXPRESSIVE_VOICE", DEFAULT_EXPRESSIVE_VOICE))
    parser.add_argument("--stable-voice", default=os.environ.get("QWEN_BENCH_STABLE_VOICE", DEFAULT_STABLE_VOICE))
    parser.add_argument(
        "--cases",
        default=os.environ.get("QWEN_BENCH_CASES", "short,medium,long"),
        help="Comma-separated list of cases: short,medium,long",
    )
    parser.add_argument(
        "--lanes",
        default=os.environ.get("QWEN_BENCH_LANES", "expressive,stableNarrator"),
        help="Comma-separated list of voice behaviors: expressive,stableNarrator",
    )
    parser.add_argument(
        "--surfaces",
        default=os.environ.get("QWEN_BENCH_SURFACES", "cli,daemon,stream"),
        help="Comma-separated list of surfaces: cli,daemon,stream",
    )
    parser.add_argument("--artifacts-dir", default=os.environ.get("QWEN_BENCH_ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    parser.add_argument("--dry-run", action="store_true", help="Print the planned benchmark matrix and exit.")
    parser.add_argument("--no-validate", action="store_true", help="Record benchmark data without enforcing the built-in acceptance checks.")
    return parser.parse_args(argv)


def split_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def run_command(cmd: list[str], stdout_path: Path | None = None, stderr_path: Path | None = None) -> tuple[int, float]:
    start = time.perf_counter()
    stdout_handle = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
    stderr_handle = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        ACTIVE_CHILD_PROCESSES.add(process)
        exit_code = process.wait()
    finally:
        if process is not None:
            ACTIVE_CHILD_PROCESSES.discard(process)
        if stdout_path:
            stdout_handle.close()
        if stderr_path:
            stderr_handle.close()
    elapsed = time.perf_counter() - start
    return exit_code, elapsed


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
        chunk_start = cursor + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            break

        if chunk_id == b"fmt " and chunk_size >= 16:
            sample_rate = int.from_bytes(data[chunk_start + 4:chunk_start + 8], "little")
            block_align = int.from_bytes(data[chunk_start + 12:chunk_start + 14], "little")
        elif chunk_id == b"data":
            data_size = chunk_size

        cursor = chunk_end + (chunk_size % 2)

    if not sample_rate or not block_align or data_size is None or block_align == 0:
        return 0.0
    frame_count = data_size / block_align
    return frame_count / float(sample_rate)


def peak_rss_kb_from_time(stderr_path: Path) -> int | None:
    if not stderr_path.exists():
        return None
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "maximum resident set size" not in line:
            continue
        match = re.search(r"(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def fetch_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def runtime_sampler(daemon_url: str, trace_path: Path, stop_event: threading.Event) -> None:
    runtime_url = daemon_url.rstrip("/") + "/runtime"
    with trace_path.open("w", encoding="utf-8") as handle:
        while not stop_event.is_set():
            record = {
                "timestamp": time.time(),
                "runtime": fetch_json(runtime_url),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            stop_event.wait(1.0)


def summarize_runtime_trace(trace_path: Path) -> dict:
    max_active = 0
    max_stalled = 0
    max_segment_index = 0
    max_segment_count = 0
    max_chunk_character_count = 0
    max_generated_token_count = 0
    max_prefill_token_count = 0
    max_segment_wall_time_seconds = 0.0
    max_segment_audio_duration_seconds = 0.0
    max_segment_prefill_time_seconds = 0.0
    max_segment_decode_time_seconds = 0.0
    max_anchor_segment_decode_time_seconds = 0.0
    max_continuation_segment_decode_time_seconds = 0.0
    max_sampling_time_seconds = 0.0
    max_eval_time_seconds = 0.0
    max_token_materialization_time_seconds = 0.0
    max_embedding_assembly_time_seconds = 0.0
    max_talker_forward_time_seconds = 0.0
    max_code_predictor_time_seconds = 0.0
    anchor_segments_observed: set[tuple[str, int]] = set()
    continuation_segments_observed: set[tuple[str, int]] = set()
    continuation_outlier_segments: set[tuple[str, int]] = set()
    max_token_counts_seen: set[int] = set()
    execution_modes_seen: set[str] = set()
    requests_seen: set[str] = set()
    if not trace_path.exists():
        return {
            "maxActiveSynthesisCount": 0,
            "maxStalledSynthesisCount": 0,
            "maxSegmentIndex": 0,
            "maxSegmentCount": 0,
            "maxChunkCharacterCount": 0,
            "maxGeneratedTokenCount": 0,
            "maxPrefillTokenCount": 0,
            "maxSegmentWallTimeSeconds": 0.0,
            "maxSegmentAudioDurationSeconds": 0.0,
            "maxSegmentPrefillTimeSeconds": 0.0,
            "maxSegmentDecodeTimeSeconds": 0.0,
            "maxAnchorSegmentDecodeTimeSeconds": 0.0,
            "maxContinuationSegmentDecodeTimeSeconds": 0.0,
            "maxSamplingTimeSeconds": 0.0,
            "maxEvalTimeSeconds": 0.0,
            "maxTokenMaterializationTimeSeconds": 0.0,
            "maxEmbeddingAssemblyTimeSeconds": 0.0,
            "maxTalkerForwardTimeSeconds": 0.0,
            "maxCodePredictorTimeSeconds": 0.0,
            "anchorSegmentsObserved": 0,
            "continuationSegmentsObserved": 0,
            "continuationOutlierCount": 0,
            "continuationOutlierSegments": [],
            "maxTokenCountsSeen": [],
            "executionModesSeen": [],
            "requestIDs": [],
        }

    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            runtime = payload.get("runtime") or {}
            max_active = max(max_active, int(runtime.get("activeSynthesisCount") or 0))
            max_stalled = max(max_stalled, int(runtime.get("stalledSynthesisCount") or 0))
            for request in runtime.get("activeSynthesisRequests") or []:
                if request.get("id"):
                    requests_seen.add(request["id"])
                if request.get("executionMode"):
                    execution_modes_seen.add(str(request["executionMode"]))
                max_segment_index = max(max_segment_index, int(request.get("segmentIndex") or 0))
                max_segment_count = max(max_segment_count, int(request.get("segmentCount") or 0))
                max_chunk_character_count = max(max_chunk_character_count, int(request.get("chunkCharacterCount") or 0))
                max_generated_token_count = max(max_generated_token_count, int(request.get("generatedTokenCount") or 0))
                max_prefill_token_count = max(max_prefill_token_count, int(request.get("prefillTokenCount") or 0))
                max_segment_wall_time_seconds = max(
                    max_segment_wall_time_seconds,
                    float(request.get("segmentWallTimeSeconds") or 0.0),
                )
                max_segment_audio_duration_seconds = max(
                    max_segment_audio_duration_seconds,
                    float(request.get("segmentAudioDurationSeconds") or 0.0),
                )
                max_segment_prefill_time_seconds = max(
                    max_segment_prefill_time_seconds,
                    float(request.get("segmentPrefillTimeSeconds") or 0.0),
                )
                max_segment_decode_time_seconds = max(
                    max_segment_decode_time_seconds,
                    float(request.get("segmentDecodeTimeSeconds") or 0.0),
                )
                max_anchor_segment_decode_time_seconds = max(
                    max_anchor_segment_decode_time_seconds,
                    float(request.get("anchorSegmentDecodeTimeSeconds") or 0.0),
                )
                max_continuation_segment_decode_time_seconds = max(
                    max_continuation_segment_decode_time_seconds,
                    float(request.get("continuationSegmentDecodeTimeSeconds") or 0.0),
                )
                max_sampling_time_seconds = max(
                    max_sampling_time_seconds,
                    float(request.get("samplingTimeSeconds") or 0.0),
                )
                max_eval_time_seconds = max(
                    max_eval_time_seconds,
                    float(request.get("evalTimeSeconds") or 0.0),
                )
                max_token_materialization_time_seconds = max(
                    max_token_materialization_time_seconds,
                    float(request.get("tokenMaterializationTimeSeconds") or 0.0),
                )
                max_embedding_assembly_time_seconds = max(
                    max_embedding_assembly_time_seconds,
                    float(request.get("embeddingAssemblyTimeSeconds") or 0.0),
                )
                max_talker_forward_time_seconds = max(
                    max_talker_forward_time_seconds,
                    float(request.get("talkerForwardTimeSeconds") or 0.0),
                )
                max_code_predictor_time_seconds = max(
                    max_code_predictor_time_seconds,
                    float(request.get("codePredictorTimeSeconds") or 0.0),
                )
                request_id = str(request.get("id") or "")
                segment_index = int(request.get("segmentIndex") or 0)
                if request.get("usesAnchorConditioning") is True and request_id:
                    anchor_segments_observed.add((request_id, segment_index))
                elif request.get("usesAnchorConditioning") is False and request_id:
                    continuation_segments_observed.add((request_id, segment_index))
                if request.get("continuationOutlier") is True and request_id:
                    continuation_outlier_segments.add((request_id, segment_index))
                if request.get("maxTokenCount") is not None:
                    max_token_counts_seen.add(int(request["maxTokenCount"]))

    return {
        "maxActiveSynthesisCount": max_active,
        "maxStalledSynthesisCount": max_stalled,
        "maxSegmentIndex": max_segment_index,
        "maxSegmentCount": max_segment_count,
        "maxChunkCharacterCount": max_chunk_character_count,
        "maxGeneratedTokenCount": max_generated_token_count,
        "maxPrefillTokenCount": max_prefill_token_count,
        "maxSegmentWallTimeSeconds": max_segment_wall_time_seconds,
        "maxSegmentAudioDurationSeconds": max_segment_audio_duration_seconds,
        "maxSegmentPrefillTimeSeconds": max_segment_prefill_time_seconds,
        "maxSegmentDecodeTimeSeconds": max_segment_decode_time_seconds,
        "maxAnchorSegmentDecodeTimeSeconds": max_anchor_segment_decode_time_seconds,
        "maxContinuationSegmentDecodeTimeSeconds": max_continuation_segment_decode_time_seconds,
        "maxSamplingTimeSeconds": max_sampling_time_seconds,
        "maxEvalTimeSeconds": max_eval_time_seconds,
        "maxTokenMaterializationTimeSeconds": max_token_materialization_time_seconds,
        "maxEmbeddingAssemblyTimeSeconds": max_embedding_assembly_time_seconds,
        "maxTalkerForwardTimeSeconds": max_talker_forward_time_seconds,
        "maxCodePredictorTimeSeconds": max_code_predictor_time_seconds,
        "anchorSegmentsObserved": len(anchor_segments_observed),
        "continuationSegmentsObserved": len(continuation_segments_observed),
        "continuationOutlierCount": len(continuation_outlier_segments),
        "continuationOutlierSegments": [
            {"requestID": request_id, "segmentIndex": segment_index}
            for request_id, segment_index in sorted(continuation_outlier_segments)
        ],
        "maxTokenCountsSeen": sorted(max_token_counts_seen),
        "executionModesSeen": sorted(execution_modes_seen),
        "requestIDs": sorted(requests_seen),
    }


def git_head(root_dir: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_machine_metadata(out_path: Path, root_dir: Path) -> None:
    payload = {
        "timestampUTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pythonVersion": platform.python_version(),
        "gitHead": git_head(root_dir),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def runtime_is_idle(runtime: dict | None) -> bool:
    if not runtime:
        return False
    if int(runtime.get("activeSynthesisCount") or 0) != 0:
        return False
    if runtime.get("activeSynthesisRequests"):
        return False
    for model in runtime.get("residentModels") or []:
        if int(model.get("activeSessionCount") or 0) != 0:
            return False
    return True


def wait_for_idle_runtime(daemon_url: str, timeout_seconds: float = DAEMON_IDLE_TIMEOUT_SECONDS) -> dict | None:
    deadline = time.time() + timeout_seconds
    runtime_url = daemon_url.rstrip("/") + "/runtime"
    last_runtime = None
    while time.time() < deadline:
        last_runtime = fetch_json(runtime_url)
        if runtime_is_idle(last_runtime):
            return last_runtime
        time.sleep(0.5)
    return last_runtime


def load_voice_index(cli_bin: Path) -> dict[str, dict]:
    if not cli_bin.exists():
        return {}
    try:
        raw = subprocess.check_output([str(cli_bin), "voices", "list", "--json"], text=True)
        payload = json.loads(raw)
        voices = payload.get("data", {}).get("voices", [])
    except Exception:
        return {}

    index: dict[str, dict] = {}
    for voice in voices:
        identifier = voice.get("id")
        label = voice.get("label")
        if identifier:
            index[identifier] = voice
        if label:
            index[label] = voice
    return index


def resolved_voice_records(voice_index: dict[str, dict]) -> list[dict]:
    records: dict[str, dict] = {}
    for voice in voice_index.values():
        identifier = voice.get("id")
        if identifier:
            records[identifier] = voice
    return sorted(
        records.values(),
        key=lambda voice: ((voice.get("label") or "").lower(), voice["id"].lower())
    )


def resolve_voice_record(
    voice_index: dict[str, dict],
    requested_identifier: str | None,
    required_kind: str,
) -> dict:
    voices = resolved_voice_records(voice_index)
    lowered = (requested_identifier or "").strip().lower()
    if lowered and lowered != "auto":
        for voice in voices:
            if voice.get("id", "").lower() == lowered or voice.get("label", "").lower() == lowered:
                return voice
        raise SystemExit(
            f"Unable to resolve voice '{requested_identifier}'. "
            f"Available saved voices: {', '.join(voice.get('label') or voice['id'] for voice in voices)}"
        )

    candidates = [voice for voice in voices if voice.get("voiceKind") == required_kind]
    if not candidates:
        raise SystemExit(
            f"No saved voice with voiceKind='{required_kind}' is available. "
            f"Create one first or pass the voice explicitly."
        )
    return candidates[0]


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


def run_cli_case(
    cli_bin: Path,
    model_id: str,
    voice: dict,
    behavior: str,
    case_name: str,
    prompt_text: str,
    run_dir: Path,
) -> dict:
    prefix = f"{case_name}-{behavior}-cli"
    output_path = run_dir / f"{prefix}.wav"
    stdout_path = run_dir / f"{prefix}.stdout.log"
    stderr_path = run_dir / f"{prefix}.stderr.log"
    cmd = [
        "/usr/bin/time",
        "-l",
        str(cli_bin),
        "speak",
        "--model",
        model_id,
        "--voice",
        voice["id"],
        "--voice-behavior",
        behavior,
        "--format",
        "wav",
        "--output",
        str(output_path),
        "--text",
        prompt_text,
    ]
    exit_code, elapsed = run_command(cmd, stdout_path=stdout_path, stderr_path=stderr_path)
    audio_seconds = wav_duration_seconds(output_path) if exit_code == 0 and output_path.exists() else 0.0
    return {
        "surface": "cli",
        "case": case_name,
        "voiceBehavior": behavior,
        "voice": voice["id"],
        "voiceLabel": voice.get("label"),
        "voiceKind": voice.get("voiceKind"),
        "modelID": model_id,
        "promptCharacters": len(prompt_text),
        "promptWords": word_count(prompt_text),
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(output_path),
        "outputBytes": output_path.stat().st_size if output_path.exists() else 0,
        "peakRSSKB": peak_rss_kb_from_time(stderr_path),
        "segmentCount": None,
        "completionReason": None,
        "maxActiveSynthesisCount": None,
        "maxStalledSynthesisCount": None,
        "maxSegmentIndex": None,
        "maxSegmentCount": None,
        "maxPrefillTokenCount": None,
        "maxSegmentPrefillTimeSeconds": None,
        "maxSegmentDecodeTimeSeconds": None,
        "maxAnchorSegmentDecodeTimeSeconds": None,
        "maxContinuationSegmentDecodeTimeSeconds": None,
        "maxTalkerForwardTimeSeconds": None,
        "maxCodePredictorTimeSeconds": None,
        "continuationOutlierCount": None,
        "continuationOutlierSegments": [],
        "exitCode": exit_code,
    }


def daemon_request_json(
    model_id: str,
    voice: str,
    behavior: str,
    prompt_text: str,
    response_format: str,
) -> bytes:
    return json.dumps(
        {
            "model": model_id,
            "input": prompt_text,
            "voice": voice,
            "voice_behavior": behavior,
            "response_format": response_format,
        }
    ).encode("utf-8")


def run_daemon_nonstream_case(
    daemon_url: str,
    model_id: str,
    voice: dict,
    behavior: str,
    case_name: str,
    prompt_text: str,
    run_dir: Path,
) -> dict:
    pre_runtime = wait_for_idle_runtime(daemon_url)
    prefix = f"{case_name}-{behavior}-daemon"
    output_path = run_dir / f"{prefix}.wav"
    trace_path = run_dir / f"{prefix}.runtime.jsonl"
    stop_event = threading.Event()
    sampler = threading.Thread(target=runtime_sampler, args=(daemon_url, trace_path, stop_event), daemon=True)
    sampler.start()
    request = urllib.request.Request(
        daemon_url.rstrip("/") + "/audio/speech",
        data=daemon_request_json(model_id, voice["id"], behavior, prompt_text, "wav"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
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
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        sampler.join(timeout=2)

    metrics = summarize_runtime_trace(trace_path)
    runtime_status = wait_for_idle_runtime(daemon_url) or fetch_json(daemon_url.rstrip("/") + "/runtime") or {}
    audio_seconds = wav_duration_seconds(output_path) if status == 200 and output_path.exists() else 0.0
    resident_active_session_count = sum(
        int(model.get("activeSessionCount") or 0)
        for model in runtime_status.get("residentModels") or []
    )
    result = {
        "surface": "daemon",
        "case": case_name,
        "voiceBehavior": behavior,
        "voice": voice["id"],
        "voiceLabel": voice.get("label"),
        "voiceKind": voice.get("voiceKind"),
        "modelID": model_id,
        "promptCharacters": len(prompt_text),
        "promptWords": word_count(prompt_text),
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(output_path),
        "outputBytes": output_path.stat().st_size if output_path.exists() else 0,
        "completionReason": runtime_status.get("lastSynthesisCompletionReason"),
        "segmentCount": metrics["maxSegmentCount"] or None,
        "maxActiveSynthesisCount": metrics["maxActiveSynthesisCount"],
        "maxStalledSynthesisCount": metrics["maxStalledSynthesisCount"],
        "maxSegmentIndex": metrics["maxSegmentIndex"] or None,
        "maxSegmentCount": metrics["maxSegmentCount"] or None,
        "maxChunkCharacterCount": metrics["maxChunkCharacterCount"] or None,
        "maxGeneratedTokenCount": metrics["maxGeneratedTokenCount"] or None,
        "maxPrefillTokenCount": metrics["maxPrefillTokenCount"] or None,
        "maxSegmentWallTimeSeconds": metrics["maxSegmentWallTimeSeconds"] or None,
        "maxSegmentAudioDurationSeconds": metrics["maxSegmentAudioDurationSeconds"] or None,
        "maxSegmentPrefillTimeSeconds": metrics["maxSegmentPrefillTimeSeconds"] or None,
        "maxSegmentDecodeTimeSeconds": metrics["maxSegmentDecodeTimeSeconds"] or None,
        "maxAnchorSegmentDecodeTimeSeconds": metrics["maxAnchorSegmentDecodeTimeSeconds"] or None,
        "maxContinuationSegmentDecodeTimeSeconds": metrics["maxContinuationSegmentDecodeTimeSeconds"] or None,
        "maxSamplingTimeSeconds": metrics["maxSamplingTimeSeconds"] or None,
        "maxEvalTimeSeconds": metrics["maxEvalTimeSeconds"] or None,
        "maxTokenMaterializationTimeSeconds": metrics["maxTokenMaterializationTimeSeconds"] or None,
        "maxEmbeddingAssemblyTimeSeconds": metrics["maxEmbeddingAssemblyTimeSeconds"] or None,
        "maxTalkerForwardTimeSeconds": metrics["maxTalkerForwardTimeSeconds"] or None,
        "maxCodePredictorTimeSeconds": metrics["maxCodePredictorTimeSeconds"] or None,
        "anchorSegmentsObserved": metrics["anchorSegmentsObserved"],
        "continuationSegmentsObserved": metrics["continuationSegmentsObserved"],
        "continuationOutlierCount": metrics["continuationOutlierCount"],
        "continuationOutlierSegments": metrics["continuationOutlierSegments"],
        "maxTokenCountsSeen": metrics["maxTokenCountsSeen"],
        "observedExecutionModes": metrics["executionModesSeen"],
        "httpStatus": status,
        "error": error_message,
        "preRunIdle": runtime_is_idle(pre_runtime),
        "postRunIdle": runtime_is_idle(runtime_status),
        "postRunActiveSynthesisCount": int(runtime_status.get("activeSynthesisCount") or 0),
        "postRunStalledSynthesisCount": int(runtime_status.get("stalledSynthesisCount") or 0),
        "postRunResidentActiveSessionCount": resident_active_session_count,
        "exitCode": 0 if status == 200 else 1,
    }
    return result


def parse_sse_event_stream(response) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    current_event = "message"
    data_lines: list[str] = []
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


def run_daemon_stream_case(
    daemon_url: str,
    model_id: str,
    voice: dict,
    behavior: str,
    case_name: str,
    prompt_text: str,
    run_dir: Path,
) -> dict:
    pre_runtime = wait_for_idle_runtime(daemon_url)
    prefix = f"{case_name}-{behavior}-stream"
    trace_path = run_dir / f"{prefix}.runtime.jsonl"
    raw_sse_path = run_dir / f"{prefix}.sse.log"
    pcm_path = run_dir / f"{prefix}.pcm_f32le"
    wav_path = run_dir / f"{prefix}.wav"
    stop_event = threading.Event()
    sampler = threading.Thread(target=runtime_sampler, args=(daemon_url, trace_path, stop_event), daemon=True)
    sampler.start()
    request = urllib.request.Request(
        daemon_url.rstrip("/") + "/audio/speech/stream",
        data=daemon_request_json(model_id, voice["id"], behavior, prompt_text, "pcm_f32le"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    status = None
    error_message = None
    pcm_chunks = bytearray()
    chunk_count = 0
    sample_rate = 24_000
    reported_total_chunks = None
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
                elif event_name == "complete":
                    event = json.loads(payload)
                    reported_total_chunks = event.get("totalChunks")
                    sample_rate = int(event.get("sampleRate") or sample_rate)
                elif event_name == "error":
                    error_message = payload
    except urllib.error.HTTPError as error:
        status = error.code
        error_message = error.read().decode("utf-8", errors="replace")
    except Exception as error:
        error_message = str(error)
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        sampler.join(timeout=2)

    pcm_path.write_bytes(bytes(pcm_chunks))
    total_samples = write_float_stream_wav(bytes(pcm_chunks), sample_rate, wav_path) if pcm_chunks else 0
    audio_seconds = 0.0 if sample_rate == 0 else total_samples / sample_rate
    metrics = summarize_runtime_trace(trace_path)
    runtime_status = wait_for_idle_runtime(daemon_url) or fetch_json(daemon_url.rstrip("/") + "/runtime") or {}
    observed_segment_count = max(
        int(metrics["maxSegmentCount"] or 0),
        int(reported_total_chunks or 0),
        int(chunk_count or 0),
    )
    resident_active_session_count = sum(
        int(model.get("activeSessionCount") or 0)
        for model in runtime_status.get("residentModels") or []
    )
    return {
        "surface": "stream",
        "case": case_name,
        "voiceBehavior": behavior,
        "voice": voice["id"],
        "voiceLabel": voice.get("label"),
        "voiceKind": voice.get("voiceKind"),
        "modelID": model_id,
        "promptCharacters": len(prompt_text),
        "promptWords": word_count(prompt_text),
        "elapsedSeconds": round(elapsed, 6),
        "audioSeconds": round(audio_seconds, 6),
        "rtf": round(elapsed / audio_seconds, 6) if audio_seconds > 0 else None,
        "outputPath": str(wav_path),
        "outputBytes": wav_path.stat().st_size if wav_path.exists() else 0,
        "completionReason": runtime_status.get("lastSynthesisCompletionReason"),
        "segmentCount": observed_segment_count or None,
        "maxActiveSynthesisCount": metrics["maxActiveSynthesisCount"],
        "maxStalledSynthesisCount": metrics["maxStalledSynthesisCount"],
        "maxSegmentIndex": metrics["maxSegmentIndex"] or None,
        "maxSegmentCount": observed_segment_count or None,
        "maxChunkCharacterCount": metrics["maxChunkCharacterCount"] or None,
        "maxGeneratedTokenCount": metrics["maxGeneratedTokenCount"] or None,
        "maxPrefillTokenCount": metrics["maxPrefillTokenCount"] or None,
        "maxSegmentWallTimeSeconds": metrics["maxSegmentWallTimeSeconds"] or None,
        "maxSegmentAudioDurationSeconds": metrics["maxSegmentAudioDurationSeconds"] or None,
        "maxSegmentPrefillTimeSeconds": metrics["maxSegmentPrefillTimeSeconds"] or None,
        "maxSegmentDecodeTimeSeconds": metrics["maxSegmentDecodeTimeSeconds"] or None,
        "maxAnchorSegmentDecodeTimeSeconds": metrics["maxAnchorSegmentDecodeTimeSeconds"] or None,
        "maxContinuationSegmentDecodeTimeSeconds": metrics["maxContinuationSegmentDecodeTimeSeconds"] or None,
        "maxSamplingTimeSeconds": metrics["maxSamplingTimeSeconds"] or None,
        "maxEvalTimeSeconds": metrics["maxEvalTimeSeconds"] or None,
        "maxTokenMaterializationTimeSeconds": metrics["maxTokenMaterializationTimeSeconds"] or None,
        "maxEmbeddingAssemblyTimeSeconds": metrics["maxEmbeddingAssemblyTimeSeconds"] or None,
        "maxTalkerForwardTimeSeconds": metrics["maxTalkerForwardTimeSeconds"] or None,
        "maxCodePredictorTimeSeconds": metrics["maxCodePredictorTimeSeconds"] or None,
        "anchorSegmentsObserved": metrics["anchorSegmentsObserved"],
        "continuationSegmentsObserved": metrics["continuationSegmentsObserved"],
        "continuationOutlierCount": metrics["continuationOutlierCount"],
        "continuationOutlierSegments": metrics["continuationOutlierSegments"],
        "maxTokenCountsSeen": metrics["maxTokenCountsSeen"],
        "observedExecutionModes": metrics["executionModesSeen"],
        "streamChunkCount": chunk_count,
        "reportedTotalChunks": reported_total_chunks,
        "sampleRate": sample_rate,
        "httpStatus": status,
        "error": error_message,
        "preRunIdle": runtime_is_idle(pre_runtime),
        "postRunIdle": runtime_is_idle(runtime_status),
        "postRunActiveSynthesisCount": int(runtime_status.get("activeSynthesisCount") or 0),
        "postRunStalledSynthesisCount": int(runtime_status.get("stalledSynthesisCount") or 0),
        "postRunResidentActiveSessionCount": resident_active_session_count,
        "exitCode": 0 if status == 200 and error_message is None else 1,
    }


def validate_result(result: dict) -> list[str]:
    issues: list[str] = []
    expectation = CASE_EXPECTATIONS.get((result["case"], result["voiceBehavior"]), {})

    if result.get("exitCode", 1) != 0:
        issues.append(f"{result['surface']} transport/process failed")

    min_audio_seconds = expectation.get("minAudioSeconds")
    audio_seconds = float(result.get("audioSeconds") or 0.0)
    if min_audio_seconds is not None and audio_seconds < float(min_audio_seconds):
        issues.append(f"audio duration {audio_seconds:.2f}s below required {float(min_audio_seconds):.2f}s")
    if result.get("httpStatus") == 200 and audio_seconds <= 0:
        issues.append("benchmark produced no audio")
    if result.get("httpStatus") == 200 and int(result.get("outputBytes") or 0) == 0:
        issues.append("benchmark produced an empty output artifact")

    if result["surface"] in {"daemon", "stream"}:
        if not result.get("preRunIdle", False):
            issues.append("daemon was not idle before the case started")
        if result.get("completionReason") != "completed":
            issues.append(f"daemon completion reason was '{result.get('completionReason')}'")
        if int(result.get("maxStalledSynthesisCount") or 0) != 0:
            issues.append("daemon reported stalled synthesis during the run")
        if not result.get("postRunIdle", False):
            issues.append("daemon did not return to idle after the run")
        if int(result.get("postRunActiveSynthesisCount") or 0) != 0:
            issues.append("daemon still reported active synthesis after the run")
        if int(result.get("postRunResidentActiveSessionCount") or 0) != 0:
            issues.append("daemon still reported resident active sessions after the run")
        if int(result.get("postRunStalledSynthesisCount") or 0) != 0:
            issues.append("daemon still reported stalled synthesis after the run")

        segment_count = int(result.get("segmentCount") or 0)
        execution_modes = set(result.get("observedExecutionModes") or [])
        expect_segmented = bool(expectation.get("segmented"))
        if expect_segmented:
            saw_segmented_mode = "segmentedContinuation" in execution_modes
            if not saw_segmented_mode and segment_count < 2:
                issues.append(f"expected segmented continuation, observed segmentCount={segment_count}")
            if execution_modes and not saw_segmented_mode:
                issues.append(f"expected segmentedContinuation execution mode, saw {sorted(execution_modes)}")
        else:
            if "segmentedContinuation" in execution_modes:
                issues.append("expected one-shot execution but saw segmentedContinuation")
        if result["surface"] == "stream":
            stream_chunk_count = int(result.get("streamChunkCount") or 0)
            reported_total_chunks = int(result.get("reportedTotalChunks") or 0)
            if stream_chunk_count == 0:
                issues.append("streaming run produced zero chunk events")
            if reported_total_chunks == 0:
                issues.append("streaming completion reported zero total chunks")

    return issues


def markdown_summary(results: list[dict]) -> str:
    lines = [
        "# Qwen Benchmark Summary",
        "",
        "| Case | Lane | Surface | Voice | Voice Kind | Elapsed | Audio | RTF | Prefill | Decode | Anchor | Cont. | Sample | Eval | Token | Embed | Talker | Predictor | Outliers | Exit | Completion | Validation |",
        "|------|------|---------|-------|------------|---------|-------|-----|---------|--------|--------|-------|--------|------|-------|-------|--------|-----------|----------|------|------------|------------|",
    ]
    for result in results:
        elapsed = f"{result.get('elapsedSeconds', 0):.3f}s"
        audio = f"{result.get('audioSeconds', 0):.3f}s"
        rtf_value = result.get("rtf")
        rtf = f"{rtf_value:.3f}x" if isinstance(rtf_value, (int, float)) else "-"
        prefill_seconds = result.get("maxSegmentPrefillTimeSeconds")
        decode_seconds = result.get("maxSegmentDecodeTimeSeconds")
        prefill = f"{prefill_seconds:.3f}s" if isinstance(prefill_seconds, (int, float)) else "-"
        decode = f"{decode_seconds:.3f}s" if isinstance(decode_seconds, (int, float)) else "-"
        anchor_decode_seconds = result.get("maxAnchorSegmentDecodeTimeSeconds")
        continuation_decode_seconds = result.get("maxContinuationSegmentDecodeTimeSeconds")
        sampling_seconds = result.get("maxSamplingTimeSeconds")
        eval_seconds = result.get("maxEvalTimeSeconds")
        token_materialization_seconds = result.get("maxTokenMaterializationTimeSeconds")
        embedding_assembly_seconds = result.get("maxEmbeddingAssemblyTimeSeconds")
        talker_forward_seconds = result.get("maxTalkerForwardTimeSeconds")
        code_predictor_seconds = result.get("maxCodePredictorTimeSeconds")
        anchor_decode = f"{anchor_decode_seconds:.3f}s" if isinstance(anchor_decode_seconds, (int, float)) else "-"
        continuation_decode = f"{continuation_decode_seconds:.3f}s" if isinstance(continuation_decode_seconds, (int, float)) else "-"
        sampling = f"{sampling_seconds:.3f}s" if isinstance(sampling_seconds, (int, float)) else "-"
        eval_time = f"{eval_seconds:.3f}s" if isinstance(eval_seconds, (int, float)) else "-"
        token_materialization = f"{token_materialization_seconds:.3f}s" if isinstance(token_materialization_seconds, (int, float)) else "-"
        embedding_assembly = f"{embedding_assembly_seconds:.3f}s" if isinstance(embedding_assembly_seconds, (int, float)) else "-"
        talker_forward = f"{talker_forward_seconds:.3f}s" if isinstance(talker_forward_seconds, (int, float)) else "-"
        code_predictor = f"{code_predictor_seconds:.3f}s" if isinstance(code_predictor_seconds, (int, float)) else "-"
        outliers = int(result.get("continuationOutlierCount") or 0)
        completion = result.get("completionReason") or "-"
        validation = "pass" if not result.get("validationIssues") else "; ".join(result["validationIssues"])
        lines.append(
            f"| {result['case']} | {result['voiceBehavior']} | {result['surface']} | "
            f"{result.get('voiceLabel') or result['voice']} | {result.get('voiceKind') or '-'} | {elapsed} | {audio} | {rtf} | {prefill} | {decode} | {anchor_decode} | {continuation_decode} | {sampling} | {eval_time} | {token_materialization} | {embedding_assembly} | {talker_forward} | {code_predictor} | {outliers} | "
            f"{result.get('exitCode', 1)} | {completion} | {validation} |"
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
    write_machine_metadata(run_dir / "machine.json", ROOT_DIR)

    cases = split_csv(args.cases)
    lanes = split_csv(args.lanes)
    surfaces = split_csv(args.surfaces)
    valid_cases = set(CASE_FILES.keys())
    valid_lanes = {"expressive", "stableNarrator"}
    valid_surfaces = {"cli", "daemon", "stream"}

    for case in cases:
        if case not in valid_cases:
            raise SystemExit(f"Unknown case '{case}'. Valid cases: {sorted(valid_cases)}")
    for lane in lanes:
        if lane not in valid_lanes:
            raise SystemExit(f"Unknown lane '{lane}'. Valid lanes: {sorted(valid_lanes)}")
    for surface in surfaces:
        if surface not in valid_surfaces:
            raise SystemExit(f"Unknown surface '{surface}'. Valid surfaces: {sorted(valid_surfaces)}")

    voice_for_lane = {
        "expressive": args.expressive_voice,
        "stableNarrator": args.stable_voice,
    }

    plan = {
        "runID": run_id,
        "artifactsDir": str(run_dir),
        "modelID": args.model,
        "daemonURL": daemon_url,
        "cliBin": str(cli_bin),
        "cases": cases,
        "lanes": lanes,
        "surfaces": surfaces,
        "voices": voice_for_lane,
        "gitHead": git_head(ROOT_DIR),
    }
    if not cli_bin.exists():
        raise SystemExit(f"CLI binary not found: {cli_bin}")

    voice_index = load_voice_index(cli_bin)
    resolved_voices = {
        "expressive": resolve_voice_record(voice_index, voice_for_lane["expressive"], "legacyPrompt"),
        "stableNarrator": resolve_voice_record(voice_index, voice_for_lane["stableNarrator"], "clonePrompt"),
    }
    plan["resolvedVoices"] = {
        lane: {
            "id": voice["id"],
            "label": voice.get("label"),
            "voiceKind": voice.get("voiceKind"),
        }
        for lane, voice in resolved_voices.items()
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    results: list[dict] = []
    failures = 0
    validation_failures = 0

    for case in cases:
        prompt_text = CASE_FILES[case].read_text(encoding="utf-8").strip()
        for lane in lanes:
            voice = resolved_voices[lane]
            for surface in surfaces:
                if surface == "cli":
                    result = run_cli_case(cli_bin, args.model, voice, lane, case, prompt_text, run_dir)
                elif surface == "daemon":
                    result = run_daemon_nonstream_case(daemon_url, args.model, voice, lane, case, prompt_text, run_dir)
                else:
                    result = run_daemon_stream_case(daemon_url, args.model, voice, lane, case, prompt_text, run_dir)
                result["gitHead"] = plan["gitHead"]
                result["validationIssues"] = [] if args.no_validate else validate_result(result)
                results.append(result)
                if result.get("exitCode", 1) != 0:
                    failures += 1
                if result["validationIssues"]:
                    validation_failures += 1

    summary = {
        "runID": run_id,
        "artifactsDir": str(run_dir),
        "gitHead": plan["gitHead"],
        "modelID": args.model,
        "daemonURL": daemon_url,
        "results": results,
        "failureCount": failures,
        "validationFailureCount": validation_failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown_summary(results), encoding="utf-8")

    print(f"Saved Qwen benchmark artifacts to {run_dir}")
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Summary Markdown: {run_dir / 'summary.md'}")
    return 1 if failures or validation_failures else 0


raise SystemExit(main())
PY
