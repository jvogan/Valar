#!/usr/bin/env python3
"""Convert Voxtral preset voice embeddings from raw `.pt` files to safe `.bin`.

This script is intentionally the only place in the repo that loads the official
pickle-backed `voice_embedding/*.pt` files. Runtime code should consume only the
generated `voice_embedding_safe/*.bin` files and `voice_embedding_safe/index.json`.

Example isolated environment:

    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r scripts/voxtral/requirements.txt
    python scripts/voxtral/convert_voice_embeddings.py /path/to/Voxtral-4B-TTS-2603

The input may be either:
- the model root containing `voice_embedding/` and `params.json`, or
- the `voice_embedding/` directory itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path
from typing import Any

import torch


PRESET_VOICE_IDS: dict[str, int] = {
    "casual_female": 0,
    "casual_male": 1,
    "cheerful_female": 2,
    "neutral_female": 3,
    "neutral_male": 4,
    "pt_male": 5,
    "pt_female": 6,
    "nl_male": 7,
    "nl_female": 8,
    "it_male": 9,
    "it_female": 10,
    "fr_male": 11,
    "fr_female": 12,
    "es_male": 13,
    "es_female": 14,
    "de_male": 15,
    "de_female": 16,
    "ar_male": 17,
    "hi_male": 18,
    "hi_female": 19,
}

VOICE_ID_TO_NAME = {str(voice_id): voice_name for voice_name, voice_id in PRESET_VOICE_IDS.items()}
ALLOWED_DTYPES = {torch.float32, torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Voxtral voice_embedding/*.pt files into safe raw float32 .bin files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Model root containing voice_embedding/ or the voice_embedding/ directory itself.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for safe files. Defaults to <model_root>/voice_embedding_safe.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional config JSON used to resolve the expected embedding dimension.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="Optional expected embedding dimension. Overrides config autodiscovery.",
    )
    return parser.parse_args()


def resolve_voice_embedding_dir(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidate = input_path / "voice_embedding"
    if candidate.is_dir():
        return candidate
    if input_path.is_dir() and input_path.name == "voice_embedding":
        return input_path
    raise FileNotFoundError(
        f"Expected a voice_embedding directory at {candidate} or passed directly as input."
    )


def resolve_model_root(input_path: Path, voice_embedding_dir: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path == voice_embedding_dir:
        return voice_embedding_dir.parent
    return input_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_expected_dim_from_config(config: Any) -> int | None:
    if isinstance(config, dict):
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            hidden_size = text_config.get("hidden_size")
            if isinstance(hidden_size, int):
                return hidden_size

        hidden_size = config.get("hidden_size")
        if isinstance(hidden_size, int):
            return hidden_size

        dim = config.get("dim")
        if isinstance(dim, int):
            return dim

    return None


def resolve_expected_dim(args: argparse.Namespace, model_root: Path) -> int:
    if args.expected_dim is not None:
        if args.expected_dim <= 0:
            raise ValueError(f"--expected-dim must be positive, got {args.expected_dim}")
        return args.expected_dim

    config_candidates: list[Path] = []
    if args.model_config is not None:
        config_candidates.append(args.model_config.expanduser().resolve())
    else:
        config_candidates.extend(
            [
                model_root / "params.json",
                model_root / "config.json",
            ]
        )

    for candidate in config_candidates:
        if not candidate.is_file():
            continue
        dim = extract_expected_dim_from_config(load_json(candidate))
        if dim is not None:
            return dim

    searched = ", ".join(str(path) for path in config_candidates) or "<none>"
    raise ValueError(
        "Unable to resolve expected embedding dimension from model config. "
        f"Searched: {searched}. Pass --expected-dim explicitly if needed."
    )


def canonical_voice_name(stem: str) -> str:
    if stem in PRESET_VOICE_IDS:
        return stem
    if stem in VOICE_ID_TO_NAME:
        return VOICE_ID_TO_NAME[stem]
    raise ValueError(
        f"Unexpected voice embedding file '{stem}.pt'. "
        "Expected one of the 20 preset voice names or numeric speaker IDs 0-19."
    )


def collect_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for item in value.values():
            tensors.extend(collect_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(collect_tensors(item))
        return tensors
    return []


def load_embedding_tensor(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raise RuntimeError(
            "This script requires a torch build that supports torch.load(..., weights_only=True). "
            "Install a current torch version in the isolated venv from scripts/voxtral/requirements.txt."
        ) from None

    tensors = collect_tensors(payload)
    if len(tensors) != 1:
        raise ValueError(
            f"Expected exactly one tensor in {path.name}, found {len(tensors)}."
        )

    tensor = tensors[0].detach().cpu()
    if tensor.dtype not in ALLOWED_DTYPES:
        raise ValueError(
            f"{path.name} has unsupported dtype {tensor.dtype}; "
            "expected float32 or bfloat16."
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"{path.name} has invalid shape {list(tensor.shape)}; expected [N, dim]."
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(
            f"{path.name} has invalid non-positive shape {list(tensor.shape)}."
        )

    return tensor


def write_float32_bin(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.to(torch.float32).contiguous().reshape(-1).tolist()
    packed = array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    with path.open("wb") as handle:
        packed.tofile(handle)


def assert_no_raw_pt_files(output_dir: Path) -> None:
    raw_pt_files = sorted(output_dir.glob("*.pt"))
    if raw_pt_files:
        names = ", ".join(path.name for path in raw_pt_files)
        raise ValueError(
            f"Output directory must not contain raw .pt files, found: {names}"
        )


def main() -> int:
    args = parse_args()
    voice_embedding_dir = resolve_voice_embedding_dir(args.input)
    model_root = resolve_model_root(args.input, voice_embedding_dir)
    expected_dim = resolve_expected_dim(args, model_root)
    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (model_root / "voice_embedding_safe").resolve()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    assert_no_raw_pt_files(output_dir)

    pt_files = sorted(voice_embedding_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {voice_embedding_dir}")

    manifest_entries: list[dict[str, Any]] = []
    seen_voices: set[str] = set()

    for pt_file in pt_files:
        voice_name = canonical_voice_name(pt_file.stem)
        if voice_name in seen_voices:
            raise ValueError(
                f"Duplicate inputs resolved to voice '{voice_name}'. "
                "Keep only one source file per preset voice."
            )

        tensor = load_embedding_tensor(pt_file)
        frame_count, dim = tensor.shape
        if dim != expected_dim:
            raise ValueError(
                f"{pt_file.name} has embedding dim {dim}; expected {expected_dim}."
            )

        output_file = output_dir / f"{voice_name}.bin"
        write_float32_bin(output_file, tensor)

        expected_bytes = frame_count * dim * 4
        actual_bytes = output_file.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{output_file.name} wrote {actual_bytes} bytes; expected {expected_bytes}."
            )

        manifest_entries.append(
            {
                "voice_name": voice_name,
                "file": output_file.name,
                "dtype": "float32",
                "shape": [frame_count, dim],
                "frame_count": frame_count,
            }
        )
        seen_voices.add(voice_name)

    missing_voices = [voice_name for voice_name in PRESET_VOICE_IDS if voice_name not in seen_voices]
    if missing_voices:
        raise ValueError(
            "Missing preset voice embeddings: " + ", ".join(missing_voices)
        )

    manifest_entries.sort(key=lambda entry: PRESET_VOICE_IDS[entry["voice_name"]])
    manifest = {
        "format": "voxtral_voice_embedding_safe/v1",
        "expected_dim": expected_dim,
        "voice_count": len(manifest_entries),
        "voices": manifest_entries,
    }

    manifest_path = output_dir / "index.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    assert_no_raw_pt_files(output_dir)

    print(f"Converted {len(manifest_entries)} preset voice embeddings.")
    print(f"Input: {voice_embedding_dir}")
    print(f"Output: {output_dir}")
    print(f"Expected embedding dim: {expected_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
