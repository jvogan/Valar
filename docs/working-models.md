# Working Models

This page lists the working model families that should be easy for newcomers and agents to install and use right away.

The v1 working set is intentionally narrow:

- `Soprano` for the easiest first run
- `Qwen` for the main TTS, ASR, and alignment lane
- `VibeVoice` for compatibility-preview preset voices
- `Voxtral` for explicit non-commercial opt-in

## Working Models

Only the exact IDs below are part of the main public onboarding path.

| Family | Exact install ID | Support status | Download posture | Rough local footprint | License | Best use |
| --- | --- | --- | --- | --- | --- | --- |
| Soprano 1.1 80M | `mlx-community/Soprano-1.1-80M-bf16` | Supported | Recommended first install | About `285 MB` | Apache 2.0 | Fastest first run and starter demo lane |
| Qwen3-TTS 1.7B Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Supported | Optional install | About `4.2 GB` | Qwen License Agreement | Main narrator lane and stable long-form speech |
| Qwen3-TTS 1.7B CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | Supported | Optional install | About `4.2 GB` | Qwen License Agreement | Named speakers and saved voices |
| Qwen3-TTS 1.7B VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | Supported | Optional install | About `4.2 GB` | Qwen License Agreement | Text-described voice creation |
| Qwen3-ASR 0.6B | `mlx-community/Qwen3-ASR-0.6B-8bit` | Supported | Optional install | About `1.0 GB` | Qwen License Agreement | Speech recognition and transcription |
| Qwen3-ForcedAligner 0.6B | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | Supported | Optional install | About `1.3 GB` | Qwen License Agreement | Word-level timestamps and alignment |
| VibeVoice Realtime 0.5B (4-bit) | `mlx-community/VibeVoice-Realtime-0.5B-4bit` | Preview | Preview-only optional install | About `700 MB` | MIT | Fast preset-voice TTS, English-first |
| Voxtral 4B-TTS 2603 (MLX 4-bit) | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | Preview | Explicit non-commercial opt-in | About `2.4 GB` | CC BY-NC 4.0 | Non-commercial multilingual preset-voice lane |

## Public API Fields

The CLI and daemon expose these public compatibility fields directly:

- `supportTier`: `supported` or `preview`
- `distributionTier`: `bundledFirstRun`, `optionalInstall`, or `compatibilityPreview`
- `releaseEligible`: whether the model is part of the default release-facing path
- `qualityTierByLanguage`: language-specific quality posture, especially important for `VibeVoice`

## Compatibility Notes

- Footprint values are rough public planning numbers for local installs on Apple Silicon and can drift as upstream packs change.
- `Soprano` is the shortest path to a working first clip.
- In this source-first repo, `bundledFirstRun` means “recommended first install” rather than “weights are checked into git.”
- The table above translates those enum-style API fields into plain-language download guidance.
- `Qwen` is the main supported product lane for high-quality speech, transcription, and alignment.
- `VibeVoice` is preset-only and should be described as preview, not parity.
- `VibeVoice` language contract is English-supported, `de/es/fr/it/ja/ko/nl/pl/pt` preview, and Hindi hidden from release-facing defaults.
- `VibeVoice` reuses the upstream `Qwen/Qwen2.5-0.5B` tokenizer; `valartts models install` materializes that companion automatically when needed.
- `Voxtral` is hidden by default unless the user intentionally opts in to non-commercial models with `VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1`.

## License Boundary

- Repo code is MIT.
- Model weights keep their own upstream licenses.
- Non-commercial models stay opt-in and should not be presented as default downloads.
