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
| Soprano 1.1 80M | `mlx-community/Soprano-1.1-80M-bf16` | Supported | Recommended first install | About `285 MB` | Apache-2.0 | Smallest first run and starter demo lane |
| Qwen3-TTS 1.7B Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Supported | Optional install | About `4.2 GB` | Apache-2.0 | Main narrator lane and stable long-form speech |
| Qwen3-TTS 1.7B CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | Supported | Optional install | About `4.2 GB` | Apache-2.0 | Named speakers and saved voices |
| Qwen3-TTS 1.7B VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | Supported | Optional install | About `4.2 GB` | Apache-2.0 | Text-described voice creation |
| Qwen3-ASR 0.6B | `mlx-community/Qwen3-ASR-0.6B-8bit` | Supported | Optional install | About `1.0 GB` | Apache-2.0 | Speech recognition and transcription |
| Qwen3-ForcedAligner 0.6B | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | Supported | Optional install | About `1.3 GB` | Apache-2.0 | Word-level timestamps and alignment |
| VibeVoice Realtime 0.5B (4-bit) | `mlx-community/VibeVoice-Realtime-0.5B-4bit` | Preview | Preview-only optional install | About `700 MB` | MIT | Fast preset-voice TTS, English-first |
| Voxtral 4B-TTS 2603 (MLX 4-bit) | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | Preview | Explicit non-commercial opt-in | About `2.4 GB` | CC BY-NC 4.0 | Non-commercial multilingual preset-voice lane |

## System Backups

These IDs are built into macOS and require no model download. They are fallback surfaces, not the recommended public model lanes.

| Surface | Model ID | Support status | Download posture | License | Best use |
| --- | --- | --- | --- | --- | --- |
| Apple System TTS | `apple/system-tts` | Supported backup | Built in, no download | macOS system service | Emergency/default speech when no MLX TTS model is installed |
| Apple System ASR | `apple/system-asr` | Supported backup | Built in, no download | macOS system service | App-hosted local transcription when macOS Speech permission is available |

`apple/system-asr` uses on-device macOS Speech recognition. Raw CLI and daemon binaries may not be able to request Speech permission because they do not carry an app Info.plist with `NSSpeechRecognitionUsageDescription`; install `Qwen ASR` for the default CLI and daemon transcription path.

## Public API Fields

The CLI and daemon expose these public compatibility fields directly:

- `supportTier`: `supported` or `preview`
- `distributionTier`: `bundledFirstRun`, `optionalInstall`, or `compatibilityPreview`
- `releaseEligible`: whether the model is part of the default release-facing path
- `qualityTierByLanguage`: language-specific quality posture, especially important for `VibeVoice`

## Compatibility Notes

- Footprint values are rough public planning numbers for local installs on Apple Silicon and can drift as upstream packs change.
- The public docs do not make universal latency or real-time-factor claims. Measure on your own hardware with your own text and installed model set before relying on throughput numbers.
- `Soprano` is the shortest path to a working first clip.
- Apple System TTS and ASR are backup surfaces. They are useful for graceful degradation, but do not replace the Soprano/Qwen onboarding recommendations.
- In this source-first repo, `bundledFirstRun` means “recommended first install” rather than “weights are checked into git.”
- The table above translates those enum-style API fields into plain-language download guidance.
- `Qwen` is the main supported product lane for high-quality speech, transcription, and alignment.
- `Qwen Base` is the stable narrator lane. `Qwen VoiceDesign` is the text-described expressive voice lane. `Qwen CustomVoice` is the official named-speaker lane.
- `VibeVoice` is preset-only and should be described as preview, not parity.
- `VibeVoice` language contract is English-supported, `de/es/fr/it/ja/ko/nl/pl/pt` preview, and Hindi hidden from release-facing defaults.
- `VibeVoice` reuses the upstream `Qwen/Qwen2.5-0.5B` tokenizer; `valartts models install` materializes that companion automatically when needed.
- `VibeVoice` does not support local cloning, voice design, or reference-audio conditioning.
- `Voxtral` is hidden by default unless the user intentionally opts in to non-commercial models with `VALARTTS_ENABLE_NONCOMMERCIAL_MODELS=1`.
- `Voxtral` is preset-only in Valar's local public surface; do not present it as a saved-voice or reference-audio lane.

## Public License Summary

| Family | Public posture |
| --- | --- |
| `Soprano` | Apache-2.0; recommended first install. |
| `Qwen` | Apache-2.0; check the exact upstream model card before deployment. |
| `VibeVoice` | MIT model license in the public catalog; optional compatibility preview. |
| `Voxtral` | CC BY-NC 4.0; hidden unless explicitly enabled for non-commercial use. |

## License Boundary

- Repo code is MIT.
- Model weights keep their own upstream licenses.
- Non-commercial models stay opt-in and should not be presented as default downloads.
