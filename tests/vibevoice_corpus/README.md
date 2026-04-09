# VibeVoice Benchmark Corpus

Golden corpus for `mlx-community/VibeVoice-Realtime-0.5B-4bit`.

## Coverage

The bundled voice pack contains eleven language presets. The live quality-gated benchmark
enforces acceptance on English plus nine multilingual languages, while the bundled Hindi
preset remains advisory-only until it clears the live suite consistently.

| Language   | Code | Voice preset      |
|------------|------|-------------------|
| English    | en   | en-Emma_woman     |
| German     | de   | de-Spk1_woman     |
| French     | fr   | fr-Spk1_woman     |
| Spanish    | es   | sp-Spk0_woman     |
| Italian    | it   | it-Spk0_woman     |
| Dutch      | nl   | nl-Spk1_woman     |
| Portuguese | pt   | pt-Spk0_woman     |
| Polish     | pl   | pl-Spk1_woman     |
| Hindi      | hi   | in-Samuel_man     |
| Japanese   | ja   | jp-Spk1_woman     |
| Korean     | ko   | kr-Spk0_woman     |

Quality-gated benchmark languages: `en`, `de`, `fr`, `es`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`.

Advisory-only similarity language: `hi`.

## Cases

Each language file contains two lines:

- **Line 1 (short)**: one sentence, ~12 words — used for **first-chunk latency** measurement.
- **Line 2 (medium)**: one longer sentence, ~25 words — used for **steady-state RTF** measurement.

## Benchmark targets (Apple Silicon, warm model)

| Metric                  | Target  | Notes                            |
|-------------------------|---------|----------------------------------|
| First-chunk latency     | < 500 ms | Time to first audio byte on warm start |
| Short RTF               | < 1.5 ×  | synth_time / audio_duration      |
| Medium RTF              | < 1.5 ×  | synth_time / audio_duration      |
| English transcript similarity | >= 0.90 | Local Qwen ASR round-trip |
| Multilingual transcript similarity (short) | >= 0.60 | Quality-gated non-English short cases |
| Multilingual transcript similarity (medium) | >= 0.50 | Quality-gated non-English medium cases |
| Daemon vs CLI similarity gap (short) | <= 0.10 | `daemon` may trail `cli` by at most 0.10 |
| Stream vs daemon similarity gap (short) | <= 0.10 | `stream` may trail `daemon` by at most 0.10 |
| Surface similarity gap (medium) | <= 0.15 | Longer prompts allow more ASR round-trip variance |

VibeVoice is designed for near-realtime streaming (~300 ms first chunk per upstream claims).
The < 500 ms target is conservative to account for daemon round-trip overhead.

## Running benchmarks

```bash
bash scripts/vibevoice/benchmark.sh
```

See [scripts/vibevoice/benchmark.sh](../../scripts/vibevoice/benchmark.sh) for options.

For the full live validation + benchmark sweep with SSD-backed retention cleanup, use:

```bash
bash scripts/vibevoice/run_suite.sh
```

To preview or apply artifact cleanup without rerunning the suite:

```bash
bash scripts/vibevoice/run_suite.sh --prune-only --dry-run
bash scripts/vibevoice/run_suite.sh --prune-only
```

## File layout

See [manifest.json](./manifest.json) for the full language-to-voice mapping.
