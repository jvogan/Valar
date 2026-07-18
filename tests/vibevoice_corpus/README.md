# VibeVoice Sample Corpus

Multilingual sample prompts for `mlx-community/VibeVoice-Realtime-0.5B-4bit`.

## Coverage

The bundled voice pack contains eleven language presets. These files provide short and
medium prompts for repeatable local smoke tests across those presets.

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

## Cases

Each language file contains two lines:

- **Line 1 (short)**: one sentence of roughly 12 words.
- **Line 2 (medium)**: one longer sentence of roughly 25 words.

## File layout

See [manifest.json](./manifest.json) for the full language-to-voice mapping.
