# Long-Form Project Cookbook

This is the automation-friendly path for turning a manuscript into organized local speech output.
The examples below assume `valartts` is installed on `PATH`; from a source checkout,
replace it with `swift run --package-path apps/ValarCLI valartts`.

## Import

```bash
valartts projects import manuscript.md \
  --name "Manuscript Draft" \
  --split-mode markdown-headings \
  --speaker Narrator
```

Other split modes are `paragraphs`, `lines`, `dialogue`, and `whole-document`.
Dialogue mode accepts simple forms:

```text
[S1] We should leave now.
[S2|style:calm] Not before the signal.
Narrator: The room went quiet.
```

Lint the project before rendering:

```bash
valartts projects lint
valartts projects lint --path Manuscript.valarproject
valartts projects lint --model <model-id>
```

The linter builds a voice bible from speaker labels, style and language attributes,
dialogue markers, colon speaker lines, and common nonverbal tags. It warns when the
selected model may ignore expressive tags, drift across a cast, or struggle with long
segments.

## Render

```bash
valartts renders queue
valartts renders start
valartts renders status --watch
```

For long work:

- Keep one active project session.
- Save after import and after render batches.
- Poll JSON status instead of scraping formatted text.
- Resume by reopening the `.valarproject` and running `renders start` again.
- Keep exports until checksums and durations have been inspected.

## Export

```bash
valartts exports list
valartts exports create --chapter <chapter-uuid>
valartts projects export-pack
```

`projects export-pack` writes `Exports/PublishPack/valar-export-pack.json` by default.
The manifest includes chapter hashes, voice-bible entries, recorded export artifacts,
byte counts, checksums when available, and notes for missing chapter audio. Keep the
manifest with the rendered output when handing a project to another tool or comparing
regenerated output.

## Voice Consistency

Current render paths preserve speaker labels in project state, but model-specific voice
conditioning is not yet consumed by every render path. For more consistent results:

- Use one narrator model for a full project.
- Prefer a saved stable narrator or clone-prompt voice where available.
- Keep temperature and top-p stable across chapters.
- Regenerate nearby segments together when a voice drifts.
- Record the model ID, voice ID, language, and generation options with the project.
