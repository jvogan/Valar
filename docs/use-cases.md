# Use Cases

Valar is designed to work well as a local speech engine, not just as a demo app. These are the three public tracks the repo is optimized for.

## Developer: Local Speech API

Use the CLI and daemon when you want a local speech backend for scripts, apps, or experiments.

Typical path:

1. `make quickstart`
2. `make first-clip`
3. `make build-daemon`
4. `swift run --package-path apps/ValarDaemon valarttsd`
5. Call `/v1/health`, `/v1/models`, `/v1/runtime`, and `/v1/audio/speech`

Best starting model:

- `Soprano` for proof-of-life
- `Qwen Base` for the main narration lane

## Agent: MCP Automation

Use the bridge when you want a local speech tool for an MCP-capable agent or automation client.

Typical path:

1. Get the CLI path working first
2. Start `valarttsd`
3. `make bootstrap-bridge`
4. `cd bridge && bun server.ts`
5. Point your agent at the bridge

Best starting model:

- `Soprano` for the quickest end-to-end loop
- `Qwen` if the agent needs the main supported TTS or ASR lane

![Valar CLI and MCP preview](../assets/media/cli-mcp-preview.png)

## Creator: Narration And Voice Workflows

Use Valar when you want a local narration workflow with progressively richer voice features.

Typical path:

1. Start with `Soprano` to confirm the machine is healthy
2. Move to `Qwen Base` for longer-form narration
3. Add `Qwen VoiceDesign` when you want text-driven voice creation
4. Build the macOS app from Xcode if you want a desktop workflow

Best starting model:

- `Qwen Base` for the main supported narration path
- `VibeVoice` only if you specifically want preset voices and accept preview posture

![Valar app preview](../assets/media/app-preview.png)
