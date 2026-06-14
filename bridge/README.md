# Valar MCP Bridge

Bun-based MCP bridge that exposes the local Valar daemon as MCP tools for agents and automation workflows.

## Prerequisites

- [Bun](https://bun.sh/) 1.2 or later
- A running local `valarttsd` daemon

## Quick Start

```bash
# From the repo root
make bootstrap-bridge
make start-daemon  # in another terminal
cd bridge && bun server.ts
```

Point your MCP-capable client at the bridge's stdio server.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VALAR_DAEMON_URL` | `http://127.0.0.1:8787` | Daemon address |
| `VALARTTS_BRIDGE_STORAGE_ROOT` | `~/Library/Application Support/Valar/bridge-storage` | Local storage path |

The bridge expects the daemon to stay on a loopback address. Use `valar_status` or `health_check` before asking an agent workflow to synthesize, transcribe, align, or mutate project state.

## Public Tools

The bridge exposes:

- readiness and runtime: `valar_status`, `health_check`
- model management: `valar_models`, `valar_install_model`, `remove_model`, `purge_model_cache`, `get_operation_status`
- voice library: `valar_voices`, `valar_create_voice`, `valar_design_voice`, `valar_clone_voice_from_file`, `valar_delete_voice`
- speech workflows: `valar_speak`, `valar_transcribe`, `valar_align`
- project sessions: `create_session`, `list_chapters`, `add_chapter`, `update_chapter`, `save_session`, `close_session`

Keep channel-specific delivery, bot tokens, and hosted service credentials outside this repo. The public bridge is a local stdio MCP surface over the loopback daemon.

## Documentation

See [docs/integrations.md](../docs/integrations.md) for the full integration guide.
