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

## Documentation

See [docs/integrations.md](../docs/integrations.md) for the full integration guide.
