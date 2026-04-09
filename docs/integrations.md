# Integrations

Valar keeps the public integration story simple:

- **Default path**: CLI first
- **Local API path**: daemon second
- **Agent path**: MCP bridge third

The beginner path does not require any integration layer. Get the CLI and daemon working first.

## MCP Bridge

The Bun bridge in `bridge/` exposes the local daemon as MCP tools. It is the main public integration surface for local agents and automation.

Bridge prerequisites:

- Bun 1.2 or newer
- a running local `valarttsd`
- optional `VALAR_DAEMON_URL` override if you do not use the default loopback daemon address

Typical flow:

1. Start `valarttsd`
2. `make bootstrap-bridge`
3. `cd bridge && bun server.ts`
4. Point your MCP-capable agent or client at that stdio server

The bridge assumes the daemon is available on `http://127.0.0.1:8787` unless `VALAR_DAEMON_URL` is set explicitly to another loopback address.

Bridge storage defaults to `~/Library/Application Support/Valar/bridge-storage`. Override it with `VALARTTS_BRIDGE_STORAGE_ROOT` if you want a different local directory.

## When To Use Which Surface

- Use the **CLI** for the fastest first success and simple scripts
- Use the **daemon** when you want a local HTTP API
- Use the **MCP bridge** when you want an agent or tool-calling workflow
- Use the **app** when the CLI path already works and you want a desktop UI

## Advanced Add-Ons

Channel-specific delivery adapters are intentionally outside the main public quickstart. Start from the MCP bridge first, then layer any additional automation on top of that local foundation.
