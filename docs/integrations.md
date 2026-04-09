# Integrations

Valar keeps the public integration story simple:

- **Core integration**: a local MCP bridge in `bridge/`
- **Advanced channel adapters**: intentionally deferred from the public v1 export until they are generalized away from private operator assumptions

The beginner path does not require any integration layer. Start with the CLI and daemon first.

## Core MCP Bridge

The Bun bridge in `bridge/` exposes the local daemon as MCP tools. It is the primary public integration surface for agents and local automation.

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

Bridge-safe I/O directories default to `~/Library/Application Support/Valar/bridge-storage`. The public bridge does not archive channel identifiers, sender metadata, or transcription/reply sidecars by default. Override the storage root with `VALARTTS_BRIDGE_STORAGE_ROOT` if you need a different local location.

## Advanced Channel Adapters

Public `Valar` v1 keeps the bridge surface generic. Channel-specific adapters are not part of the exported newcomer path yet.

If you need Discord, Telegram, or other channel delivery:

- start from the MCP bridge first
- keep channel configuration entirely environment-driven
- treat any channel adapter as a separate advanced add-on, not part of the core quickstart
- do not assume personal state directories, pair/allowlist files, or workstation-specific service wrappers

## Public Safety Boundary

Public docs should teach:

- CLI + daemon first
- MCP bridge second
- channel adapters only after the core local stack is already working
