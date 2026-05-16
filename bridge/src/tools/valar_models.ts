import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { daemonUnavailableMessage, sanitizeMessage } from "../security/redaction.js";
import { daemonFetch, readDaemonJSON, readDaemonText } from "../security/daemon.js";

function ok(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function err(text: string) {
  return { content: [{ type: "text" as const, text: sanitizeMessage(text) }], isError: true };
}

export function register(
  server: McpServer,
  daemonURL: (path: string) => string,
): void {
  server.tool(
    "valar_models",
    "List all TTS and ASR models known to the Valar daemon with full metadata, including the canonical `installState` (`supported`, `cached`, or `installed`).",
    {},
    async () => {
      try {
        const res = await daemonFetch(daemonURL("/models"));
        if (!res.ok) {
          const text = await readDaemonText(res).catch(() => "");
          return err(`Failed to list models (${res.status}): ${text}`);
        }
        const models = await readDaemonJSON(res);
        return ok(JSON.stringify(models, null, 2));
      } catch {
        return err(daemonUnavailableMessage());
      }
    },
  );
}
