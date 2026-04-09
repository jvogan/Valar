import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

function ok(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}

export function register(
  server: McpServer,
  daemonURL: (path: string) => string,
): void {
  server.tool(
    "valar_voices",
    "List all available voices (built-in and saved) in the Valar voice library, including their IDs and labels.",
    {},
    async () => {
      try {
        const res = await fetch(daemonURL("/voices"));
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          return err(`Failed to list voices (${res.status}): ${text}`);
        }
        const voices = await res.json();
        return ok(JSON.stringify(voices, null, 2));
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }
    },
  );
}
