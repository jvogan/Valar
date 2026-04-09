import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

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
    "valar_design_voice",
    "Design a new saved voice from a text description. Qwen is the primary local voice-design family; the resulting saved voice can be reused with compatible Qwen speech models.",
    {
      name: z
        .string()
        .min(1)
        .describe("Display name for the designed voice."),
      description: z
        .string()
        .min(1)
        .describe(
          "Text description of the voice style, tone, and characteristics (e.g. 'warm, mid-30s female narrator with a slight British accent and measured pace').",
        ),
    },
    async ({ name, description }) => {
      let res: Response;
      try {
        res = await fetch(daemonURL("/voices/design"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, description }),
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Voice design failed (${res.status}): ${text}`);
      }

      const voice = await res.json();
      return ok(JSON.stringify(voice, null, 2));
    },
  );
}
