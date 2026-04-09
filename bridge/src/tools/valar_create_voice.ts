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
    "valar_create_voice",
    "Create a placeholder voice record in the Valar voice library. For designing a voice from a text description, use valar_design_voice instead. Use this only for manually managed placeholder records that need a human-authored description; use valar_clone_voice_from_file for saved-voice creation from reference audio.",
    {
      name: z
        .string()
        .min(1)
        .describe("Display name for the new voice."),
      description: z
        .string()
        .min(1)
        .describe(
          "Required voice prompt: a natural-language description of the intended voice style, tone, and characteristics for this metadata-only record.",
        ),
    },
    async ({ name, description }) => {
      const body: Record<string, string> = { name, description };

      let res: Response;
      try {
        res = await fetch(daemonURL("/voices/create"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Voice creation failed (${res.status}): ${text}`);
      }

      const voice = await res.json();
      return ok(JSON.stringify(voice, null, 2));
    },
  );
}
