import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { daemonUnavailableMessage, sanitizeMessage } from "../security/redaction.js";

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
    "valar_delete_voice",
    "Delete a saved voice from the Valar voice library. This removes the voice record and any retained reference-audio or saved-conditioning assets. Preset voices cannot be deleted.",
    {
      voice_id: z
        .string()
        .uuid()
        .describe("UUID of the saved voice to delete."),
    },
    async ({ voice_id }) => {
      let res: Response;
      try {
        res = await fetch(daemonURL(`/voices/${voice_id}`), {
          method: "DELETE",
        });
      } catch {
        return err(daemonUnavailableMessage());
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Voice deletion failed (${res.status}): ${text}`);
      }

      return ok(`Deleted saved voice ${voice_id}.`);
    },
  );
}
