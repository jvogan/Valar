import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { readFile } from "fs/promises";
import { basename } from "path";
import { validateInputPath } from "../security/paths.js";
import { daemonUnavailableMessage, redactPath, sanitizeMessage } from "../security/redaction.js";

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
    "valar_align",
    "Forced-align an audio file against a known transcript using the Valar alignment engine. Returns word- and segment-level timestamps as JSON.",
    {
      file_path: z
        .string()
        .describe("Absolute path to the audio file to align."),
      transcript: z
        .string()
        .min(1)
        .describe("The known transcript text to align against the audio."),
      model: z
        .string()
        .optional()
        .describe("Alignment model ID to use. Omit to use the daemon default."),
      language: z
        .string()
        .optional()
        .describe('Optional language hint such as "en" or "ja".'),
    },
    async ({ file_path, transcript, model, language }) => {
      try {
        validateInputPath(file_path);
      } catch (e) {
        return err(String(e));
      }
      let fileData: Buffer;
      try {
        fileData = await readFile(file_path);
      } catch (e) {
        return err(`Cannot read file "${redactPath(file_path)}": ${e}`);
      }

      const filename = basename(file_path);
      const form = new FormData();
      form.append("file", new Blob([fileData]), filename);
      form.append("transcript", transcript);
      if (model) form.append("model", model);
      if (language) form.append("language", language);

      let res: Response;
      try {
        res = await fetch(daemonURL("/alignments"), {
          method: "POST",
          body: form,
        });
      } catch {
        return err(daemonUnavailableMessage());
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Alignment failed (${res.status}): ${text}`);
      }

      const data = await res.json();
      return ok(JSON.stringify(data, null, 2));
    },
  );
}
