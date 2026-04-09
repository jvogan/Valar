import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { readFile } from "fs/promises";
import { basename } from "path";
import { validateInputPath } from "../security/paths.js";

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
    "valar_transcribe",
    "Transcribe an audio file to text using the Valar ASR engine.",
    {
      file_path: z
        .string()
        .describe("Absolute path to the audio file to transcribe."),
      model: z
        .string()
        .optional()
        .describe("ASR model ID to use. Omit to use the daemon default."),
      language: z
        .string()
        .optional()
        .describe('Language code (for example \"en\"). Omit to auto-detect.'),
      response_format: z
        .enum(["text", "json", "verbose_json", "srt", "vtt"])
        .optional()
        .describe('Response format. Defaults to "text".'),
    },
    async ({ file_path, model, language, response_format }) => {
      try {
        validateInputPath(file_path);
      } catch (e) {
        return err(String(e));
      }
      let fileData: Buffer;
      try {
        fileData = await readFile(file_path);
      } catch (e) {
        return err(`Cannot read file "${file_path}": ${e}`);
      }

      const filename = basename(file_path);
      const form = new FormData();
      form.append("file", new Blob([fileData]), filename);
      if (model) form.append("model", model);
      if (language) form.append("language", language);
      if (response_format) form.append("response_format", response_format);

      let res: Response;
      try {
        res = await fetch(daemonURL("/audio/transcriptions"), {
          method: "POST",
          body: form,
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Transcription failed (${res.status}): ${text}`);
      }

      const fmt = response_format ?? "text";
      if (fmt === "text" || fmt === "srt" || fmt === "vtt") {
        const text = await res.text();
        return ok(text);
      }
      const data = await res.json();
      return ok(JSON.stringify(data, null, 2));
    },
  );
}
