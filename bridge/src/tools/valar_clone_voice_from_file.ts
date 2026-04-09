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
    "valar_clone_voice_from_file",
    "Create a saved voice from a reference audio file. The file must be WAV or M4A, between 5 and 30 seconds, sampled at 16 kHz or higher. transcript is required because the daemon uses it for saved-voice conditioning. Pass model only when you need to force a specific clone-capable lane explicitly; otherwise the daemon uses its default clone lane.",
    {
      file_path: z
        .string()
        .describe(
          "Absolute path to the reference audio file (WAV or M4A, 5–30 seconds, at least 16 kHz).",
        ),
      name: z
        .string()
        .min(1)
        .describe("Display name for the cloned voice."),
      transcript: z
        .string()
        .describe(
          "Reference transcript: the text spoken in the audio clip. This is required for saved voice creation.",
        ),
      model: z
        .string()
        .optional()
        .describe(
          "Optional TTS model ID to use for cloning. Omit this to use the daemon's default clone-capable lane.",
        ),
    },
    async ({ file_path, name, transcript, model }) => {
      try {
        validateInputPath(file_path);
      } catch (e) {
        return err(String(e));
      }
      let fileData: Buffer;
      try {
        fileData = await readFile(file_path);
      } catch (e) {
        return err(`Failed to read audio file: ${e}`);
      }

      const filename = basename(file_path);
      const form = new FormData();
      form.append("file", new Blob([fileData]), filename);
      form.append("name", name);
      form.append("transcript", transcript);
      if (model) form.append("model", model);

      let res: Response;
      try {
        res = await fetch(daemonURL("/voices/clone"), {
          method: "POST",
          body: form,
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Voice cloning failed (${res.status}): ${text}`);
      }

      const voice = await res.json();
      return ok(JSON.stringify(voice, null, 2));
    },
  );
}
