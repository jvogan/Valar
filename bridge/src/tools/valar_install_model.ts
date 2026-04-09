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
    "valar_install_model",
    "Install a TTS or ASR model into the Valar daemon. Set allow_download=true for first-time remote installs. Set refresh_cache=true to purge the shared Hugging Face cache for this model before reinstalling.",
    {
      model_id: z
        .string()
        .describe(
          "Model ID to install (for example `mlx-community/Soprano-1.1-80M-bf16`).",
        ),
      allow_download: z
        .boolean()
        .optional()
        .describe(
          "Allow the daemon to download model artifacts if they are not already cached locally. Defaults to true.",
        ),
      refresh_cache: z
        .boolean()
        .optional()
        .describe(
          "Purge the shared HF cache for this model before reinstalling. Requires allow_download=true for remote models.",
        ),
    },
    async ({ model_id, allow_download, refresh_cache }) => {
      const body: Record<string, unknown> = {
        model: model_id,
        allow_download: allow_download ?? true,
      };
      if (refresh_cache !== undefined) body.refresh_cache = refresh_cache;

      let res: Response;
      try {
        res = await fetch(daemonURL("/models/install"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (res.status === 409) {
        const json = await res.json().catch(() => ({})) as Record<string, unknown>;
        return err(
          `Model '${model_id}' is not cached locally and requires a network download.\n` +
            `Retry with allow_download=true, or use the CLI instead:\n` +
            `  valartts models install ${model_id} --allow-download\n` +
            `Daemon response: ${JSON.stringify(json, null, 2)}`,
        );
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Model installation failed (${res.status}): ${text}`);
      }

      const result = await res.json();
      return ok(JSON.stringify(result, null, 2));
    },
  );
}
