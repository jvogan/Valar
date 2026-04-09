#!/usr/bin/env bun
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import * as valarModels from "./src/tools/valar_models.js";
import * as valarVoices from "./src/tools/valar_voices.js";
import * as valarCreateVoice from "./src/tools/valar_create_voice.js";
import * as valarDesignVoice from "./src/tools/valar_design_voice.js";
import * as valarCloneVoiceFromFile from "./src/tools/valar_clone_voice_from_file.js";
import * as valarDeleteVoice from "./src/tools/valar_delete_voice.js";
import * as valarSpeak from "./src/tools/valar_speak.js";
import * as valarTranscribe from "./src/tools/valar_transcribe.js";
import * as valarAlign from "./src/tools/valar_align.js";
import * as valarInstallModel from "./src/tools/valar_install_model.js";

function requireLoopbackDaemonURL(raw: string): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`VALAR_DAEMON_URL is not a valid URL: ${raw}`);
  }
  const loopbackHosts = new Set(["127.0.0.1", "::1", "localhost"]);
  if (!loopbackHosts.has(parsed.hostname)) {
    throw new Error(
      `VALAR_DAEMON_URL must point to a loopback address (127.0.0.1, ::1, or localhost). Got: ${parsed.hostname}`,
    );
  }
  return raw;
}

const DAEMON_URL = requireLoopbackDaemonURL(process.env.VALAR_DAEMON_URL ?? "http://127.0.0.1:8787");

function daemonURL(path: string): string {
  return `${DAEMON_URL}/v1${path}`;
}

async function daemonGet(path: string): Promise<unknown> {
  const res = await fetch(daemonURL(path));
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Daemon GET ${path} → ${res.status}: ${text}`);
  }
  return res.json();
}

async function daemonPost(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(daemonURL(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Daemon POST ${path} → ${res.status}: ${text}`);
  }
  return res.json();
}

function ok(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}

async function checkDaemonHealthy(): Promise<boolean> {
  try {
    const res = await fetch(daemonURL("/health"), {
      signal: AbortSignal.timeout(5_000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

const server = new McpServer({
  name: "valartts-bridge",
  version: "1.0.0",
});

valarModels.register(server, daemonURL);
valarVoices.register(server, daemonURL);
valarSpeak.register(server, daemonURL);
valarTranscribe.register(server, daemonURL);
valarAlign.register(server, daemonURL);
valarCreateVoice.register(server, daemonURL);
valarDesignVoice.register(server, daemonURL);
valarCloneVoiceFromFile.register(server, daemonURL);
valarDeleteVoice.register(server, daemonURL);
valarInstallModel.register(server, daemonURL);

server.tool(
  "valar_status",
  "Get a capability snapshot for this machine: whether the daemon is reachable vs ready, which operations are ready, which models are installed, and what prerequisites are missing.",
  {},
  async () => {
    try {
      const snapshot = await daemonGet("/capabilities");
      return ok(JSON.stringify(snapshot, null, 2));
    } catch (e) {
      return err(`Daemon unreachable or capabilities endpoint failed: ${e}`);
    }
  },
);

server.tool(
  "health_check",
  "Check whether the Valar daemon is healthy and inspect capability plus runtime state. Prefer this over a bare readiness probe when deciding whether local TTS, ASR, or voice-cloning workflows can run right now.",
  {},
  async () => {
    try {
      const [health, capabilities, runtime] = await Promise.all([
        daemonGet("/health"),
        daemonGet("/capabilities"),
        daemonGet("/runtime").catch(() => null),
      ]);
      return ok(JSON.stringify({ health, capabilities, runtime }, null, 2));
    } catch (e) {
      return err(`Daemon unreachable: ${e}`);
    }
  },
);

server.tool(
  "remove_model",
  "Remove an installed Valar model pack from the Valar daemon. Shared Hugging Face cache entries, if any, are not purged by this operation.",
  {
    model: z.string().describe("Model ID to remove."),
  },
  async ({ model }) => {
    try {
      const result = await daemonPost("/models/remove", { model });
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "purge_model_cache",
  "Remove shared Hugging Face cache entries for one model while keeping any installed Valar pack. Use this to reclaim disk after installs or to prune stale shared-cache baggage intentionally.",
  {
    model: z.string().describe("Model ID whose shared HF cache entries should be removed."),
  },
  async ({ model }) => {
    try {
      const result = await daemonPost("/models/purge-cache", { model });
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "get_operation_status",
  "Get the status of an async operation (e.g. model install) by its operation ID.",
  {
    operation_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("The operation ID returned by install_model."),
  },
  async ({ operation_id }) => {
    try {
      const result = await daemonGet(`/operations/${operation_id}`);
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "create_session",
  "Create or open a Valar project session. Returns a session ID for subsequent chapter operations.",
  {
    path: z.string().describe("Absolute path to the .valarproject bundle to open or create."),
  },
  async ({ path }) => {
    try {
      const result = await daemonPost("/sessions/new", { path });
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "list_chapters",
  "List all chapters in a Valar project session.",
  {
    session_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Session ID returned by create_session."),
  },
  async ({ session_id }) => {
    try {
      const result = await daemonGet(`/sessions/${session_id}/chapters`);
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "add_chapter",
  "Add a new chapter to a Valar project session.",
  {
    session_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Session ID returned by create_session."),
    title: z.string().describe("Chapter title."),
    text: z.string().describe("Chapter script / body text."),
    index: z.number().int().optional().describe("Insertion index. Negative or omitted appends after the last chapter."),
    speaker_label: z.string().optional().describe("Optional speaker label for multi-speaker projects."),
  },
  async ({ session_id, title, text, index, speaker_label }) => {
    try {
      const body: Record<string, unknown> = {
        title,
        text,
        index: index ?? -1,
      };
      if (speaker_label) body.speakerLabel = speaker_label;

      const result = await daemonPost(`/sessions/${session_id}/chapters`, body);
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "update_chapter",
  "Update an existing chapter's title or text in a Valar project session.",
  {
    session_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Session ID returned by create_session."),
    chapter_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Chapter UUID from list_chapters."),
    title: z.string().optional().describe("New title (omit to keep current)."),
    text: z.string().optional().describe("New script text (omit to keep current)."),
  },
  async ({ session_id, chapter_id, title, text }) => {
    try {
      const body: Record<string, string> = {};
      if (title !== undefined) body.title = title;
      if (text !== undefined) body.text = text;

      const res = await fetch(daemonURL(`/sessions/${session_id}/chapters/${chapter_id}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const t = await res.text().catch(() => "");
        return err(`Chapter update failed (${res.status}): ${t}`);
      }

      const result = await res.json();
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "save_session",
  "Persist a Valar project session's current state to its bundle file on disk.",
  {
    session_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Session ID returned by create_session."),
  },
  async ({ session_id }) => {
    try {
      const result = await daemonPost(`/sessions/${session_id}/save`, {});
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

server.tool(
  "close_session",
  "Close a Valar project session and release its resources in the daemon.",
  {
    session_id: z.string().regex(/^[a-zA-Z0-9_-]+$/).describe("Session ID returned by create_session."),
  },
  async ({ session_id }) => {
    try {
      const result = await daemonPost(`/sessions/${session_id}/close`, {});
      return ok(JSON.stringify(result, null, 2));
    } catch (e) {
      return err(String(e));
    }
  },
);

async function main(): Promise<void> {
  const healthy = await checkDaemonHealthy();
  if (healthy) {
    process.stderr.write(`[valartts-bridge] Daemon is healthy at ${DAEMON_URL}\n`);
  } else {
    process.stderr.write(
      `[valartts-bridge] Warning: daemon not healthy or not reachable at ${DAEMON_URL}. Tools will return errors until the daemon starts.\n`,
    );
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((e) => {
  process.stderr.write(`[valartts-bridge] Fatal: ${e}\n`);
  process.exit(1);
});
