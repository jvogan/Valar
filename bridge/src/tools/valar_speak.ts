import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { readFile, writeFile } from "fs/promises";
import { extname } from "path";
import { resolveProfile } from "./profiles.js";
import { validateInputPath, validateOutputPath } from "../security/paths.js";

function ok(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function err(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}

async function encodeReferenceAudioDataURL(filePath: string): Promise<string> {
  validateInputPath(filePath);
  const bytes = await readFile(filePath);
  const ext = extname(filePath).toLowerCase();
  const mime =
    ext === ".m4a" ? "audio/mp4" :
    ext === ".wav" ? "audio/wav" :
    "application/octet-stream";
  return `data:${mime};base64,${bytes.toString("base64")}`;
}

export function register(
  server: McpServer,
  daemonURL: (path: string) => string,
): void {
  server.tool(
    "valar_speak",
    "Synthesize speech from text using Valar. Saves the audio to output_path and returns the path. " +
      "Pass a profile (e.g. 'default-fast', 'default-quality') to apply " +
      "a pre-configured voice/model/format bundle; explicit voice/model/format args override the profile. " +
      "Use Qwen for designed voices and stable narrator voices, and use Voxtral only when you have intentionally opted into non-commercial preset voices.",
    {
      text: z.string().min(1).describe("The text to synthesize into speech."),
      output_path: z
        .string()
        .describe("Absolute path where the output audio file will be written."),
      profile: z
        .string()
        .optional()
        .describe(
          "Voice profile ID (e.g. 'default-fast', 'default-quality'). " +
            "Sets voice, model, and format from the profile; explicit args take precedence.",
        ),
      voice: z
        .string()
        .optional()
        .describe(
          "Voice ID or label to use. Overrides profile. For local Voxtral, pass a preset name or alias only; custom/reference-audio Voxtral voices are not available locally yet.",
        ),
      model: z
        .string()
        .optional()
        .describe(
          "Model ID to use. Overrides profile. Omit to use the profile or daemon default.",
        ),
      voice_behavior: z
        .enum(["auto", "expressive", "stableNarrator"])
        .optional()
        .describe(
          "Optional Qwen voice behavior override. Use 'expressive' for VoiceDesign or legacy prompt voices, and 'stableNarrator' for Base clone-prompt narration.",
        ),
      format: z
        .enum(["wav", "ogg_opus"])
        .optional()
        .describe('Output audio format. Overrides profile. Defaults to "wav".'),
      temperature: z.number().min(0).max(2).optional()
        .describe("Sampling temperature (0.0-2.0). Lower = more deterministic. Default: model default."),
      top_p: z.number().gt(0).max(1).optional()
        .describe("Top-p nucleus sampling (0.0-1.0, exclusive of 0). Default: model default."),
      repetition_penalty: z.number().min(1).max(2).optional()
        .describe("Repetition penalty factor (1.0-2.0). Default: model default."),
      max_tokens: z.number().int().min(1).max(8192).optional()
        .describe("Maximum speech tokens to generate. Default: model default."),
      reference_audio_path: z
        .string()
        .optional()
        .describe(
          "Absolute path to a WAV or M4A reference clip for Qwen-compatible cloning or stable narrator creation. Local Voxtral does not use this path.",
        ),
      reference_transcript: z
        .string()
        .optional()
        .describe(
          "Transcript spoken in the reference clip. Recommended for all cloning and stable narrator creation flows.",
        ),
      language: z
        .string()
        .optional()
        .describe(
          "Optional language hint such as 'en', 'fr', or 'ja'. Helps multilingual reference-audio flows resolve correctly.",
        ),
    },
    async ({ text, output_path, profile, voice, model, voice_behavior, format, temperature, top_p, repetition_penalty, max_tokens, reference_audio_path, reference_transcript, language }) => {
      let resolvedVoice = voice;
      let resolvedModel = model;
      let resolvedFormat = format;
      let resolvedVoiceBehavior: "auto" | "expressive" | "stableNarrator" | undefined = voice_behavior;

      if (profile) {
        const prof = await resolveProfile(profile);
        if (prof) {
          if (!resolvedVoice && prof.voiceId) resolvedVoice = prof.voiceId;
          if (!resolvedModel && prof.model) resolvedModel = prof.model;
          if (!resolvedFormat && prof.format) resolvedFormat = prof.format;
          if (!resolvedVoiceBehavior && prof.voiceBehavior) resolvedVoiceBehavior = prof.voiceBehavior;
        }
      }

      const body: Record<string, unknown> = { input: text };
      if (resolvedVoice) body.voice = resolvedVoice;
      if (resolvedModel) body.model = resolvedModel;
      if (resolvedFormat) body.response_format = resolvedFormat;
      if (resolvedVoiceBehavior) body.voice_behavior = resolvedVoiceBehavior;
      if (temperature !== undefined) body.temperature = temperature;
      if (top_p !== undefined) body.top_p = top_p;
      if (repetition_penalty !== undefined) body.repetition_penalty = repetition_penalty;
      if (max_tokens !== undefined) body.max_tokens = max_tokens;
      if (reference_transcript) body.reference_transcript = reference_transcript;
      if (language) body.language = language;
      if (reference_audio_path) {
        try {
          body.reference_audio = await encodeReferenceAudioDataURL(reference_audio_path);
        } catch (e) {
          return err(`Failed to read reference audio: ${e}`);
        }
      }

      let res: Response;
      try {
        res = await fetch(daemonURL("/audio/speech"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
      } catch (e) {
        return err(`Daemon unreachable: ${e}`);
      }

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        return err(`Speech synthesis failed (${res.status}): ${text}`);
      }

      try {
        const buffer = await res.arrayBuffer();
        validateOutputPath(output_path);
        await writeFile(output_path, Buffer.from(buffer));
        return ok(
          `Audio written to ${output_path} (${buffer.byteLength} bytes)`,
        );
      } catch (e) {
        return err(`Failed to write audio file: ${e}`);
      }
    },
  );
}
