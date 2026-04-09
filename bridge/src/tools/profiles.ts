import { readFile } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

export interface ChannelPrefs {
  format?: "wav" | "ogg_opus";
}

export interface AgentProfile {
  voiceId: string | null;
  model: string | null;
  format: "wav" | "ogg_opus" | null;
  voiceBehavior?: "auto" | "expressive" | "stableNarrator";
  channelPrefs: ChannelPrefs;
}

interface ProfilesFile {
  profiles: Record<string, AgentProfile>;
}

const BUNDLED_CONFIG_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../config/profiles.json",
);

const APP_SUPPORT_CONFIG_PATH = process.env.HOME
  ? join(
      process.env.HOME,
      "Library/Application Support/Valar/bridge-config/profiles.json",
    )
  : null;

const OVERRIDE_CONFIG_PATH =
  process.env.VALARTTS_BRIDGE_PROFILE_PATH?.trim() || null;

function profileSearchPaths(): string[] {
  return Array.from(
    new Set(
      [OVERRIDE_CONFIG_PATH, APP_SUPPORT_CONFIG_PATH, BUNDLED_CONFIG_PATH].filter(
        (path): path is string => Boolean(path),
      ),
    ),
  );
}

let cachedProfiles: Record<string, AgentProfile> | null = null;

async function loadProfiles(): Promise<Record<string, AgentProfile>> {
  if (cachedProfiles !== null) return cachedProfiles;

  for (const path of profileSearchPaths()) {
    try {
      const raw = await readFile(path, "utf8");
      const parsed = JSON.parse(raw) as ProfilesFile;
      if (parsed?.profiles && typeof parsed.profiles === "object") {
        cachedProfiles = parsed.profiles;
        return cachedProfiles;
      }
    } catch {
      // Not available — try next path.
    }
  }

  cachedProfiles = {};
  return cachedProfiles;
}

export async function resolveProfile(
  profileId: string,
): Promise<AgentProfile | null> {
  const profiles = await loadProfiles();
  return profiles[profileId] ?? null;
}
