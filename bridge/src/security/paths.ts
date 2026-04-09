import { resolve, join, extname, dirname, basename } from "path";
import { statSync, lstatSync, realpathSync } from "fs";
import { homedir } from "os";
import { INBOX_DIR, OUTBOX_DIR } from "../storage.js";

const AUDIO_EXTENSIONS = new Set([
  ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac",
  ".aac", ".pcm", ".raw", ".webm", ".aiff", ".aif",
]);

const MAX_INPUT_BYTES = 50 * 1024 * 1024;

function resolveWithSymlinks(p: string): string {
  try {
    return realpathSync(p);
  } catch {
    try {
      const parent = realpathSync(dirname(p));
      const leaf = basename(p);
      const leafPath = join(parent, leaf);
      try {
        const stat = lstatSync(leafPath);
        if (stat.isSymbolicLink()) {
          return realpathSync(leafPath);
        }
      } catch {
      }
      return leafPath;
    } catch {
      return resolve(p);
    }
  }
}

function resolveAllowedDirs(): string[] {
  const home = homedir();
  const outbox = process.env.VALAR_OUTBOX;
  const outboxDir = outbox && outbox.length > 0 ? outbox : "/tmp/valar-outbox";

  const raw = [
    "/tmp",
    join(home, "Desktop"),
    join(home, "Downloads"),
    join(home, "Documents"),
    INBOX_DIR,
    OUTBOX_DIR,
    outboxDir,
  ];

  const resolved: string[] = [];
  for (const d of raw) {
    resolved.push(d + "/");
    try {
      const real = realpathSync(d);
      if (real !== d) resolved.push(real + "/");
    } catch {
    }
  }
  return resolved;
}

const OUTPUT_ALLOWED = resolveAllowedDirs();

function resolveInputAllowedDirs(): string[] {
  const home = homedir();
  const raw = [
    "/tmp",
    join(home, "Desktop"),
    join(home, "Downloads"),
    join(home, "Documents"),
    join(home, "Music"),
    INBOX_DIR,
    OUTBOX_DIR,
  ];
  const resolved: string[] = [];
  for (const d of raw) {
    resolved.push(d + "/");
    try {
      const real = realpathSync(d);
      if (real !== d) resolved.push(real + "/");
    } catch {}
  }
  return resolved;
}

const INPUT_ALLOWED = resolveInputAllowedDirs();

export function validateInputPath(p: string): void {
  const resolved = resolveWithSymlinks(p);
  const ext = extname(resolved).toLowerCase();
  if (!AUDIO_EXTENSIONS.has(ext)) {
    throw new Error(
      `Input path must be an audio file (${[...AUDIO_EXTENSIONS].join(", ")}). Got extension: "${ext}"`,
    );
  }
  let size: number;
  try {
    size = statSync(resolved).size;
  } catch {
    throw new Error(`Cannot stat input file: ${p}`);
  }
  if (size > MAX_INPUT_BYTES) {
    throw new Error(
      `Input file exceeds 50 MB limit (${(size / 1024 / 1024).toFixed(1)} MB). Got: ${p}`,
    );
  }
  const resolvedWithSlash = resolved.endsWith("/") ? resolved : resolved + "/";
  if (!INPUT_ALLOWED.some((dir) => resolvedWithSlash.startsWith(dir))) {
    throw new Error(
      "Input path must be within Desktop, Downloads, Documents, Music, /tmp, or Valar bridge storage.",
    );
  }
}

export function validateOutputPath(p: string): void {
  const resolved = resolveWithSymlinks(p);
  const ext = extname(resolved).toLowerCase();
  if (!AUDIO_EXTENSIONS.has(ext)) {
    throw new Error(
      `Output path must be an audio file (${[...AUDIO_EXTENSIONS].join(", ")}). Got extension: "${ext}"`,
    );
  }
  const resolvedWithSlash = resolved.endsWith("/") ? resolved : resolved + "/";
  if (!OUTPUT_ALLOWED.some((dir) => resolvedWithSlash.startsWith(dir))) {
    throw new Error(
      `Output path must be within an allowed directory (Desktop, Downloads, Documents, /tmp, or Valar bridge storage). Got: ${p}`,
    );
  }
}
