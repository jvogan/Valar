/**
 * storage.ts - safe local directories for bridge input/output media.
 *
 * The public bridge uses these directories only as an allowlisted root for
 * local audio files. It does not archive channel metadata or transcript sidecars.
 */

import { mkdir } from "node:fs/promises";
import { join } from "node:path";

const DEFAULT_STORAGE_ROOT = process.env.HOME
  ? join(process.env.HOME, "Library/Application Support/Valar/bridge-storage")
  : "/tmp/valartts-bridge-storage";

export const BRIDGE_STORAGE_ROOT =
  process.env.VALARTTS_BRIDGE_STORAGE_ROOT?.trim() || DEFAULT_STORAGE_ROOT;
export const INBOX_DIR = join(BRIDGE_STORAGE_ROOT, "incoming-audio");
export const OUTBOX_DIR = join(BRIDGE_STORAGE_ROOT, "generated-audio");

export async function ensureStorageDirs(): Promise<void> {
  await Promise.all([
    mkdir(INBOX_DIR, { recursive: true }),
    mkdir(OUTBOX_DIR, { recursive: true }),
  ]);
}
