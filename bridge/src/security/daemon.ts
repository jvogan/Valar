import { sanitizeMessage } from "./redaction.js";

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_TEXT_LIMIT_BYTES = 512 * 1024;
const DEFAULT_BINARY_LIMIT_BYTES = 200 * 1024 * 1024;

export function requireLoopbackDaemonURL(raw: string): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`VALAR_DAEMON_URL is not a valid URL: ${sanitizeMessage(raw)}`);
  }

  const loopbackHosts = new Set(["127.0.0.1", "::1", "[::1]", "localhost"]);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("VALAR_DAEMON_URL must use http: or https:.");
  }
  if (parsed.username || parsed.password) {
    throw new Error("VALAR_DAEMON_URL must not include credentials.");
  }
  if (!loopbackHosts.has(parsed.hostname)) {
    throw new Error(
      `VALAR_DAEMON_URL must point to a loopback address (127.0.0.1, ::1, or localhost). Got: ${parsed.hostname}`,
    );
  }

  return parsed.origin;
}

export function createDaemonURLBuilder(baseURL: string): (path: string) => string {
  return (path: string) => `${baseURL}/v1${path}`;
}

export async function daemonFetch(
  url: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const signal = AbortSignal.timeout(timeoutMs);
  return fetch(url, { ...init, signal });
}

async function readLimitedBytes(response: Response, limitBytes: number): Promise<Uint8Array> {
  const reader = response.body?.getReader();
  if (!reader) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > limitBytes) {
      await reader.cancel();
      throw new Error(`Daemon response exceeded ${(limitBytes / 1024 / 1024).toFixed(1)} MB limit.`);
    }
    chunks.push(value);
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

export async function readDaemonText(
  response: Response,
  limitBytes = DEFAULT_TEXT_LIMIT_BYTES,
): Promise<string> {
  const bytes = await readLimitedBytes(response, limitBytes);
  return new TextDecoder().decode(bytes);
}

export async function readDaemonJSON<T = unknown>(
  response: Response,
  limitBytes = DEFAULT_TEXT_LIMIT_BYTES,
): Promise<T> {
  const text = await readDaemonText(response, limitBytes);
  return JSON.parse(text) as T;
}

export async function readDaemonBinary(
  response: Response,
  limitBytes = DEFAULT_BINARY_LIMIT_BYTES,
): Promise<ArrayBuffer> {
  const bytes = await readLimitedBytes(response, limitBytes);
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return copy;
}
