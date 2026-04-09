import { homedir } from "os";

const HOME = homedir();

export function redactPath(path: string): string {
  return sanitizeMessage(path);
}

export function sanitizeMessage(text: string): string {
  let sanitized = text;

  if (HOME) {
    sanitized = sanitized.split(HOME).join("~");
  }

  sanitized = sanitized
    .replace(/file:\/\/\/Users\/[^/\s:]+/g, "file://~")
    .replace(/\/Users\/[^/\s:]+/g, "~")
    .replace(/\/Volumes\/[^/\s:]+/g, "/Volumes/<volume>");

  return sanitized;
}

export function daemonUnavailableMessage(): string {
  return "Daemon unreachable. Start local valarttsd and retry.";
}
