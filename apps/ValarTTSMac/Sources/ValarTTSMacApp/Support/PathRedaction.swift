import Foundation

/// Replaces the current user's home directory prefix with `~` in filesystem
/// paths so that the username is not exposed in the UI.
enum PathRedaction {
    private static let homeDirectory: String = NSHomeDirectory()

    /// Redact a POSIX path string (e.g. `~/Library/...` after redaction).
    static func redact(_ path: String) -> String {
        guard !homeDirectory.isEmpty, path.hasPrefix(homeDirectory) else {
            return path
        }
        let suffix = path.dropFirst(homeDirectory.count)
        return "~" + suffix
    }

    /// Redact a `file://` URL string while leaving other schemes untouched.
    static func redactURL(_ urlString: String) -> String {
        guard urlString.hasPrefix("file://") else { return urlString }
        // file:// URLs encode the path after the scheme+authority.
        // Convert to URL, redact the path, and reconstruct.
        guard let url = URL(string: urlString) else { return urlString }
        let redactedPath = redact(url.path)
        return "file://" + redactedPath
    }

    /// Expand a leading `~` back to the real home directory (for round-trip editing).
    static func expand(_ path: String) -> String {
        guard path.hasPrefix("~") else { return path }
        return homeDirectory + path.dropFirst(1)
    }
}
