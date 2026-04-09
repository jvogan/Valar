import Foundation

public enum ValarPathRedaction {
    private static let homeDirectory = FileManager.default.homeDirectoryForCurrentUser
        .standardizedFileURL
        .path

    public static func sanitizeMessage(_ value: String) -> String {
        guard value.isEmpty == false else {
            return value
        }

        var sanitized = value
        if homeDirectory.isEmpty == false {
            sanitized = sanitized.replacingOccurrences(of: homeDirectory, with: "~")
        }

        let replacements: [(pattern: String, template: String)] = [
            (#"file:///Users/[^/\s:]+"#, "file://~"),
            (#"/Users/[^/\s:]+"#, "~"),
            (#"/Volumes/[^/\s:]+"#, "/Volumes/<volume>"),
        ]

        for replacement in replacements {
            sanitized = replacingMatches(
                in: sanitized,
                pattern: replacement.pattern,
                template: replacement.template
            )
        }

        return sanitized
    }

    public static func redact(_ path: String) -> String {
        let standardizedPath = URL(fileURLWithPath: path).standardizedFileURL.path
        guard homeDirectory.isEmpty == false else {
            return redactVolumeName(in: standardizedPath)
        }

        if standardizedPath == homeDirectory {
            return "~"
        }
        if standardizedPath.hasPrefix(homeDirectory + "/") {
            return "~" + standardizedPath.dropFirst(homeDirectory.count)
        }
        return redactVolumeName(in: standardizedPath)
    }

    public static func redact(_ url: URL) -> String {
        redact(url.path)
    }

    public static func redactURLString(_ value: String) -> String {
        guard value.hasPrefix("file://"), let url = URL(string: value) else {
            return value
        }
        return "file://" + redact(url)
    }

    private static func replacingMatches(
        in value: String,
        pattern: String,
        template: String
    ) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return value
        }

        let range = NSRange(value.startIndex..., in: value)
        return regex.stringByReplacingMatches(in: value, options: [], range: range, withTemplate: template)
    }

    private static func redactVolumeName(in path: String) -> String {
        guard path.hasPrefix("/Volumes/") else {
            return path
        }

        let remainder = path.dropFirst("/Volumes/".count)
        guard let slashIndex = remainder.firstIndex(of: "/") else {
            return "/Volumes/<volume>"
        }
        return "/Volumes/<volume>/" + remainder[remainder.index(after: slashIndex)...]
    }
}
