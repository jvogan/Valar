import CryptoKit
import Foundation

public enum ProjectScriptMarkup {
    public struct ParsedLine: Sendable, Equatable {
        public let lineNumber: Int
        public let speakerLabel: String?
        public let text: String
        public let attributes: [String: String]
        public let tags: [String]

        public init(
            lineNumber: Int,
            speakerLabel: String? = nil,
            text: String,
            attributes: [String: String] = [:],
            tags: [String] = []
        ) {
            self.lineNumber = lineNumber
            self.speakerLabel = speakerLabel
            self.text = text
            self.attributes = attributes
            self.tags = tags
        }
    }

    public static func parseLine(_ line: String, lineNumber: Int) -> ParsedLine {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ParsedLine(lineNumber: lineNumber, text: "")
        }

        if trimmed.hasPrefix("["),
           let closing = trimmed.firstIndex(of: "]") {
            let rawSpeakerSpec = String(trimmed[trimmed.index(after: trimmed.startIndex)..<closing])
            let body = String(trimmed[trimmed.index(after: closing)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let parsedSpec = parseSpeakerSpec(rawSpeakerSpec)
            return ParsedLine(
                lineNumber: lineNumber,
                speakerLabel: parsedSpec.speakerLabel,
                text: body,
                attributes: parsedSpec.attributes,
                tags: normalizedTags(parsedSpec.tags + inlineTags(in: body))
            )
        }

        if let colon = trimmed.firstIndex(of: ":") {
            let prefix = String(trimmed[..<colon]).trimmingCharacters(in: .whitespacesAndNewlines)
            let body = String(trimmed[trimmed.index(after: colon)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if isSpeakerPrefix(prefix) {
                return ParsedLine(
                    lineNumber: lineNumber,
                    speakerLabel: prefix,
                    text: body,
                    tags: normalizedTags(inlineTags(in: body))
                )
            }
        }

        return ParsedLine(
            lineNumber: lineNumber,
            text: trimmed,
            tags: normalizedTags(inlineTags(in: trimmed))
        )
    }

    public static func parseText(_ text: String) -> [ParsedLine] {
        text.components(separatedBy: .newlines).enumerated().compactMap { index, line in
            let parsed = parseLine(line, lineNumber: index + 1)
            return parsed.text.isEmpty ? nil : parsed
        }
    }

    public static func sourceHash(title: String, text: String, speakerLabel: String?) -> String {
        let raw = [title, speakerLabel ?? "", text].joined(separator: "\u{1f}")
        let digest = SHA256.hash(data: Data(raw.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

private extension ProjectScriptMarkup {
    static func parseSpeakerSpec(_ rawSpec: String) -> (speakerLabel: String?, attributes: [String: String], tags: [String]) {
        let components = rawSpec
            .split(separator: "|", omittingEmptySubsequences: false)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        let speaker = components.first?.nonEmpty
        var attributes: [String: String] = [:]
        var tags: [String] = []

        for component in components.dropFirst() where !component.isEmpty {
            if let separator = component.firstIndex(where: { $0 == ":" || $0 == "=" }) {
                let key = String(component[..<separator])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                let value = String(component[component.index(after: separator)...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !key.isEmpty, !value.isEmpty {
                    attributes[key] = value
                }
            } else {
                tags.append(component)
            }
        }

        return (speaker, attributes, tags)
    }

    static func isSpeakerPrefix(_ prefix: String) -> Bool {
        !prefix.isEmpty
            && prefix.count <= 64
            && prefix.rangeOfCharacter(from: .letters) != nil
            && prefix.rangeOfCharacter(from: .newlines) == nil
    }

    static func inlineTags(in text: String) -> [String] {
        var tags: [String] = []
        tags.append(contentsOf: regexMatches(pattern: #"\[([A-Za-z][A-Za-z0-9_-]{1,31})(?::[^\]]+)?\]"#, in: text))
        tags.append(contentsOf: regexMatches(pattern: #"<([A-Za-z][A-Za-z0-9_-]{1,31})(?:\s[^>]*)?>"#, in: text))
        tags.append(contentsOf: parentheticalTags(in: text))
        return tags
    }

    static func regexMatches(pattern: String, in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            guard match.numberOfRanges > 1,
                  let swiftRange = Range(match.range(at: 1), in: text) else {
                return nil
            }
            return String(text[swiftRange])
        }
    }

    static func parentheticalTags(in text: String) -> [String] {
        let allowed: Set<String> = [
            "breath", "breathes", "chuckle", "chuckles", "cough", "coughs",
            "gasp", "gasps", "laugh", "laughs", "pause", "sigh", "sighs",
            "sniff", "sniffs", "whisper", "whispers", "yawn", "yawns",
        ]
        guard let regex = try? NSRegularExpression(pattern: #"\(([A-Za-z][A-Za-z -]{1,31})\)"#) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            guard match.numberOfRanges > 1,
                  let swiftRange = Range(match.range(at: 1), in: text) else {
                return nil
            }
            let normalized = String(text[swiftRange])
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return allowed.contains(normalized) ? normalized : nil
        }
    }

    static func normalizedTags(_ tags: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for tag in tags {
            let normalized = tag
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !normalized.isEmpty, !seen.contains(normalized) else { continue }
            seen.insert(normalized)
            result.append(normalized)
        }
        return result
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
