import Foundation

public enum ProjectTextImporter {
    public struct SegmentDraft: Sendable, Equatable {
        public let title: String
        public let text: String
        public let speakerLabel: String?
        public let sourceHash: String

        public init(
            title: String,
            text: String,
            speakerLabel: String? = nil,
            sourceHash: String
        ) {
            self.title = title
            self.text = text
            self.speakerLabel = speakerLabel
            self.sourceHash = sourceHash
        }
    }

    public static func parse(
        text: String,
        fallbackTitle: String,
        splitMode: String,
        defaultSpeakerLabel: String? = nil
    ) -> [SegmentDraft] {
        let normalizedMode = splitMode
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let drafts: [(title: String, text: String, speaker: String?)]

        switch normalizedMode {
        case "markdown", "markdown-headings", "headings":
            drafts = markdownHeadingSegments(text: text, fallbackTitle: fallbackTitle, speakerLabel: defaultSpeakerLabel)
        case "paragraph", "paragraphs", "blank-line":
            drafts = paragraphSegments(text: text, fallbackTitle: fallbackTitle, speakerLabel: defaultSpeakerLabel)
        case "dialogue", "script", "speaker-lines":
            drafts = dialogueSegments(text: text, fallbackTitle: fallbackTitle, defaultSpeakerLabel: defaultSpeakerLabel)
        case "line", "lines":
            drafts = lineSegments(text: text, fallbackTitle: fallbackTitle, speakerLabel: defaultSpeakerLabel)
        default:
            drafts = wholeDocumentSegment(text: text, fallbackTitle: fallbackTitle, speakerLabel: defaultSpeakerLabel)
        }

        return drafts.enumerated().map { index, draft in
            let title = draft.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "\(fallbackTitle) \(index + 1)"
                : draft.title.trimmingCharacters(in: .whitespacesAndNewlines)
            let body = draft.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return SegmentDraft(
                title: title,
                text: body,
                speakerLabel: draft.speaker?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty,
                sourceHash: ProjectScriptMarkup.sourceHash(title: title, text: body, speakerLabel: draft.speaker)
            )
        }
        .filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    private static func markdownHeadingSegments(
        text: String,
        fallbackTitle: String,
        speakerLabel: String?
    ) -> [(title: String, text: String, speaker: String?)] {
        var segments: [(title: String, text: String, speaker: String?)] = []
        var currentTitle = fallbackTitle
        var currentLines: [String] = []

        for line in text.components(separatedBy: .newlines) {
            if let heading = markdownHeadingTitle(from: line) {
                appendSegment(title: currentTitle, lines: currentLines, speaker: speakerLabel, to: &segments)
                currentTitle = heading
                currentLines = []
            } else {
                currentLines.append(line)
            }
        }

        appendSegment(title: currentTitle, lines: currentLines, speaker: speakerLabel, to: &segments)
        return segments.isEmpty ? wholeDocumentSegment(text: text, fallbackTitle: fallbackTitle, speakerLabel: speakerLabel) : segments
    }

    private static func paragraphSegments(
        text: String,
        fallbackTitle: String,
        speakerLabel: String?
    ) -> [(title: String, text: String, speaker: String?)] {
        splitOnBlankLines(text).enumerated().map { index, paragraph in
            let title = firstSentenceTitle(paragraph) ?? "\(fallbackTitle) \(index + 1)"
            return (title: title, text: paragraph, speaker: speakerLabel)
        }
    }

    private static func lineSegments(
        text: String,
        fallbackTitle: String,
        speakerLabel: String?
    ) -> [(title: String, text: String, speaker: String?)] {
        text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .enumerated()
            .map { index, line in
                (title: firstSentenceTitle(line) ?? "\(fallbackTitle) \(index + 1)", text: line, speaker: speakerLabel)
            }
    }

    private static func dialogueSegments(
        text: String,
        fallbackTitle: String,
        defaultSpeakerLabel: String?
    ) -> [(title: String, text: String, speaker: String?)] {
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        var segments: [(title: String, text: String, speaker: String?)] = []

        for (index, line) in lines.enumerated() {
            let parsed = ProjectScriptMarkup.parseLine(line, lineNumber: index + 1)
            let speaker = parsed.speakerLabel ?? defaultSpeakerLabel
            let body = parsed.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !body.isEmpty else { continue }
            let speakerTitle = speaker.map { "\($0): " } ?? ""
            segments.append((
                title: "\(speakerTitle)\(firstSentenceTitle(body) ?? "\(fallbackTitle) \(index + 1)")",
                text: body,
                speaker: speaker
            ))
        }

        return segments.isEmpty ? wholeDocumentSegment(text: text, fallbackTitle: fallbackTitle, speakerLabel: defaultSpeakerLabel) : segments
    }

    private static func wholeDocumentSegment(
        text: String,
        fallbackTitle: String,
        speakerLabel: String?
    ) -> [(title: String, text: String, speaker: String?)] {
        [(title: fallbackTitle, text: text, speaker: speakerLabel)]
    }

    private static func appendSegment(
        title: String,
        lines: [String],
        speaker: String?,
        to segments: inout [(title: String, text: String, speaker: String?)]
    ) {
        let text = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        segments.append((title: title, text: text, speaker: speaker))
    }

    private static func markdownHeadingTitle(from line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("#") else { return nil }
        let marker = trimmed.prefix { $0 == "#" }
        guard (1...6).contains(marker.count),
              trimmed.dropFirst(marker.count).first == " " else {
            return nil
        }
        return String(trimmed.dropFirst(marker.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nonEmpty
    }

    private static func splitOnBlankLines(_ text: String) -> [String] {
        text.components(separatedBy: "\n\n")
            .map {
                $0.components(separatedBy: .newlines)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
            }
            .filter { !$0.isEmpty }
    }

    private static func firstSentenceTitle(_ text: String) -> String? {
        let normalized = text
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard !normalized.isEmpty else { return nil }
        let sentence = normalized.split(whereSeparator: { ".!?".contains($0) }).first.map(String.init) ?? normalized
        let title = sentence.count > 80 ? String(sentence.prefix(77)) + "..." : sentence
        return title.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
