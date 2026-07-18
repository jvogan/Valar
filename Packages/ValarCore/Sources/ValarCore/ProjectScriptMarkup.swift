import CryptoKit
import Foundation
import ValarModelKit
import ValarPersistence

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

        public var dto: ScriptMarkupLineDTO {
            ScriptMarkupLineDTO(
                id: "line-\(lineNumber)",
                lineNumber: lineNumber,
                speakerLabel: speakerLabel,
                text: text,
                attributes: attributes,
                tags: tags
            )
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

    public static func lintProject(
        project: ProjectRecord,
        chapters: [ChapterRecord],
        speakers: [ProjectSpeakerRecord],
        model: CatalogModel?,
        generatedAt: Date = .now
    ) -> ProjectScriptLintPayloadDTO {
        let sortedChapters = chapters.sorted { lhs, rhs in
            lhs.index == rhs.index ? lhs.title < rhs.title : lhs.index < rhs.index
        }
        let parsedLinesByChapter = sortedChapters.map { chapter in
            (chapter: chapter, lines: parseText(chapter.script))
        }
        let allLines = parsedLinesByChapter.flatMap { $0.lines.map(\.dto) }
        let voiceBible = voiceBible(
            projectID: project.id,
            chapters: sortedChapters,
            speakers: speakers,
            generatedAt: generatedAt
        )
        var issues: [ScriptLintIssueDTO] = []

        func addIssue(
            severity: String,
            code: String,
            message: String,
            lineNumber: Int? = nil,
            chapterID: UUID? = nil,
            speakerLabel: String? = nil,
            tag: String? = nil
        ) {
            issues.append(
                ScriptLintIssueDTO(
                    id: "issue-\(issues.count + 1)",
                    severity: severity,
                    code: code,
                    message: message,
                    lineNumber: lineNumber,
                    chapterID: chapterID?.uuidString,
                    speakerLabel: speakerLabel,
                    modelID: model?.id.rawValue,
                    tag: tag
                )
            )
        }

        let declaredSpeakerNames = Set(speakers.map { $0.name.lowercased() })
        let inferredSpeakerNames = Set(
            (parsedLinesByChapter.flatMap { $0.lines.compactMap(\.speakerLabel) }
                + sortedChapters.compactMap { $0.speakerLabel?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty })
                .map { $0.lowercased() }
        )
        for speaker in inferredSpeakerNames.sorted() where !declaredSpeakerNames.contains(speaker) {
            addIssue(
                severity: "info",
                code: "speaker_missing_voice_profile",
                message: "Speaker '\(speaker)' appears in script markup but has no project voice profile.",
                speakerLabel: speaker
            )
        }

        if inferredSpeakerNames.count > 1, let model, !supportsMultipleSpeakers(model) {
            addIssue(
                severity: "warning",
                code: "model_may_not_hold_cast_consistency",
                message: "\(model.descriptor.displayName) does not declare named-speaker, preset-voice, or voice-design support; multi-speaker renders may drift."
            )
        }

        for chapterLines in parsedLinesByChapter {
            for line in chapterLines.lines {
                let expressiveTokens = line.tags + line.attributes.keys.sorted()
                guard !expressiveTokens.isEmpty else { continue }
                if let model, !supportsExpressiveMarkup(model) {
                    for tag in expressiveTokens {
                        addIssue(
                            severity: "warning",
                            code: "model_may_ignore_expression",
                            message: "\(model.descriptor.displayName) may ignore expressive tag '\(tag)'. Choose a voice-design model or remove unsupported markup.",
                            lineNumber: line.lineNumber,
                            chapterID: chapterLines.chapter.id,
                            speakerLabel: line.speakerLabel,
                            tag: tag
                        )
                    }
                }
            }
        }

        if let model, !model.descriptor.voiceSupport.features.contains(.stableNarrator) {
            for chapter in sortedChapters where chapter.script.count > 8_000 {
                addIssue(
                    severity: "warning",
                    code: "long_segment_drift_risk",
                    message: "Chapter '\(chapter.title)' is long and the selected model does not declare stable narrator support. Split the chapter or use a stable narrator voice for long renders.",
                    chapterID: chapter.id,
                    speakerLabel: chapter.speakerLabel
                )
            }
        }

        let warningCount = issues.filter { $0.severity == "warning" }.count
        let errorCount = issues.filter { $0.severity == "error" }.count
        let message = issues.isEmpty
            ? "No script lint issues found for '\(project.title)'."
            : "Found \(issues.count) script lint issue(s) for '\(project.title)'."

        return ProjectScriptLintPayloadDTO(
            message: message,
            projectID: project.id.uuidString,
            projectTitle: project.title,
            modelID: model?.id.rawValue,
            issueCount: issues.count,
            warningCount: warningCount,
            errorCount: errorCount,
            lines: allLines,
            issues: issues,
            voiceBible: voiceBible
        )
    }

    public static func voiceBible(
        projectID: UUID,
        chapters: [ChapterRecord],
        speakers: [ProjectSpeakerRecord],
        generatedAt: Date = .now
    ) -> ProjectVoiceBibleDTO {
        let inferredChapterSpeakers = chapters.flatMap { chapter in
            let markupSpeakers = parseText(chapter.script)
                .compactMap { $0.speakerLabel?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty }
            if !markupSpeakers.isEmpty {
                return markupSpeakers
            }
            return chapter.speakerLabel
                .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty }
                .map { [$0] } ?? []
        }
        let speakerCounts = Dictionary(
            grouping: inferredChapterSpeakers,
            by: { $0.lowercased() }
        ).mapValues(\.count)
        let declaredByName = Dictionary(
            uniqueKeysWithValues: speakers.map { ($0.name.lowercased(), $0) }
        )
        let names = Set(speakerCounts.keys).union(declaredByName.keys).sorted()
        let profiles = names.map { key in
            let declared = declaredByName[key]
            let displayName = declared?.name ?? key
            let warnings = declared == nil
                ? ["No voice profile has been assigned for this speaker."]
                : []
            return ProjectVoiceProfileDTO(
                id: declared?.id.uuidString ?? stableID(for: "\(projectID.uuidString):\(key)"),
                name: displayName,
                voiceModelID: declared?.voiceModelID,
                language: declared?.language ?? "auto",
                segmentCount: speakerCounts[key] ?? 0,
                warnings: warnings
            )
        }
        return ProjectVoiceBibleDTO(
            projectID: projectID.uuidString,
            generatedAt: iso8601String(from: generatedAt),
            profiles: profiles,
            consistencyPolicy: VoiceConsistencyPolicyDTO()
        )
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

    static func supportsMultipleSpeakers(_ model: CatalogModel) -> Bool {
        let voiceFeatures = Set(model.descriptor.voiceSupport.features)
        return voiceFeatures.contains(.namedSpeakers)
            || voiceFeatures.contains(.presetVoices)
            || voiceFeatures.contains(.voiceDesign)
            || model.descriptor.capabilities.contains(.presetVoices)
    }

    static func supportsExpressiveMarkup(_ model: CatalogModel) -> Bool {
        let capabilities = model.descriptor.capabilities
        let voiceFeatures = Set(model.descriptor.voiceSupport.features)
        return capabilities.contains(.voiceDesign)
            || capabilities.contains(.audioConditioning)
            || voiceFeatures.contains(.voiceDesign)
    }

    static func stableID(for raw: String) -> String {
        let digest = SHA256.hash(data: Data(raw.utf8))
        return digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
