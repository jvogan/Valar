import Foundation
import ValarModelKit

/// Conservative paragraph-first chunking for Qwen long-form narration.
///
/// The goal is to preserve voice/prosody continuity by keeping chunk counts low and
/// only segmenting once text is well beyond a safe one-shot range.
public struct QwenTextChunker: Sendable {

    public struct Policy: Sendable, Equatable {
        public let minimumLongFormCharacters: Int
        public let minimumLongFormWords: Int?
        public let targetChunkCharacters: Int
        public let hardMaxCharacters: Int

        public init(
            minimumLongFormCharacters: Int,
            minimumLongFormWords: Int? = nil,
            targetChunkCharacters: Int,
            hardMaxCharacters: Int
        ) {
            self.minimumLongFormCharacters = minimumLongFormCharacters
            self.minimumLongFormWords = minimumLongFormWords
            self.targetChunkCharacters = targetChunkCharacters
            self.hardMaxCharacters = hardMaxCharacters
        }

        // Avoid the old microchunk behavior, but segment medium/long narration before
        // Qwen falls into pathological one-shot runs that are slow and truncate early.
        public static let expressive = Policy(
            minimumLongFormCharacters: 1_500,
            minimumLongFormWords: 240,
            targetChunkCharacters: 1_500,
            hardMaxCharacters: 2_200
        )
        public static let stableNarrator = Policy(
            minimumLongFormCharacters: 900,
            minimumLongFormWords: 140,
            targetChunkCharacters: 1_000,
            hardMaxCharacters: 1_400
        )
    }

    public struct ChunkPlan: Sendable, Equatable {
        public let chunks: [String]
        public let isLongForm: Bool
        public let policy: Policy
    }

    public static func policy(for behavior: SpeechSynthesisVoiceBehavior) -> Policy {
        switch behavior {
        case .stableNarrator:
            return .stableNarrator
        case .auto, .expressive:
            return .expressive
        }
    }

    public static func plan(
        text: String,
        behavior: SpeechSynthesisVoiceBehavior
    ) -> ChunkPlan {
        let policy = policy(for: behavior)
        return plan(text: text, policy: policy)
    }

    public static func plan(
        text: String,
        policy: Policy = .expressive
    ) -> ChunkPlan {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ChunkPlan(chunks: [], isLongForm: false, policy: policy)
        }
        let wordCount = estimatedWordCount(in: trimmed)
        let exceedsCharacterThreshold = trimmed.count > policy.minimumLongFormCharacters
        let exceedsWordThreshold = policy.minimumLongFormWords.map { wordCount > $0 } ?? false
        guard exceedsCharacterThreshold || exceedsWordThreshold else {
            return ChunkPlan(chunks: [trimmed], isLongForm: false, policy: policy)
        }

        let paragraphs = splitIntoParagraphs(trimmed)
        let merged = greedilyMerge(paragraphs, policy: policy)
        let normalized = merged.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard !normalized.isEmpty else {
            return ChunkPlan(chunks: [trimmed], isLongForm: false, policy: policy)
        }

        if normalized.count == 1, exceedsWordThreshold, !exceedsCharacterThreshold {
            let fallbackChunks = splitWordDenseParagraph(
                trimmed,
                targetWordCount: max(140, Int(Double(policy.minimumLongFormWords ?? 240) * 0.75)),
                hardMaxCharacters: policy.hardMaxCharacters
            )
            if fallbackChunks.count > 1 {
                return ChunkPlan(chunks: fallbackChunks, isLongForm: true, policy: policy)
            }
        }

        return ChunkPlan(
            chunks: normalized,
            isLongForm: normalized.count > 1,
            policy: policy
        )
    }

    private static func splitIntoParagraphs(_ text: String) -> [String] {
        if let blankLineRegex = try? NSRegularExpression(pattern: #"\n\s*\n+"#, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            let paragraphs = blankLineRegex
                .stringByReplacingMatches(in: text, range: range, withTemplate: "\u{0000}")
                .components(separatedBy: "\u{0000}")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if paragraphs.count > 1 {
                return paragraphs
            }
        }

        let lines = text
            .components(separatedBy: CharacterSet.newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if lines.count > 1 {
            return lines
        }

        return [text]
    }

    private static func greedilyMerge(_ paragraphs: [String], policy: Policy) -> [String] {
        var chunks: [String] = []
        var current = ""

        for paragraph in paragraphs {
            if paragraph.count > policy.hardMaxCharacters {
                if !current.isEmpty {
                    chunks.append(current)
                    current = ""
                }
                chunks.append(contentsOf: splitOversizedParagraph(paragraph, policy: policy))
                continue
            }

            if current.isEmpty {
                current = paragraph
                continue
            }

            let candidate = current + "\n\n" + paragraph
            if candidate.count <= policy.targetChunkCharacters {
                current = candidate
            } else {
                chunks.append(current)
                current = paragraph
            }
        }

        if !current.isEmpty {
            chunks.append(current)
        }

        return chunks
    }

    private static func splitOversizedParagraph(_ paragraph: String, policy: Policy) -> [String] {
        let sentences = splitIntoSentences(paragraph)
        guard sentences.count > 1 else {
            return hardWrap(paragraph, hardMaxCharacters: policy.hardMaxCharacters)
        }

        var chunks: [String] = []
        var current = ""

        for sentence in sentences {
            if sentence.count > policy.hardMaxCharacters {
                if !current.isEmpty {
                    chunks.append(current)
                    current = ""
                }
                chunks.append(contentsOf: hardWrap(sentence, hardMaxCharacters: policy.hardMaxCharacters))
                continue
            }

            if current.isEmpty {
                current = sentence
                continue
            }

            let candidate = current + " " + sentence
            if candidate.count <= policy.targetChunkCharacters {
                current = candidate
            } else {
                chunks.append(current)
                current = sentence
            }
        }

        if !current.isEmpty {
            chunks.append(current)
        }

        return chunks
    }

    private static func splitIntoSentences(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: #"(?<=[.!?])\s+"#, options: []) else {
            return [text]
        }
        let range = NSRange(text.startIndex..., in: text)
        return regex
            .stringByReplacingMatches(in: text, range: range, withTemplate: "\u{0000}")
            .components(separatedBy: "\u{0000}")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func splitWordDenseParagraph(
        _ text: String,
        targetWordCount: Int,
        hardMaxCharacters: Int
    ) -> [String] {
        let words = text
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
        guard words.count > targetWordCount else {
            return [text]
        }

        var chunks: [String] = []
        var currentWords: [String] = []
        var currentCharacterCount = 0

        for word in words {
            let separatorCount = currentWords.isEmpty ? 0 : 1
            let candidateCharacterCount = currentCharacterCount + separatorCount + word.count
            if !currentWords.isEmpty,
               (currentWords.count >= targetWordCount || candidateCharacterCount > hardMaxCharacters) {
                chunks.append(currentWords.joined(separator: " "))
                currentWords.removeAll(keepingCapacity: true)
                currentCharacterCount = 0
            }

            currentWords.append(word)
            currentCharacterCount += (currentWords.count == 1 ? 0 : 1) + word.count
        }

        if !currentWords.isEmpty {
            chunks.append(currentWords.joined(separator: " "))
        }

        return chunks
    }

    private static func hardWrap(_ text: String, hardMaxCharacters: Int) -> [String] {
        guard text.count > hardMaxCharacters else { return [text] }

        var chunks: [String] = []
        var remaining = text[...]

        while remaining.count > hardMaxCharacters {
            let splitIndex = remaining.index(remaining.startIndex, offsetBy: hardMaxCharacters)
            let prefix = String(remaining[..<splitIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !prefix.isEmpty {
                chunks.append(prefix)
            }
            remaining = remaining[splitIndex...]
            while remaining.first?.isWhitespace == true {
                remaining = remaining.dropFirst()
            }
        }

        let tail = String(remaining).trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty {
            chunks.append(tail)
        }

        return chunks
    }

    private static func estimatedWordCount(in text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        if words > 0 {
            return words
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return 1
        }
        return max(1, Int(ceil(Double(trimmed.count) / 5.0)))
    }
}
