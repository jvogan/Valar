import Foundation

struct VoxtralTTSTekkenFile: Decodable {
    struct Config: Decodable {
        let pattern: String?
        let defaultVocabSize: Int?
        let defaultNumSpecialTokens: Int?

        enum CodingKeys: String, CodingKey {
            case pattern
            case defaultVocabSize = "default_vocab_size"
            case defaultNumSpecialTokens = "default_num_special_tokens"
        }
    }

    struct SpecialToken: Decodable {
        let rank: Int
        let tokenString: String
        let isControl: Bool?

        enum CodingKeys: String, CodingKey {
            case rank
            case tokenString = "token_str"
            case isControl = "is_control"
        }
    }

    struct VocabEntry: Decodable {
        let rank: Int?
        let tokenBytes: String
        let tokenString: String?

        enum CodingKeys: String, CodingKey {
            case rank
            case tokenBytes = "token_bytes"
            case tokenString = "token_str"
        }
    }

    let vocab: [VocabEntry]
    let config: Config?
    let specialTokens: [SpecialToken]?

    enum CodingKeys: String, CodingKey {
        case vocab
        case config
        case specialTokens = "special_tokens"
    }
}

enum VoxtralTTSTokenizerError: Error, LocalizedError {
    case missingTekken(URL)
    case invalidRegex(String)
    case missingSpecialToken(String)

    var errorDescription: String? {
        switch self {
        case .missingTekken(let modelDir):
            return "tekken.json not found at \(modelDir.path)"
        case .invalidRegex(let pattern):
            return "Invalid Tekken regex pattern: \(pattern)"
        case .missingSpecialToken(let token):
            return "Missing Tekken special token \(token)"
        }
    }
}

final class VoxtralTTSTokenizer {
    private enum SpecialTokenString {
        static let bos = "~~"
        static let audio = "[AUDIO]"
        static let beginAudio = "[BEGIN_AUDIO]"
        static let nextAudioText = "[NEXT_AUDIO_TEXT]"
        static let repeatAudioText = "[REPEAT_AUDIO_TEXT]"
    }

    let nSpecial: Int
    let specialIds: Set<Int>
    let bosTokenId: Int
    let audioTokenId: Int
    let beginAudioTokenId: Int
    let nextAudioTextTokenId: Int
    let repeatAudioTextTokenId: Int

    private let mergeableRanks: [Data: Int]
    private let tokenBytesByRank: [Data]
    private let specialTokenIdsByString: [String: Int]
    private let regex: NSRegularExpression?

    init(
        tekkenURL: URL,
        audioTokenId defaultAudioTokenId: Int = 24,
        beginAudioTokenId defaultBeginAudioTokenId: Int = 25
    ) throws {
        let data = try Data(contentsOf: tekkenURL)
        let parsed = try JSONDecoder().decode(VoxtralTTSTekkenFile.self, from: data)

        nSpecial = parsed.config?.defaultNumSpecialTokens ?? 1000
        let defaultVocabSize = parsed.config?.defaultVocabSize ?? (parsed.vocab.count + nSpecial)
        let mergeableVocabCount = max(0, min(parsed.vocab.count, defaultVocabSize - nSpecial))

        var mergeableRanks: [Data: Int] = [:]
        var tokenBytesByRank = Array(repeating: Data(), count: mergeableVocabCount)

        for (index, entry) in parsed.vocab.prefix(mergeableVocabCount).enumerated() {
            let rank = entry.rank ?? index
            guard rank >= 0, rank < mergeableVocabCount else { continue }
            let bytes = Data(base64Encoded: entry.tokenBytes) ?? Data()
            mergeableRanks[bytes] = rank
            tokenBytesByRank[rank] = bytes
        }

        self.mergeableRanks = mergeableRanks
        self.tokenBytesByRank = tokenBytesByRank

        let specialTokens = parsed.specialTokens ?? []
        specialIds = Set(specialTokens.map(\.rank))
        specialTokenIdsByString = Dictionary(
            uniqueKeysWithValues: specialTokens.map { ($0.tokenString, $0.rank) }
        )

        if let pattern = parsed.config?.pattern, !pattern.isEmpty {
            guard pattern.count < 500 else {
                throw VoxtralTTSTokenizerError.invalidRegex(pattern)
            }
            do {
                regex = try NSRegularExpression(pattern: pattern)
            } catch {
                throw VoxtralTTSTokenizerError.invalidRegex(pattern)
            }
        } else {
            regex = nil
        }

        bosTokenId = specialTokenIdsByString[SpecialTokenString.bos] ?? 1
        audioTokenId = specialTokenIdsByString[SpecialTokenString.audio] ?? defaultAudioTokenId
        beginAudioTokenId = specialTokenIdsByString[SpecialTokenString.beginAudio] ?? defaultBeginAudioTokenId
        nextAudioTextTokenId = try Self.requiredTokenID(
            for: SpecialTokenString.nextAudioText,
            in: specialTokenIdsByString
        )
        repeatAudioTextTokenId = try Self.requiredTokenID(
            for: SpecialTokenString.repeatAudioText,
            in: specialTokenIdsByString
        )
    }

    static func fromModelDirectory(
        _ modelDir: URL,
        audioTokenId: Int = 24,
        beginAudioTokenId: Int = 25
    ) throws -> VoxtralTTSTokenizer {
        let tekkenURL = modelDir.appendingPathComponent("tekken.json")
        guard FileManager.default.fileExists(atPath: tekkenURL.path) else {
            throw VoxtralTTSTokenizerError.missingTekken(modelDir)
        }
        return try VoxtralTTSTokenizer(
            tekkenURL: tekkenURL,
            audioTokenId: audioTokenId,
            beginAudioTokenId: beginAudioTokenId
        )
    }

    func encode(text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        var tokenIds: [Int] = []
        for chunk in chunked(text: text) {
            tokenIds.append(contentsOf: encodeChunk(Array(chunk.utf8)))
        }
        return tokenIds
    }

    func decode(tokenIds: [Int]) -> String {
        var out = Data()
        out.reserveCapacity(tokenIds.count * 2)

        for tokenId in tokenIds {
            guard tokenId >= nSpecial, !specialIds.contains(tokenId) else { continue }
            let rank = tokenId - nSpecial
            guard rank >= 0, rank < tokenBytesByRank.count else { continue }
            out.append(tokenBytesByRank[rank])
        }

        return String(decoding: out, as: UTF8.self)
    }

    func packSpeechRequest(text: String, voiceFrameCount: Int) -> [Int] {
        let placeholderCount = max(0, voiceFrameCount)
        let textTokenIds = encode(text: text)

        var tokens = [bosTokenId, beginAudioTokenId]
        tokens.append(contentsOf: Array(repeating: audioTokenId, count: placeholderCount))
        tokens.append(nextAudioTextTokenId)
        tokens.append(contentsOf: textTokenIds)
        tokens.append(repeatAudioTextTokenId)
        tokens.append(beginAudioTokenId)
        return tokens
    }

    private static func requiredTokenID(for token: String, in idsByString: [String: Int]) throws -> Int {
        guard let tokenId = idsByString[token] else {
            throw VoxtralTTSTokenizerError.missingSpecialToken(token)
        }
        return tokenId
    }

    private func chunked(text: String) -> [String] {
        guard let regex else { return [text] }

        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let matches = regex.matches(in: text, range: fullRange)
        guard !matches.isEmpty else { return [text] }

        var chunks: [String] = []
        var cursor = 0

        for match in matches {
            if match.range.location > cursor {
                let gapRange = NSRange(location: cursor, length: match.range.location - cursor)
                let gap = nsText.substring(with: gapRange)
                if !gap.isEmpty {
                    chunks.append(gap)
                }
            }

            let chunk = nsText.substring(with: match.range)
            if !chunk.isEmpty {
                chunks.append(chunk)
            }
            cursor = match.range.location + match.range.length
        }

        if cursor < nsText.length {
            let trailing = nsText.substring(from: cursor)
            if !trailing.isEmpty {
                chunks.append(trailing)
            }
        }

        return chunks
    }

    private func encodeChunk(_ bytes: [UInt8]) -> [Int] {
        guard !bytes.isEmpty else { return [] }

        var pieces = bytes.map { Data([$0]) }

        // Merge adjacent byte pieces using the lowest-rank mergeable token at each step.
        while pieces.count > 1 {
            var bestIndex: Int?
            var bestMerged: Data?
            var bestRank = Int.max

            for index in 0..<(pieces.count - 1) {
                var merged = pieces[index]
                merged.append(pieces[index + 1])
                guard let rank = mergeableRanks[merged], rank < bestRank else { continue }
                bestIndex = index
                bestMerged = merged
                bestRank = rank
            }

            guard let bestIndex, let bestMerged else { break }
            pieces[bestIndex] = bestMerged
            pieces.remove(at: bestIndex + 1)
        }

        var tokenIds: [Int] = []
        tokenIds.reserveCapacity(pieces.count)

        for piece in pieces {
            if let rank = mergeableRanks[piece] {
                tokenIds.append(nSpecial + rank)
            } else if piece.count == 1, let byte = piece.first {
                let bytePiece = Data([byte])
                if let rank = mergeableRanks[bytePiece] {
                    tokenIds.append(nSpecial + rank)
                }
            }
        }

        return tokenIds
    }
}
