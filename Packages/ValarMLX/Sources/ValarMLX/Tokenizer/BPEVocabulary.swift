import Foundation

/// Loaded BPE vocabulary and merge rules for a tokenizer.
///
/// Parses the standard Hugging Face `vocab.json` (token → ID map) and `merges.txt`
/// (ordered merge rules) files used by BPE tokenizers. This is the data-loading layer
/// only — it does not perform encoding or decoding.
public struct BPEVocabulary: Sendable {
    /// Token string → token ID.
    public let tokenToID: [String: Int]

    /// Token ID → token string (inverse of `tokenToID`).
    public let idToToken: [Int: String]

    /// Ordered merge rules. Each element is a pair of tokens to merge.
    /// Priority is determined by position: lower index = higher priority.
    public let merges: [MergePair]

    /// Merge-pair lookup keyed by the two constituent tokens, value is the priority rank
    /// (0 = highest priority). Used for O(1) merge-priority lookups during encoding.
    public let mergePriority: [MergePair: Int]

    /// The number of tokens in the vocabulary.
    public var count: Int { tokenToID.count }

    /// A single BPE merge rule: two tokens that should be combined.
    public struct MergePair: Hashable, Sendable {
        public let first: String
        public let second: String

        public init(first: String, second: String) {
            self.first = first
            self.second = second
        }
    }

    /// Creates a vocabulary from pre-parsed data.
    public init(tokenToID: [String: Int], merges: [MergePair]) {
        self.tokenToID = tokenToID
        self.idToToken = Dictionary(tokenToID.map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        self.merges = merges
        self.mergePriority = Dictionary(
            merges.enumerated().map { ($1, $0) },
            uniquingKeysWith: { first, _ in first }
        )
    }

    /// Loads a BPE vocabulary from `vocab.json` and `merges.txt` files on disk.
    ///
    /// - Parameters:
    ///   - vocabURL: Path to a JSON file mapping token strings to integer IDs.
    ///   - mergesURL: Path to a text file containing ordered BPE merge rules.
    /// - Throws: ``BPELoadError`` if either file is missing, malformed, or unreadable.
    public init(vocabURL: URL, mergesURL: URL) throws {
        let tokenToID = try Self.loadVocab(from: vocabURL)
        let merges = try Self.loadMerges(from: mergesURL)
        self.init(tokenToID: tokenToID, merges: merges)
    }

    /// Loads a BPE vocabulary by resolving `vocab.json` and `merges.txt` within a directory.
    ///
    /// - Parameter directory: A directory containing both `vocab.json` and `merges.txt`.
    /// - Throws: ``BPELoadError`` if either file is missing or malformed.
    public init(directory: URL) throws {
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")
        try self.init(vocabURL: vocabURL, mergesURL: mergesURL)
    }

    /// Returns the token ID for a given token string, or `nil` if not found.
    public func id(for token: String) -> Int? {
        tokenToID[token]
    }

    /// Returns the token string for a given token ID, or `nil` if not found.
    public func token(for id: Int) -> String? {
        idToToken[id]
    }

    /// Returns the merge priority (0 = highest) for a pair, or `nil` if the pair has no rule.
    public func priority(of pair: MergePair) -> Int? {
        mergePriority[pair]
    }
}

// MARK: - File parsing

extension BPEVocabulary {
    static func loadVocab(from url: URL) throws -> [String: Int] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw BPELoadError.fileUnreadable(path: url.path, underlying: error)
        }

        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw BPELoadError.invalidVocabJSON(path: url.path, underlying: error)
        }

        guard let dict = parsed as? [String: Any] else {
            throw BPELoadError.invalidVocabJSON(
                path: url.path,
                underlying: BPELoadError.unexpectedJSONStructure
            )
        }

        var result = [String: Int](minimumCapacity: dict.count)
        for (key, value) in dict {
            guard let intValue = value as? Int ?? (value as? NSNumber)?.intValue else {
                throw BPELoadError.invalidVocabEntry(token: key, path: url.path)
            }
            result[key] = intValue
        }
        return result
    }

    static func loadMerges(from url: URL) throws -> [MergePair] {
        let contents: String
        do {
            contents = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw BPELoadError.fileUnreadable(path: url.path, underlying: error)
        }

        var pairs: [MergePair] = []
        for line in contents.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Skip empty lines and the version header (e.g. "#version: 0.2")
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            // Each merge line is exactly two space-separated tokens.
            // Find the first space to split — tokens themselves do not contain spaces
            // in standard BPE merges files.
            guard let spaceIndex = trimmed.firstIndex(of: " ") else {
                throw BPELoadError.invalidMergeLine(line: String(trimmed), path: url.path)
            }

            let first = String(trimmed[trimmed.startIndex..<spaceIndex])
            let rest = trimmed[trimmed.index(after: spaceIndex)...]

            // Validate no additional spaces (should be exactly two tokens).
            guard !rest.contains(" ") else {
                throw BPELoadError.invalidMergeLine(line: String(trimmed), path: url.path)
            }

            let second = String(rest)
            guard !first.isEmpty, !second.isEmpty else {
                throw BPELoadError.invalidMergeLine(line: String(trimmed), path: url.path)
            }

            pairs.append(MergePair(first: first, second: second))
        }

        return pairs
    }
}

// MARK: - Errors

public enum BPELoadError: Error, Sendable, LocalizedError {
    case fileUnreadable(path: String, underlying: any Error)
    case invalidVocabJSON(path: String, underlying: any Error)
    case invalidVocabEntry(token: String, path: String)
    case invalidMergeLine(line: String, path: String)
    case unexpectedJSONStructure

    public var errorDescription: String? {
        switch self {
        case .fileUnreadable(let path, let underlying):
            return "Cannot read BPE file at \(path): \(underlying.localizedDescription)"
        case .invalidVocabJSON(let path, let underlying):
            return "Invalid vocab JSON at \(path): \(underlying.localizedDescription)"
        case .invalidVocabEntry(let token, let path):
            return "Non-integer value for token '\(token)' in \(path)"
        case .invalidMergeLine(let line, let path):
            return "Invalid merge line '\(line)' in \(path)"
        case .unexpectedJSONStructure:
            return "Expected a JSON object mapping strings to integers"
        }
    }
}
