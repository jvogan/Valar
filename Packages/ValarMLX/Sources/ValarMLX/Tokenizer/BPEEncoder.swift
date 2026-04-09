import Foundation

/// GPT-2-style byte-level BPE encoder and decoder.
///
/// Combines ``BPEPreTokenizer`` (regex splitting + byte encoding) and
/// ``BPEVocabulary`` (merge rules + token↔ID maps) to perform full
/// text → token-ID encoding and token-ID → text decoding.
///
/// The merge loop implements the standard greedy BPE algorithm:
/// repeatedly merge the highest-priority adjacent pair until no
/// more merges exist.
public struct BPEEncoder: Sendable {
    private static let maxCacheEntries = 10_000
    static let maxInputCharacterCount = 10_000

    /// The vocabulary backing this encoder.
    public let vocabulary: BPEVocabulary

    /// Cache of word → merged symbols to avoid re-running the merge loop
    /// for previously seen words. Keyed by the byte-encoded word string.
    ///
    /// Thread safety: `BPEEncoder` is a value type and this cache is
    /// copy-on-write; concurrent use from different copies is safe.
    private var cache: [String: [String]]

    public init(vocabulary: BPEVocabulary) {
        self.vocabulary = vocabulary
        self.cache = [:]
    }

    // MARK: - Encode

    /// Encode text into BPE token IDs.
    ///
    /// 1. Pre-tokenizes the input using ``BPEPreTokenizer``.
    /// 2. Applies the BPE merge loop to each pre-tokenized word.
    /// 3. Maps each merged symbol to its vocabulary ID.
    ///
    /// Unknown symbols (not in the vocabulary) are silently skipped.
    ///
    /// - Parameter text: The raw input text.
    /// - Throws: ``TokenizerInputError/inputTooLong(characterCount:limit:)`` when
    ///   the input exceeds ``maxInputCharacterCount`` characters.
    /// - Returns: An array of token IDs.
    public mutating func encode(_ text: String) throws -> [Int] {
        try Self.validateInputLength(text)
        let words = BPEPreTokenizer.preTokenize(text)
        var ids: [Int] = []

        for word in words {
            let symbols = bpe(word)
            for symbol in symbols {
                if let id = vocabulary.id(for: symbol) {
                    ids.append(id)
                }
            }
        }

        return ids
    }

    /// Encode text into BPE token strings (without mapping to IDs).
    ///
    /// Useful for debugging and testing the merge loop independently
    /// of the vocabulary mapping.
    ///
    /// - Parameter text: The raw input text.
    /// - Throws: ``TokenizerInputError/inputTooLong(characterCount:limit:)`` when
    ///   the input exceeds ``maxInputCharacterCount`` characters.
    /// - Returns: An array of merged BPE token strings.
    public mutating func tokenize(_ text: String) throws -> [String] {
        try Self.validateInputLength(text)
        let words = BPEPreTokenizer.preTokenize(text)
        var tokens: [String] = []

        for word in words {
            tokens.append(contentsOf: bpe(word))
        }

        return tokens
    }

    // MARK: - Decode

    /// Decode a sequence of token IDs back to text.
    ///
    /// Joins the token strings and decodes from the GPT-2 byte-to-unicode
    /// representation back to UTF-8 text.
    ///
    /// Unknown IDs (not in the vocabulary) are silently skipped.
    ///
    /// - Parameter ids: An array of token IDs.
    /// - Returns: The decoded text, or `nil` if the byte sequence is not valid UTF-8.
    public func decode(_ ids: [Int]) -> String? {
        var encoded = ""
        for id in ids {
            if let token = vocabulary.token(for: id) {
                encoded += token
            }
        }
        return BPEPreTokenizer.byteDecode(encoded)
    }

    // MARK: - BPE merge loop

    /// Run the BPE merge loop on a byte-encoded word.
    ///
    /// Starts with each character as a separate symbol, then repeatedly
    /// merges the adjacent pair with the lowest rank (highest priority)
    /// until no more merge rules apply.
    ///
    /// Results are cached to avoid redundant work for repeated words.
    ///
    /// - Parameter word: A byte-encoded word from ``BPEPreTokenizer/preTokenize(_:)``.
    /// - Returns: The final merged symbol list.
    mutating func bpe(_ word: String) -> [String] {
        if let cached = cache[word] {
            return cached
        }

        // Start with each character as its own symbol.
        var symbols = word.map { String($0) }

        guard symbols.count > 1 else {
            storeInCache(symbols, for: word)
            return symbols
        }

        // Iteratively merge the best pair.
        while symbols.count > 1 {
            // Find the pair with the lowest merge rank (highest priority).
            var bestRank = Int.max
            var bestIndex = -1

            for i in 0..<(symbols.count - 1) {
                let pair = BPEVocabulary.MergePair(first: symbols[i], second: symbols[i + 1])
                if let rank = vocabulary.priority(of: pair), rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }

            // No more merges possible.
            if bestIndex == -1 {
                break
            }

            // Merge all occurrences of the best pair in a single pass.
            let mergedSymbol = symbols[bestIndex] + symbols[bestIndex + 1]
            let targetFirst = symbols[bestIndex]
            let targetSecond = symbols[bestIndex + 1]

            var newSymbols: [String] = []
            newSymbols.reserveCapacity(symbols.count)

            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1
                    && symbols[i] == targetFirst
                    && symbols[i + 1] == targetSecond
                {
                    newSymbols.append(mergedSymbol)
                    i += 2
                } else {
                    newSymbols.append(symbols[i])
                    i += 1
                }
            }

            symbols = newSymbols
        }

        storeInCache(symbols, for: word)
        return symbols
    }

    private mutating func storeInCache(_ symbols: [String], for word: String) {
        cache[word] = symbols

        if cache.count > Self.maxCacheEntries {
            cache.removeAll(keepingCapacity: true)
            cache[word] = symbols
        }
    }

    private static func validateInputLength(_ text: String) throws {
        let characterCount = text.count
        guard characterCount <= Self.maxInputCharacterCount else {
            throw TokenizerInputError.inputTooLong(
                characterCount: characterCount,
                limit: Self.maxInputCharacterCount
            )
        }
    }
}
