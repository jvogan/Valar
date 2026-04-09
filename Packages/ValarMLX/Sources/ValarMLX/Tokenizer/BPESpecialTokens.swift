import Foundation

/// Special token registry loaded from `tokenizer_config.json`.
///
/// Parses the `added_tokens_decoder` section of a Hugging Face tokenizer config
/// to build a lookup table of special tokens (e.g. `<|im_start|>`, `tts_bos`).
/// Special tokens are excluded from the BPE merge process and mapped directly
/// to their fixed IDs.
public struct BPESpecialTokens: Sendable {
    /// A single entry from the `added_tokens_decoder` section.
    public struct Token: Sendable, Equatable {
        public let id: Int
        public let content: String
        public let special: Bool

        public init(id: Int, content: String, special: Bool) {
            self.id = id
            self.content = content
            self.special = special
        }
    }

    /// All loaded special tokens, ordered by ID.
    public let tokens: [Token]

    /// Token content string → token entry.
    public let contentToToken: [String: Token]

    /// Token ID → token entry.
    public let idToToken: [Int: Token]

    /// Set of all special token content strings, for fast membership checks
    /// during BPE encoding.
    public let specialContentSet: Set<String>

    /// Creates a special token registry from pre-parsed token entries.
    public init(tokens: [Token]) {
        self.tokens = tokens.sorted { $0.id < $1.id }
        self.contentToToken = Dictionary(
            tokens.map { ($0.content, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        self.idToToken = Dictionary(
            tokens.map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        self.specialContentSet = Set(
            tokens.filter(\.special).map(\.content)
        )
    }

    /// Creates an empty registry (no special tokens).
    public init() {
        self.init(tokens: [])
    }

    /// Returns the fixed ID for a special token, or `nil` if the content
    /// is not a registered special token.
    public func id(for content: String) -> Int? {
        guard let token = contentToToken[content], token.special else {
            return nil
        }
        return token.id
    }

    /// Returns the content string for a special token ID, or `nil` if the ID
    /// is not a registered special token.
    public func content(for id: Int) -> String? {
        idToToken[id]?.content
    }

    /// Whether the given content string is a special token that should be
    /// excluded from BPE merging.
    public func isSpecial(_ content: String) -> Bool {
        specialContentSet.contains(content)
    }
}

// MARK: - Loading from tokenizer_config.json

extension BPESpecialTokens {
    /// Loads special tokens from a `tokenizer_config.json` file.
    ///
    /// Parses the `added_tokens_decoder` key, which maps string IDs to
    /// token descriptor objects. Example:
    /// ```json
    /// {
    ///   "added_tokens_decoder": {
    ///     "151644": { "content": "<|im_start|>", "special": true },
    ///     "151645": { "content": "<|im_end|>", "special": true }
    ///   }
    /// }
    /// ```
    ///
    /// - Parameter url: Path to `tokenizer_config.json`.
    /// - Throws: ``BPELoadError`` if the file is missing or malformed.
    public init(tokenizerConfigURL url: URL) throws {
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

        guard let root = parsed as? [String: Any] else {
            throw BPELoadError.invalidVocabJSON(
                path: url.path,
                underlying: BPELoadError.unexpectedJSONStructure
            )
        }

        guard let addedTokens = root["added_tokens_decoder"] as? [String: Any] else {
            // No added_tokens_decoder section — valid config, just no special tokens.
            self.init(tokens: [])
            return
        }

        var tokens: [Token] = []
        tokens.reserveCapacity(addedTokens.count)

        for (idString, value) in addedTokens {
            guard let tokenID = Int(idString) else { continue }
            guard let entry = value as? [String: Any] else { continue }
            guard let content = entry["content"] as? String else { continue }

            let special = entry["special"] as? Bool ?? false
            tokens.append(Token(id: tokenID, content: content, special: special))
        }

        self.init(tokens: tokens)
    }

    /// Loads special tokens from a `tokenizer_config.json` in the given directory.
    ///
    /// - Parameter directory: A directory containing `tokenizer_config.json`.
    /// - Throws: ``BPELoadError`` if the file is missing or malformed.
    public init(directory: URL) throws {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        try self.init(tokenizerConfigURL: configURL)
    }
}

// MARK: - Qwen3 TTS well-known tokens

extension BPESpecialTokens {
    /// Well-known token IDs for Qwen3-TTS models.
    public enum Qwen3TTS {
        /// `<|im_start|>` — chat message start marker.
        public static let imStart = 151644
        /// `<|im_end|>` — chat message end marker.
        public static let imEnd = 151645
        /// `tts_pad` — TTS padding token.
        public static let ttsPad = 151671
        /// `tts_bos` — TTS beginning-of-sequence token.
        public static let ttsBos = 151672
        /// `tts_eos` — TTS end-of-sequence token.
        public static let ttsEos = 151673
    }
}
