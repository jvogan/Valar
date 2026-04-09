import Foundation

/// Qwen3-TTS chat template formatter.
///
/// Wraps user text in the Qwen3 chat template format expected by the
/// TTS model. The template places text within an assistant turn using
/// `<|im_start|>` / `<|im_end|>` markers, then opens a second assistant
/// turn for the model to generate speech tokens into.
///
/// Output format:
/// ```
/// <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
/// ```
public struct Qwen3ChatTemplate: Sendable {
    /// The special tokens registry used for token ID resolution.
    public let specialTokens: BPESpecialTokens

    public init(specialTokens: BPESpecialTokens) {
        self.specialTokens = specialTokens
    }

    /// Creates a template with default Qwen3-TTS special token IDs.
    ///
    /// Use this when you don't have a `tokenizer_config.json` on hand but
    /// know you're targeting Qwen3-TTS.
    public init() {
        self.specialTokens = BPESpecialTokens(tokens: [
            .init(id: BPESpecialTokens.Qwen3TTS.imStart, content: "<|im_start|>", special: true),
            .init(id: BPESpecialTokens.Qwen3TTS.imEnd, content: "<|im_end|>", special: true),
            .init(id: BPESpecialTokens.Qwen3TTS.ttsPad, content: "tts_pad", special: true),
            .init(id: BPESpecialTokens.Qwen3TTS.ttsBos, content: "tts_bos", special: true),
            .init(id: BPESpecialTokens.Qwen3TTS.ttsEos, content: "tts_eos", special: true),
        ])
    }

    // MARK: - Template formatting

    /// Formats text into the Qwen3-TTS chat template string.
    ///
    /// - Parameter text: The text content to speak.
    /// - Returns: The formatted template string.
    public func format(_ text: String) -> String {
        "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// Encodes the chat template into token IDs.
    ///
    /// The template is split into segments: special tokens are resolved
    /// to their fixed IDs, and text segments are encoded through the
    /// provided BPE encoder.
    ///
    /// - Parameters:
    ///   - text: The text content to speak.
    ///   - encoder: A BPE encoder for tokenizing the text content.
    /// - Throws: ``TokenizerInputError/inputTooLong(characterCount:limit:)`` when
    ///   any encoded text segment exceeds the BPE input limit.
    /// - Returns: An array of token IDs representing the full template.
    public func encode(_ text: String, using encoder: inout BPEEncoder) throws -> [Int] {
        var ids: [Int] = []

        // <|im_start|>
        ids.append(BPESpecialTokens.Qwen3TTS.imStart)

        // "assistant\n{text}"
        ids.append(contentsOf: try encoder.encode("assistant\n\(text)"))

        // <|im_end|>
        ids.append(BPESpecialTokens.Qwen3TTS.imEnd)

        // "\n"
        ids.append(contentsOf: try encoder.encode("\n"))

        // <|im_start|>
        ids.append(BPESpecialTokens.Qwen3TTS.imStart)

        // "assistant\n"
        ids.append(contentsOf: try encoder.encode("assistant\n"))

        return ids
    }

    /// Returns the token IDs for the TTS control prefix: `[tts_bos]`.
    public var ttsBosTokenID: Int {
        BPESpecialTokens.Qwen3TTS.ttsBos
    }

    /// Returns the token ID for `tts_eos`.
    public var ttsEosTokenID: Int {
        BPESpecialTokens.Qwen3TTS.ttsEos
    }

    /// Returns the token ID for `tts_pad`.
    public var ttsPadTokenID: Int {
        BPESpecialTokens.Qwen3TTS.ttsPad
    }
}
