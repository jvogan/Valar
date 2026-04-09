import Foundation
import ValarModelKit

/// Native Qwen3 tokenizer handle conforming to ``TokenizerWorkflow``.
///
/// Wraps ``BPEEncoder`` and ``Qwen3ChatTemplate`` to provide tokenization
/// that matches mlx-audio-swift output exactly. The actor boundary serializes
/// access to the mutable BPE encoder cache.
///
/// Initialization options:
/// - From a model directory containing `vocab.json`, `merges.txt`, and
///   `tokenizer_config.json`.
/// - From pre-loaded ``BPEVocabulary`` and optional ``BPESpecialTokens``.
public actor Qwen3TokenizerHandle: TokenizerWorkflow {
    public nonisolated let descriptor: ModelDescriptor
    public nonisolated let backendKind: BackendKind = .mlx

    private var encoder: BPEEncoder
    private let chatTemplate: Qwen3ChatTemplate

    /// Creates a handle from a model directory on disk.
    ///
    /// Expects the directory to contain `vocab.json`, `merges.txt`, and
    /// `tokenizer_config.json`.
    ///
    /// - Parameters:
    ///   - descriptor: The model descriptor for this tokenizer.
    ///   - modelDirectory: Path to the model directory.
    /// - Throws: ``BPELoadError`` if required files are missing or malformed.
    public init(descriptor: ModelDescriptor, modelDirectory: URL) throws {
        self.descriptor = descriptor
        let vocabulary = try BPEVocabulary(directory: modelDirectory)
        self.encoder = BPEEncoder(vocabulary: vocabulary)
        let specialTokens = try BPESpecialTokens(directory: modelDirectory)
        self.chatTemplate = Qwen3ChatTemplate(specialTokens: specialTokens)
    }

    /// Creates a handle from pre-loaded components.
    ///
    /// - Parameters:
    ///   - descriptor: The model descriptor for this tokenizer.
    ///   - vocabulary: A loaded BPE vocabulary.
    ///   - specialTokens: Special tokens for the chat template. When `nil`,
    ///     defaults to the well-known Qwen3-TTS special token IDs.
    public init(
        descriptor: ModelDescriptor,
        vocabulary: BPEVocabulary,
        specialTokens: BPESpecialTokens? = nil
    ) {
        self.descriptor = descriptor
        self.encoder = BPEEncoder(vocabulary: vocabulary)
        if let specialTokens {
            self.chatTemplate = Qwen3ChatTemplate(specialTokens: specialTokens)
        } else {
            self.chatTemplate = Qwen3ChatTemplate()
        }
    }

    // MARK: - TokenizerWorkflow

    public func tokenize(request: TokenizationRequest) async throws -> TokenizationResult {
        let ids = try chatTemplate.encode(request.text, using: &encoder)
        return TokenizationResult(
            model: request.model,
            tokenCount: ids.count,
            chunkCount: 1
        )
    }

    // MARK: - Extended API

    /// Tokenizes text and returns the full token ID sequence.
    ///
    /// Unlike ``tokenize(request:)`` which returns only counts, this method
    /// returns the actual token IDs — useful for downstream inference or
    /// parity testing against mlx-audio-swift.
    ///
    /// - Parameter text: The text to tokenize.
    /// - Throws: ``TokenizerInputError`` when the text exceeds the BPE input limit.
    /// - Returns: The complete token ID sequence including chat template markers.
    public func encode(_ text: String) throws -> [Int] {
        try chatTemplate.encode(text, using: &encoder)
    }

    /// Decodes a sequence of token IDs back to text.
    ///
    /// Only decodes BPE tokens — special token IDs (e.g. `<|im_start|>`)
    /// are silently skipped by the underlying ``BPEEncoder``.
    ///
    /// - Parameter ids: Token IDs to decode.
    /// - Returns: The decoded text, or `nil` if the byte sequence is invalid UTF-8.
    public func decode(_ ids: [Int]) -> String? {
        encoder.decode(ids)
    }
}
