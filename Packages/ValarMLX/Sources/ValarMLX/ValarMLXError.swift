import Foundation
import ValarModelKit

/// Unified error type for all ValarMLX operations.
///
/// All throwing functions in the ValarMLX package surface errors as
/// `ValarMLXError` so callers can handle the full range of failure modes
/// with a single `catch` clause.
///
/// Cases are organized into logical groups:
///
/// - **Backend / model loading**: Errors from `InferenceBackend` implementations.
/// - **Adapter validation**: Errors from `ModelAdapter` implementations when
///   manifests do not satisfy the adapter's constraints.
/// - **Weight file security**: Errors from model-directory validation before any
///   tensor loading takes place.
/// - **Weight file loading**: Errors encountered while loading serialized weight
///   tensors from the `speech_tokenizer/` subdirectory.
/// - **Tokenizer input**: Errors from tokenizer input validation at the call site.
///
/// `BPELoadError` is the only ValarMLX error that is intentionally kept
/// separate: its associated values include `any Error` (from `Foundation` file
/// I/O), which prevents `Equatable` synthesis and would require lossy conversion.
/// Callers that load vocabulary files should also catch `BPELoadError`.
public enum ValarMLXError: Error, Sendable, Equatable {

    // MARK: - Backend / Model Loading

    /// The backend kind specified in a ``BackendRequirement`` is not handled by
    /// this backend implementation.
    case unsupportedBackend(BackendKind)

    /// The model ID was not found in the backend's loaded-model cache when a
    /// synthesize call required a resident model.
    case modelNotFound(ModelIdentifier)

    /// An unrecoverable error occurred during model inference.
    case inferenceError(String)

    /// An unrecoverable error occurred in a streaming inference pipeline.
    case streamingError(String)

    // MARK: - Adapter Validation

    /// The manifest's family ID does not match the family this adapter handles.
    case familyMismatch(expected: ModelFamilyID, got: ModelFamilyID)

    /// The manifest specifies no weight artifacts, which are required for
    /// loading and running the model.
    case missingArtifacts(String)

    /// The requested surface or workflow is not supported by this adapter.
    case unsupportedSurface(String)

    // MARK: - Weight File Security

    /// A file with a rejected extension was found inside the model directory.
    ///
    /// Only `.safetensors` weight files are permitted. Other formats (`.bin`,
    /// `.pkl`, `.pt`) can carry unsafe serialized content and are hard-rejected
    /// before any upstream tensor loader sees the directory.
    case rejectedUnsafeWeightFile(path: String, fileExtension: String)

    /// A `.safetensors` file does not have the expected header magic bytes or
    /// its declared header length is inconsistent with the file size.
    case invalidSafeTensorsHeader(String)

    /// The model directory contains no root-level `.safetensors` weight files.
    ///
    /// Nested safetensors (e.g. in `speech_tokenizer/`) do not satisfy this
    /// requirement — at least one file in the root directory is required.
    case missingSafeTensorsWeights(String)

    /// A path component, symlink, or `..` component resolves outside the
    /// declared directory sandbox.
    ///
    /// Thrown when the canonicalized path of a weight file does not start with
    /// the canonicalized path of the model directory, indicating an attempt to
    /// load files from an arbitrary filesystem location.
    case pathTraversalDetected(String)

    // MARK: - Weight File Loading

    /// The expected weight directory does not exist on disk.
    case weightDirectoryNotFound(String)

    /// The weight directory exists but contains no `.safetensors` files.
    case noSafetensorsFiles(String)

    /// A loaded weight tensor has a shape that does not match the model
    /// configuration's expected layout.
    case shapeMismatch(key: String, expected: [Int], actual: [Int])

    // MARK: - Tokenizer Input

    /// The input text exceeds the per-request character limit.
    ///
    /// The BPE merge loop is O(n²) in the number of pre-tokenized symbols.
    /// Rejecting oversized input early prevents runaway CPU consumption from
    /// adversarially long strings.
    ///
    /// - Parameters:
    ///   - characterCount: The number of Unicode scalars in the rejected input.
    ///   - limit: The maximum allowed character count.
    case inputTooLong(characterCount: Int, limit: Int)

    /// The caller has exceeded the per-handle request rate.
    ///
    /// - Parameter retryAfter: Approximate seconds until the window clears and
    ///   another request will be accepted.
    case rateLimitExceeded(retryAfter: TimeInterval)
}

extension ValarMLXError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend(let kind):
            return "Unsupported backend: \(kind.rawValue)"
        case .modelNotFound(let id):
            return "Model not found: \(id.rawValue)"
        case .inferenceError(let message), .streamingError(let message):
            return message
        case .familyMismatch(let expected, let got):
            return "Family mismatch: expected '\(expected.rawValue)', got '\(got.rawValue)'"
        case .missingArtifacts(let message):
            return "Missing artifacts: \(message)"
        case .unsupportedSurface(let message):
            return "Unsupported surface: \(message)"
        case .rejectedUnsafeWeightFile(let path, let ext):
            return "Unsafe model weight format '.\(ext)' rejected at \(path). Only .safetensors weights are allowed."
        case .invalidSafeTensorsHeader(let path):
            return "Invalid safetensors header magic bytes at \(path). Refusing to load the file."
        case .missingSafeTensorsWeights(let directory):
            return "No .safetensors weight files were found in \(directory)."
        case .pathTraversalDetected(let path):
            return "Path traversal detected at \(path). Refusing to load files outside the model directory sandbox."
        case .weightDirectoryNotFound(let path):
            return "Weight directory not found: \(path)"
        case .noSafetensorsFiles(let path):
            return "No .safetensors files found in: \(path)"
        case .shapeMismatch(let key, let expected, let actual):
            return "Shape mismatch for '\(key)': expected \(expected), got \(actual)"
        case .inputTooLong(let count, let limit):
            return "Input too long: \(count) characters exceeds the \(limit)-character limit."
        case .rateLimitExceeded(let retryAfter):
            let seconds = String(format: "%.1f", retryAfter)
            return "Rate limit exceeded. Retry after \(seconds) seconds."
        }
    }
}
