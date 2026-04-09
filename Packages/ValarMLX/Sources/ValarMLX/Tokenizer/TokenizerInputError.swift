import Foundation

/// Errors thrown when tokenizer input fails validation.
///
/// These are the only hard rejections the tokenizer imposes before any BPE
/// work is attempted, so callers can distinguish user-fixable input problems
/// from internal encoding failures.
public enum TokenizerInputError: Error, Sendable, Equatable {

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

extension TokenizerInputError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case let .inputTooLong(count, limit):
            return "Input too long: \(count) characters exceeds the \(limit)-character limit."
        case let .rateLimitExceeded(retryAfter):
            let seconds = String(format: "%.1f", retryAfter)
            return "Rate limit exceeded. Retry after \(seconds) seconds."
        }
    }
}
