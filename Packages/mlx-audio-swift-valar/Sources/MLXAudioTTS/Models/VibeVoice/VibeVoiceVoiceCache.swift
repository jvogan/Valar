@preconcurrency import MLX
import Foundation
import os

// MARK: - KV Cache Snapshot

/// A precomputed KV-cache snapshot for a single VibeVoice preset voice.
///
/// Each voice ships as a `voices/<name>.safetensors` file containing tensors
/// for `language_model` and `tts_language_model` KV caches plus hidden states
/// for positive and negative (unconditioned) paths.
///
/// Tensor layout per voice file:
/// ```
/// lm_hidden                     [1, seq_len, 896]
/// lm_key_{i}   (i=0..3)        [1, kv_heads=2, seq_len, head_dim=64]
/// lm_value_{i} (i=0..3)        [1, kv_heads=2, seq_len, head_dim=64]
///
/// tts_lm_hidden                 [1, seq_len, 896]
/// tts_lm_key_{i}  (i=0..19)    [1, 2, seq_len, 64]
/// tts_lm_value_{i}(i=0..19)    [1, 2, seq_len, 64]
///
/// neg_tts_lm_hidden             [1, seq_len, 896]
/// neg_tts_lm_key_{i}  (i=0..19)
/// neg_tts_lm_value_{i}(i=0..19)
///
/// neg_lm_key_{i}   (optional, i=0..3)
/// neg_lm_value_{i} (optional, i=0..3)
/// ```
///
/// Storage: `(B, kv_heads, seq, head_dim)`. The Python reference transposes
/// to `(B, seq, kv_heads, head_dim)` on load — Swift callers should match
/// their attention implementation's expected cache layout.
public struct VibeVoiceKVSnapshot: Sendable {
    /// Last hidden state from `language_model` output — shape `[1, seq_len, hidden_size]`.
    public let lmHidden: MLXArray

    /// KV cache pairs for `language_model` layers (4 layers).
    /// Each tuple is `(key, value)` with shape `[1, kv_heads, seq_len, head_dim]`.
    public let lmCache: [(key: MLXArray, value: MLXArray)]

    /// Last hidden state from `tts_language_model` (positive path) — shape `[1, seq_len, hidden_size]`.
    public let ttsLmHidden: MLXArray

    /// KV cache pairs for `tts_language_model` layers (20 layers, positive path).
    public let ttsLmCache: [(key: MLXArray, value: MLXArray)]

    /// Last hidden state from the negative (unconditioned) `tts_language_model` path.
    public let negTtsLmHidden: MLXArray

    /// KV cache pairs for the negative `tts_language_model` path (20 layers).
    public let negTtsLmCache: [(key: MLXArray, value: MLXArray)]

    /// KV cache pairs for the negative `language_model` path (optional, 4 layers).
    /// Some voice files omit these tensors.
    public let negLmCache: [(key: MLXArray, value: MLXArray)]?

    /// Estimated memory footprint in bytes for all tensors in this snapshot.
    public var estimatedBytes: Int {
        var total = lmHidden.nbytes
        total += lmCache.reduce(0) { $0 + $1.key.nbytes + $1.value.nbytes }
        total += ttsLmHidden.nbytes
        total += ttsLmCache.reduce(0) { $0 + $1.key.nbytes + $1.value.nbytes }
        total += negTtsLmHidden.nbytes
        total += negTtsLmCache.reduce(0) { $0 + $1.key.nbytes + $1.value.nbytes }
        if let negLm = negLmCache {
            total += negLm.reduce(0) { $0 + $1.key.nbytes + $1.value.nbytes }
        }
        return total
    }
}

// MARK: - Voice Cache Loading

/// Errors that can occur when loading VibeVoice voice caches.
public enum VibeVoiceVoiceCacheError: Error, CustomStringConvertible {
    case voiceDirectoryNotFound(URL)
    case voiceFileNotFound(String, URL)
    case missingTensor(String, voiceName: String)

    public var description: String {
        switch self {
        case .voiceDirectoryNotFound(let url):
            return "Voice directory not found at: \(url.path)"
        case .voiceFileNotFound(let name, let url):
            return "Voice file not found for '\(name)' at: \(url.path)"
        case .missingTensor(let key, let voiceName):
            return "Voice '\(voiceName)' is missing required tensor: \(key)"
        }
    }
}

/// Load a single voice cache from a `.safetensors` file.
///
/// - Parameters:
///   - url: Path to the `.safetensors` voice file.
///   - config: Model config used to determine layer counts.
/// - Returns: A `VibeVoiceKVSnapshot` with all KV cache tensors loaded.
/// - Throws: `VibeVoiceVoiceCacheError` if a required tensor is missing.
public func loadVoiceCache(
    from url: URL,
    config: VibeVoiceModelConfig
) throws -> VibeVoiceKVSnapshot {
    let voiceName = url.deletingPathExtension().lastPathComponent
    let tensors = try MLX.loadArrays(url: url)

    let lmLayers = config.languageModelLayers
    let ttsLayers = config.ttsLanguageModelLayers

    func required(_ key: String) throws -> MLXArray {
        guard let tensor = tensors[key] else {
            throw VibeVoiceVoiceCacheError.missingTensor(key, voiceName: voiceName)
        }
        return tensor
    }

    func loadKV(prefix: String, index: Int) throws -> (key: MLXArray, value: MLXArray) {
        let k = try required("\(prefix)_key_\(index)")
        let v = try required("\(prefix)_value_\(index)")
        return (key: k, value: v)
    }

    let lmHidden = try required("lm_hidden")
    let ttsLmHidden = try required("tts_lm_hidden")
    let negTtsLmHidden = try required("neg_tts_lm_hidden")

    let lmCache = try (0..<lmLayers).map { try loadKV(prefix: "lm", index: $0) }
    let ttsLmCache = try (0..<ttsLayers).map { try loadKV(prefix: "tts_lm", index: $0) }
    let negTtsLmCache = try (0..<ttsLayers).map { try loadKV(prefix: "neg_tts_lm", index: $0) }

    // Negative LM cache is optional — some voice files omit it
    let hasNegLm = (0..<lmLayers).allSatisfy { i in
        tensors["neg_lm_key_\(i)"] != nil && tensors["neg_lm_value_\(i)"] != nil
    }
    let negLmCache: [(key: MLXArray, value: MLXArray)]? = hasNegLm
        ? try (0..<lmLayers).map { try loadKV(prefix: "neg_lm", index: $0) }
        : nil

    return VibeVoiceKVSnapshot(
        lmHidden: lmHidden,
        lmCache: lmCache,
        ttsLmHidden: ttsLmHidden,
        ttsLmCache: ttsLmCache,
        negTtsLmHidden: negTtsLmHidden,
        negTtsLmCache: negTtsLmCache,
        negLmCache: negLmCache
    )
}

/// Load all preset voices from a model's `voices/` directory.
///
/// Scans for `.safetensors` files and returns a dictionary keyed by voice name
/// (filename without extension, e.g. `"en-Emma_woman"`).
///
/// - Parameters:
///   - directory: The model pack root directory containing a `voices/` subdirectory.
///   - config: Model config used to determine layer counts.
/// - Returns: Dictionary mapping voice name to its KV snapshot.
/// - Throws: `VibeVoiceVoiceCacheError` if the voices directory is missing,
///           or if any voice file has missing tensors.
public func loadPresetVoices(
    from directory: URL,
    config: VibeVoiceModelConfig
) throws -> [String: VibeVoiceKVSnapshot] {
    let voicesDir = directory.appendingPathComponent("voices")
    let fm = FileManager.default

    guard fm.fileExists(atPath: voicesDir.path) else {
        throw VibeVoiceVoiceCacheError.voiceDirectoryNotFound(voicesDir)
    }

    let contents = try fm.contentsOfDirectory(
        at: voicesDir,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    )

    let safetensorFiles = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    var voices: [String: VibeVoiceKVSnapshot] = [:]
    voices.reserveCapacity(safetensorFiles.count)

    for file in safetensorFiles {
        let name = file.deletingPathExtension().lastPathComponent
        voices[name] = try loadVoiceCache(from: file, config: config)
    }

    return voices
}

private let vibeVoiceLogger = Logger(subsystem: "com.valar.tts", category: "VibeVoiceCache")

/// Result of a graceful bulk voice load. Includes successfully loaded voices
/// and a list of names that failed (with warnings logged).
public struct VibeVoiceBulkLoadResult: Sendable {
    /// Successfully loaded voices keyed by name.
    public let voices: [String: VibeVoiceKVSnapshot]
    /// Names of voices that failed to load (missing file or corrupt tensors).
    public let skipped: [String]
    /// Total estimated memory footprint of all loaded voices in bytes.
    public var totalEstimatedBytes: Int {
        voices.values.reduce(0) { $0 + $1.estimatedBytes }
    }
}

/// Eagerly load all preset voices from a model's `voices/` directory,
/// gracefully skipping any that are missing or corrupt.
///
/// Logs a warning for each skipped voice rather than throwing, so that
/// partial voice sets don't prevent model startup.
///
/// - Parameters:
///   - directory: The model pack root directory containing a `voices/` subdirectory.
///   - config: Model config used to determine layer counts.
/// - Returns: A `VibeVoiceBulkLoadResult` with loaded voices and skipped names.
public func loadPresetVoicesGracefully(
    from directory: URL,
    config: VibeVoiceModelConfig
) -> VibeVoiceBulkLoadResult {
    let voicesDir = directory.appendingPathComponent("voices")
    let fm = FileManager.default

    guard fm.fileExists(atPath: voicesDir.path) else {
        vibeVoiceLogger.warning("VibeVoice voices directory not found at \(voicesDir.path)")
        return VibeVoiceBulkLoadResult(voices: [:], skipped: [])
    }

    let contents: [URL]
    do {
        contents = try fm.contentsOfDirectory(
            at: voicesDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
    } catch {
        vibeVoiceLogger.warning("Failed to enumerate voices directory: \(error.localizedDescription)")
        return VibeVoiceBulkLoadResult(voices: [:], skipped: [])
    }

    let safetensorFiles = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    var voices: [String: VibeVoiceKVSnapshot] = [:]
    voices.reserveCapacity(safetensorFiles.count)
    var skipped: [String] = []

    for file in safetensorFiles {
        let name = file.deletingPathExtension().lastPathComponent
        do {
            voices[name] = try loadVoiceCache(from: file, config: config)
        } catch {
            vibeVoiceLogger.warning("Skipping voice '\(name)': \(error.localizedDescription)")
            skipped.append(name)
        }
    }

    return VibeVoiceBulkLoadResult(voices: voices, skipped: skipped)
}
