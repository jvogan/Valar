@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

public struct SpeechConditioning: Sendable, Equatable, Hashable {
    public let format: String
    public let payload: Data
    /// Named binary assets for asset-backed conditioning formats (e.g. TADA reference bundles).
    /// Keyed by filename (e.g. "token_values.f16"). Empty for payload-only formats like Qwen.
    public let namedAssets: [String: Data]

    public init(format: String, payload: Data) {
        self.format = format
        self.payload = payload
        self.namedAssets = [:]
    }

    public init(format: String, payload: Data, namedAssets: [String: Data]) {
        self.format = format
        self.payload = payload
        self.namedAssets = namedAssets
    }
}

public struct AudioGenerationHeartbeat: Sendable, Equatable {
    public let generatedTokenCount: Int
    public let maxTokens: Int
    public let wallTimeSeconds: Double

    public init(
        generatedTokenCount: Int,
        maxTokens: Int,
        wallTimeSeconds: Double
    ) {
        self.generatedTokenCount = generatedTokenCount
        self.maxTokens = maxTokens
        self.wallTimeSeconds = wallTimeSeconds
    }
}

public enum AudioGenerationObserverContext {
    @TaskLocal public static var heartbeatHandler: (@Sendable (AudioGenerationHeartbeat) -> Void)?
    @TaskLocal public static var infoHandler: (@Sendable (AudioGenerationInfo) -> Void)?
}

public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }
    var defaultGenerationParameters: GenerateParameters { get }

    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    /// Stream raw codec frames (shape [1, numCodeGroups] per step) for native decoding.
    /// Default implementation returns an empty stream for models that don't support it.
    func generateCodeStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error>
}

public protocol ConditionedSpeechGenerationModel: SpeechGenerationModel {
    func generate(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray

    func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    func generateStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error>

    func generateCodeStream(
        text: String,
        voice: String?,
        conditioning: SpeechConditioning?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error>
}

public extension SpeechGenerationModel {
    func generateCodeStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<MLXArray, Error> {
        // Default: empty stream — models must override to support native decoding
        AsyncThrowingStream { $0.finish() }
    }
    func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil
    ) async throws -> MLXArray {
        try await generate(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateSamplesStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<[Float], Error> {
        let stream = generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters ?? defaultGenerationParameters,
            streamingInterval: streamingInterval
        )
        return proxyAudioStream(stream, extract: {
            guard case .audio(let samples) = $0 else { return nil }
            return samples.asArray(Float.self)
        })
    }

#if canImport(AVFoundation)
    @MainActor
    func generatePCMBufferStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters? = nil,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<AVAudioPCMBuffer, Error> {
        let sampleStream = generateSamplesStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: streamingInterval
        )

        let (stream, continuation) = AsyncThrowingStream<AVAudioPCMBuffer, Error>.makeStream()
        let sampleRate = self.sampleRate

        Task { @MainActor in
            do {
                for try await samples in sampleStream {
                    let buffer = try makePCMBuffer(samples: samples, sampleRate: sampleRate)
                    continuation.yield(buffer)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }
#endif

    func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double = 2.0
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        _ = streamingInterval
        return generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }
}

private func proxyAudioStream<T: Sendable, U: Sendable>(
    _ upstream: AsyncThrowingStream<T, Error>,
    extract: @Sendable @escaping (T) -> U?
) -> AsyncThrowingStream<U, Error> {
    AsyncThrowingStream<U, Error> { continuation in
        let task = Task { @Sendable in
            do {
                for try await value in upstream {
                    guard let extracted = extract(value) else { continue }
                    continuation.yield(extracted)
                }
                continuation.finish()
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
    }
}

#if canImport(AVFoundation)
@MainActor
private func makePCMBuffer(samples: [Float], sampleRate: Int) throws -> AVAudioPCMBuffer {
    let frameCount = AVAudioFrameCount(samples.count)
    guard
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ),
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
        let channel = buffer.floatChannelData?[0]
    else {
        throw AudioGenerationError.audioDecodingFailed("Failed to create AVAudioPCMBuffer")
    }

    buffer.frameLength = frameCount
    for i in 0 ..< samples.count {
        channel[i] = samples[i]
    }
    return buffer
}
#endif
