import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXNN
import ValarModelKit

/// Bridges mlx-audio-swift streaming generation events to ValarTTS AudioChunk streams.
///
/// When mlx-audio-swift is linked, this converts the `AsyncThrowingStream<GenerateEvent, Error>`
/// from `model.generateStream()` into `AsyncThrowingStream<AudioChunk, Error>` that ValarTTS
/// audio pipeline and UI components consume.
///
/// The bridge supports two decode paths:
/// - **Standard path**: Consumes `.audio(MLXArray)` events from the upstream model's built-in
///   speech tokenizer decoder.
/// - **Native decoder path**: Decodes raw speech tokenizer codes `[batch, numQuantizers, time]`
///   using ValarTTS's own `SpeechTokenizerDecoder`, bypassing the upstream decoder. This path
///   activates when a loaded native decoder is provided and code arrays are available.
public enum MLXStreamBridge {

    /// Sample rate produced by the native speech tokenizer decoder (24 kHz).
    static let nativeDecoderSampleRate = 24_000

    /// Number of per-step code frames accumulated before calling the native decoder.
    /// 25 frames × 80 ms/frame = 2 s of audio per decode chunk.
    static let codeStreamChunkSize = 25

    /// Low-latency interval used by the standalone MLX VibeVoice tool for earlier
    /// chunk emission on the daemon streaming surface.
    static let vibeVoiceStreamingInterval = 0.32

    /// Feature gate for the native decoder path in `stream(from:request:nativeDecoder:)`.
    ///
    /// When `false` (default), the method falls through to the standard `.audio`-event path
    /// even when a `SpeechTokenizerDecoder` is supplied. Flip to `true` to enable live
    /// routing through the native decoder once the code stream has been validated in
    /// production.
    public static let useNativeDecoderPath: Bool = ProcessInfo.processInfo.environment["VALAR_NATIVE_DECODER"] == "1"

    // MARK: - Standard Path

    public static func stream(
        from model: any SpeechGenerationModel,
        descriptor: ModelDescriptor,
        request: SpeechSynthesisRequest
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)
        let upstream = upstreamStream(from: model, request: request, input: input)

        return stream(
            from: sampleStream(from: upstream),
            sampleRate: model.sampleRate
        )
    }

    // MARK: - Native Decoder Fast Path

    /// Streams audio from the model, optionally routing through the native decoder.
    ///
    /// When `nativeDecoder` is non-nil and the upstream generation exposes raw codec
    /// codes, the native decoder decodes the waveform directly — bypassing the upstream
    /// library's speech tokenizer decoder. When codes are not available (the current
    /// upstream API decodes internally), this falls back to the standard `.audio` event
    /// path transparently.
    ///
    /// - Parameters:
    ///   - model: The upstream speech generation model.
    ///   - request: The synthesis request.
    ///   - nativeDecoder: A loaded `SpeechTokenizerDecoder`, or `nil` to use the standard path.
    /// - Returns: A stream of `AudioChunk` values at the model's sample rate (standard path)
    ///   or 24 kHz (native decoder path).
    static func stream(
        from model: any SpeechGenerationModel,
        descriptor: ModelDescriptor,
        request: SpeechSynthesisRequest,
        nativeDecoder: SpeechTokenizerDecoder?
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        guard let nativeDecoder, Self.useNativeDecoderPath else {
            return stream(from: model, descriptor: descriptor, request: request)
        }
        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)
        let perStepFrames = upstreamCodeStream(from: model, request: request, input: input)
        let batched = batchCodeFrames(from: perStepFrames, chunkSize: Self.codeStreamChunkSize)
        return nativeDecoderStream(from: batched, decoder: nativeDecoder)
    }

    private static func upstreamStream(
        from model: any SpeechGenerationModel,
        request: SpeechSynthesisRequest,
        input: MLXModelHandle.ResolvedSpeechGenerationInput
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let generationParameters = MLXModelHandle.generationParameters(
            from: request,
            defaults: model.defaultGenerationParameters
        )
        if let qwenModel = model as? Qwen3TTSModel {
            return qwenModel.generateStream(
                text: request.text,
                speaker: input.speaker,
                instruct: input.instruct,
                conditioning: input.conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: request.language,
                generationParameters: generationParameters
            )
        }
        if let vibeVoiceModel = model as? VibeVoiceTTSModel {
            return vibeVoiceModel.generateStream(
                text: request.text,
                voice: input.voice,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: request.language,
                generationParameters: generationParameters,
                streamingInterval: Self.vibeVoiceStreamingInterval
            )
        }
        if let conditioning = input.conditioning {
            guard let conditionedModel = model as? any ConditionedSpeechGenerationModel else {
                return AsyncThrowingStream { continuation in
                    continuation.finish(
                        throwing: MLXBackendError.inferenceError(
                            "Saved voice conditioning '\(conditioning.format)' is not supported by this model."
                        )
                    )
                }
            }
            return conditionedModel.generateStream(
                text: request.text,
                voice: input.voice,
                conditioning: conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: request.language,
                generationParameters: generationParameters
            )
        }

        return model.generateStream(
            text: request.text,
            voice: input.voice,
            refAudio: input.referenceAudio,
            refText: input.referenceText,
            language: request.language,
            generationParameters: generationParameters
        )
    }

    private static func upstreamCodeStream(
        from model: any SpeechGenerationModel,
        request: SpeechSynthesisRequest,
        input: MLXModelHandle.ResolvedSpeechGenerationInput
    ) -> AsyncThrowingStream<MLXArray, Error> {
        let generationParameters = MLXModelHandle.generationParameters(
            from: request,
            defaults: model.defaultGenerationParameters
        )
        if let qwenModel = model as? Qwen3TTSModel {
            return qwenModel.generateCodeStream(
                text: request.text,
                speaker: input.speaker,
                instruct: input.instruct,
                conditioning: input.conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: request.language,
                generationParameters: generationParameters
            )
        }
        if let conditioning = input.conditioning {
            guard let conditionedModel = model as? any ConditionedSpeechGenerationModel else {
                return AsyncThrowingStream { continuation in
                    continuation.finish(
                        throwing: MLXBackendError.inferenceError(
                            "Saved voice conditioning '\(conditioning.format)' is not supported by this model."
                        )
                    )
                }
            }
            return conditionedModel.generateCodeStream(
                text: request.text,
                voice: input.voice,
                conditioning: conditioning,
                refAudio: input.referenceAudio,
                refText: input.referenceText,
                language: request.language,
                generationParameters: generationParameters
            )
        }

        return model.generateCodeStream(
            text: request.text,
            voice: input.voice,
            refAudio: input.referenceAudio,
            refText: input.referenceText,
            language: request.language,
            generationParameters: generationParameters
        )
    }

    /// Decodes a stream of code-frame arrays into AudioChunks using the native decoder.
    ///
    /// Each element of `codeFrames` must be an `MLXArray` with shape
    /// `[batch, numQuantizers, time]`. The decoder runs chunked decode on each frame
    /// and yields one `AudioChunk` per frame at 24 kHz.
    ///
    /// - Parameters:
    ///   - codeFrames: Async stream of code arrays from a generation loop.
    ///   - decoder: A loaded `SpeechTokenizerDecoder` with weights applied.
    ///   - sampleRate: Output sample rate (default 24 kHz).
    /// - Returns: A stream of decoded `AudioChunk` values.
    internal static func nativeDecoderStream(
        from codeFrames: AsyncThrowingStream<MLXArray, Error>,
        decoder: SpeechTokenizerDecoder,
        sampleRate: Int = nativeDecoderSampleRate
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        // Safety: SpeechTokenizerDecoder (an MLXNN.Module) is not Sendable, but the
        // decoder is only accessed from a single Task below — no concurrent mutation.
        nonisolated(unsafe) let decoder = decoder
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    for try await codes in codeFrames {
                        let chunk = Self.decodeCodesSync(
                            codes, using: decoder, sampleRate: sampleRate
                        )
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Accumulates per-step code frames `[1, numCodeGroups]` into batched chunks
    /// `[1, numCodeGroups, chunkSize]` suitable for `nativeDecoderStream`.
    ///
    /// Frames are collected until `chunkSize` is reached, then stacked and transposed
    /// into `[1, numCodeGroups, time]` and yielded. Any remaining frames at stream
    /// completion are flushed as a final partial chunk.
    ///
    /// - Parameters:
    ///   - perStepStream: Per-step code arrays with shape `[1, numCodeGroups]`, one per
    ///     generation step (e.g., from `SpeechGenerationModel.generateCodeStream`).
    ///   - chunkSize: Number of frames per batch. Defaults to `codeStreamChunkSize`.
    /// - Returns: Stream of batched arrays with shape `[1, numCodeGroups, time]`.
    internal static func batchCodeFrames(
        from perStepStream: AsyncThrowingStream<MLXArray, Error>,
        chunkSize: Int = codeStreamChunkSize
    ) -> AsyncThrowingStream<MLXArray, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    var buffer: [MLXArray] = []
                    buffer.reserveCapacity(chunkSize)
                    for try await frame in perStepStream {
                        buffer.append(frame)
                        if buffer.count >= chunkSize {
                            // stacked([1, nQ], axis: 1) → [1, time, nQ]; transposed → [1, nQ, time]
                            let batch = stacked(buffer, axis: 1).transposed(0, 2, 1)
                            MLX.eval(batch)
                            continuation.yield(batch)
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    if !buffer.isEmpty {
                        let batch = stacked(buffer, axis: 1).transposed(0, 2, 1)
                        MLX.eval(batch)
                        continuation.yield(batch)
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

    /// Decodes a complete codes tensor into a single `AudioChunk` using the native decoder.
    ///
    /// - Parameters:
    ///   - codes: Integer codes with shape `[batch, numQuantizers, time]`.
    ///   - decoder: A loaded `SpeechTokenizerDecoder` with weights applied.
    ///   - sampleRate: Output sample rate (default 24 kHz).
    ///   - chunked: Whether to use chunked decode for long sequences (default `true`).
    /// - Returns: A single `AudioChunk` containing the decoded waveform.
    internal static func decodeCodes(
        _ codes: MLXArray,
        using decoder: SpeechTokenizerDecoder,
        sampleRate: Int = nativeDecoderSampleRate,
        chunked: Bool = true
    ) -> AudioChunk {
        decodeCodesSync(codes, using: decoder, sampleRate: sampleRate, chunked: chunked)
    }

    // MARK: - Internal Stream Helpers

    internal static func stream(
        from upstream: AsyncThrowingStream<[Float], Error>,
        sampleRate: Int
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    for try await samples in upstream {
                        continuation.yield(
                            MLXModelHandle.audioChunk(from: samples, sampleRate: sampleRate)
                        )
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    internal static func sampleStream(
        from upstream: AsyncThrowingStream<AudioGeneration, Error>
    ) -> AsyncThrowingStream<[Float], Error> {
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    for try await event in upstream {
                        guard case .audio(let audio) = event else { continue }
                        continuation.yield(audio.asArray(Float.self))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    public static func placeholderStream() -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }

    // MARK: - Private

    private static func decodeCodesSync(
        _ codes: MLXArray,
        using decoder: SpeechTokenizerDecoder,
        sampleRate: Int = nativeDecoderSampleRate,
        chunked: Bool = true
    ) -> AudioChunk {
        let wav: MLXArray = chunked
            ? decoder.chunkedDecode(codes)
            : decoder(codes)
        let samples = wav.squeezed().asArray(Float.self)
        return MLXModelHandle.audioChunk(from: samples, sampleRate: sampleRate)
    }
}
