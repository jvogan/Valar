@preconcurrency import AVFoundation
import Foundation
@preconcurrency import Speech
import ValarAudio
import ValarModelKit

public enum AppleSpeechBackendError: LocalizedError, Sendable, Equatable {
    case unsupportedBackend(BackendKind)
    case unsupportedModel(ModelIdentifier)
    case emptyAudio
    case missingSpeechRecognitionUsageDescription
    case speechRecognitionUnavailable(String)
    case speechRecognitionNotAuthorized(String)
    case onDeviceRecognitionUnavailable(String)
    case transcriptionFailed(String)
    case synthesisFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBackend(let backendKind):
            return "Apple speech backend cannot satisfy backend '\(backendKind.rawValue)'."
        case .unsupportedModel(let identifier):
            return "Apple speech backend does not support model '\(identifier.rawValue)'."
        case .emptyAudio:
            return "Apple speech recognition requires a non-empty audio buffer."
        case .missingSpeechRecognitionUsageDescription:
            return "Apple System ASR requires the host app to include NSSpeechRecognitionUsageDescription in its Info.plist. Use the Valar macOS app, or install an MLX ASR model for raw CLI and daemon workflows."
        case .speechRecognitionUnavailable(let locale):
            return "Apple speech recognition is unavailable for locale '\(locale)'."
        case .speechRecognitionNotAuthorized(let status):
            return "Apple speech recognition is not authorized: \(status)."
        case .onDeviceRecognitionUnavailable(let locale):
            return "Apple on-device speech recognition is unavailable for locale '\(locale)'."
        case .transcriptionFailed(let message):
            return "Apple speech recognition failed: \(message)"
        case .synthesisFailed(let message):
            return "Apple speech synthesis failed: \(message)"
        }
    }
}

public enum AppleSpeechPrivacy {
    public static let speechRecognitionUsageDescriptionKey = "NSSpeechRecognitionUsageDescription"

    public static func hostHasSpeechRecognitionUsageDescription(bundle: Bundle = .main) -> Bool {
        guard let value = bundle.object(forInfoDictionaryKey: speechRecognitionUsageDescriptionKey) as? String else {
            return false
        }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

public actor AppleSpeechBackend: InferenceBackend, RuntimeBackendInventory, ModelIDUnloadingInferenceBackend {
    public nonisolated let backendKind: BackendKind = .apple
    public nonisolated let availableBackendKinds: Set<BackendKind> = [.apple]
    public nonisolated let runtimeCapabilities = BackendCapabilities(
        features: [],
        supportedFamilies: [.appleSpeechSynthesis, .appleSpeechRecognition]
    )

    private var loadedModels: [ModelIdentifier: AppleSpeechModelHandle] = [:]

    public init() {}

    public func validate(requirement: BackendRequirement) async throws {
        guard requirement.backendKind == .apple else {
            throw AppleSpeechBackendError.unsupportedBackend(requirement.backendKind)
        }
    }

    public func loadModel(
        descriptor: ModelDescriptor,
        configuration: ModelRuntimeConfiguration
    ) async throws -> any ValarModel {
        guard configuration.backendKind == .apple else {
            throw AppleSpeechBackendError.unsupportedBackend(configuration.backendKind)
        }
        guard descriptor.familyID == .appleSpeechSynthesis || descriptor.familyID == .appleSpeechRecognition else {
            throw AppleSpeechBackendError.unsupportedModel(descriptor.id)
        }
        if let existing = loadedModels[descriptor.id] {
            return existing
        }

        let handle = AppleSpeechModelHandle(descriptor: descriptor)
        loadedModels[descriptor.id] = handle
        return handle
    }

    public func unloadModel(_ model: any ValarModel) async throws {
        loadedModels.removeValue(forKey: model.descriptor.id)
    }

    public func unloadModel(withID identifier: ModelIdentifier) async {
        loadedModels.removeValue(forKey: identifier)
    }
}

public final class AppleSpeechModelHandle: ValarModel, TextToSpeechWorkflow, SpeechToTextWorkflow {
    public let descriptor: ModelDescriptor
    public let backendKind: BackendKind = .apple

    public init(descriptor: ModelDescriptor) {
        self.descriptor = descriptor
    }

    public func synthesize(
        request: SpeechSynthesisRequest,
        in session: ModelRuntimeSession
    ) async throws -> AudioChunk {
        guard descriptor.familyID == .appleSpeechSynthesis else {
            throw AppleSpeechBackendError.unsupportedModel(descriptor.id)
        }
        return try await AppleSpeechSynthesizerRenderer.render(request: request)
    }

    public func synthesizeStream(
        request: SpeechSynthesisRequest
    ) async throws -> AsyncThrowingStream<AudioChunk, Error> {
        let chunk = try await AppleSpeechSynthesizerRenderer.render(request: request)
        return AsyncThrowingStream { continuation in
            continuation.yield(chunk)
            continuation.finish()
        }
    }

    public func transcribe(
        request: SpeechRecognitionRequest,
        in session: ModelRuntimeSession
    ) async throws -> RichTranscriptionResult {
        guard descriptor.familyID == .appleSpeechRecognition else {
            throw AppleSpeechBackendError.unsupportedModel(descriptor.id)
        }
        return try await AppleSpeechRecognizerRenderer.transcribe(request: request)
    }
}

private enum AppleSpeechSynthesizerRenderer {
    static func render(request: SpeechSynthesisRequest) async throws -> AudioChunk {
        let text = request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            return AudioChunk(samples: [], sampleRate: request.sampleRate)
        }

        let utterance = AVSpeechUtterance(string: text)
        if let voice = resolveVoice(for: request) {
            utterance.voice = voice
        }

        let rendered = try await renderUtterance(utterance)
        guard !rendered.samples.isEmpty else {
            throw AppleSpeechBackendError.synthesisFailed("AVSpeechSynthesizer returned no audio.")
        }

        guard rendered.sampleRate != request.sampleRate else {
            return AudioChunk(samples: rendered.samples, sampleRate: rendered.sampleRate)
        }

        let resampler = AccelerateAudioResampler()
        let resampled = try await resampler.resample(
            AudioPCMBuffer(mono: rendered.samples, sampleRate: rendered.sampleRate),
            to: request.sampleRate
        )
        return AudioChunk(samples: resampled.channels.first ?? [], sampleRate: resampled.format.sampleRate)
    }

    private static func resolveVoice(for request: SpeechSynthesisRequest) -> AVSpeechSynthesisVoice? {
        if let identifier = request.voice?.backendVoiceID,
           let voice = AVSpeechSynthesisVoice(identifier: identifier) {
            return voice
        }

        if let identifier = request.voice?.voiceSelector,
           let voice = AVSpeechSynthesisVoice(identifier: identifier) {
            return voice
        }

        if let language = request.voice?.localeIdentifier ?? request.language,
           let voice = AVSpeechSynthesisVoice(language: Locale(identifier: language).identifier) {
            return voice
        }

        return nil
    }

    private static func renderUtterance(_ utterance: AVSpeechUtterance) async throws -> (samples: [Float], sampleRate: Double) {
        try await withCheckedThrowingContinuation { continuation in
            let synthesizer = AVSpeechSynthesizer()
            var samples: [Float] = []
            var sampleRate: Double?
            var didResume = false

            synthesizer.write(utterance) { buffer in
                guard !didResume else { return }
                guard let pcmBuffer = buffer as? AVAudioPCMBuffer else { return }

                if pcmBuffer.frameLength == 0 {
                    didResume = true
                    continuation.resume(returning: (samples, sampleRate ?? 24_000))
                    _ = synthesizer
                    return
                }

                do {
                    let extracted = try extractMonoFloatSamples(from: pcmBuffer)
                    sampleRate = extracted.sampleRate
                    samples.append(contentsOf: extracted.samples)
                } catch {
                    didResume = true
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

private enum AppleSpeechRecognizerRenderer {
    static func transcribe(request: SpeechRecognitionRequest) async throws -> RichTranscriptionResult {
        guard let audio = request.audioChunk, !audio.samples.isEmpty else {
            throw AppleSpeechBackendError.emptyAudio
        }
        guard AppleSpeechPrivacy.hostHasSpeechRecognitionUsageDescription() else {
            throw AppleSpeechBackendError.missingSpeechRecognitionUsageDescription
        }

        let localeIdentifier = request.languageHint
            .map { Locale(identifier: $0).identifier }
            ?? Locale.current.identifier
        let locale = Locale(identifier: localeIdentifier)

        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            throw AppleSpeechBackendError.speechRecognitionUnavailable(localeIdentifier)
        }

        let authorizationStatus = await requestAuthorization()
        guard authorizationStatus == .authorized else {
            throw AppleSpeechBackendError.speechRecognitionNotAuthorized(
                authorizationStatusDescription(authorizationStatus)
            )
        }

        guard recognizer.supportsOnDeviceRecognition else {
            throw AppleSpeechBackendError.onDeviceRecognitionUnavailable(localeIdentifier)
        }

        let temporaryURL = try writeTemporaryWAV(audio)
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }

        let recognitionRequest = SFSpeechURLRecognitionRequest(url: temporaryURL)
        recognitionRequest.requiresOnDeviceRecognition = true
        recognitionRequest.shouldReportPartialResults = false

        let startedAt = Date()
        let result = try await recognize(request: recognitionRequest, recognizer: recognizer)
        let transcript = result.bestTranscription.formattedString
        let duration = Double(audio.samples.count) / audio.sampleRate

        return RichTranscriptionResult(
            text: transcript,
            language: localeIdentifier,
            durationSeconds: duration,
            segments: [
                TranscriptionSegment(
                    text: transcript,
                    startTime: 0,
                    endTime: duration,
                    confidence: nil,
                    isFinal: true
                ),
            ],
            backendMetadata: BackendMetadata(
                modelId: request.model.rawValue,
                backendKind: .apple,
                inferenceTimeSeconds: Date().timeIntervalSince(startedAt)
            )
        )
    }

    private static func requestAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }

    private static func recognize(
        request: SFSpeechURLRecognitionRequest,
        recognizer: SFSpeechRecognizer
    ) async throws -> SFSpeechRecognitionResult {
        try await withCheckedThrowingContinuation { continuation in
            var didResume = false
            recognizer.recognitionTask(with: request) { result, error in
                guard !didResume else { return }
                if let result, result.isFinal {
                    didResume = true
                    continuation.resume(returning: result)
                    return
                }
                if let error {
                    didResume = true
                    continuation.resume(
                        throwing: AppleSpeechBackendError.transcriptionFailed(error.localizedDescription)
                    )
                }
            }
        }
    }

    private static func writeTemporaryWAV(_ audio: AudioChunk) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("valar-apple-asr-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: audio.sampleRate,
            channels: 1,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(audio.samples.count)
        ) else {
            throw AppleSpeechBackendError.transcriptionFailed("Could not create temporary PCM buffer.")
        }

        buffer.frameLength = AVAudioFrameCount(audio.samples.count)
        audio.samples.withUnsafeBufferPointer { source in
            if let baseAddress = source.baseAddress {
                buffer.floatChannelData?[0].update(from: baseAddress, count: audio.samples.count)
            }
        }

        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }

    private static func authorizationStatusDescription(_ status: SFSpeechRecognizerAuthorizationStatus) -> String {
        switch status {
        case .notDetermined:
            return "not determined"
        case .denied:
            return "denied"
        case .restricted:
            return "restricted"
        case .authorized:
            return "authorized"
        @unknown default:
            return "unknown"
        }
    }
}

private func extractMonoFloatSamples(from buffer: AVAudioPCMBuffer) throws -> (samples: [Float], sampleRate: Double) {
    let frameCount = Int(buffer.frameLength)
    guard frameCount > 0 else {
        return ([], buffer.format.sampleRate)
    }

    let channelCount = max(1, Int(buffer.format.channelCount))
    if let floatChannelData = buffer.floatChannelData {
        var mono = Array(repeating: Float(0), count: frameCount)
        for channel in 0..<channelCount {
            let source = floatChannelData[channel]
            for frame in 0..<frameCount {
                mono[frame] += source[frame] / Float(channelCount)
            }
        }
        return (mono, buffer.format.sampleRate)
    }

    if let int16ChannelData = buffer.int16ChannelData {
        var mono = Array(repeating: Float(0), count: frameCount)
        for channel in 0..<channelCount {
            let source = int16ChannelData[channel]
            for frame in 0..<frameCount {
                mono[frame] += (Float(source[frame]) / Float(Int16.max)) / Float(channelCount)
            }
        }
        return (mono, buffer.format.sampleRate)
    }

    throw AppleSpeechBackendError.synthesisFailed(
        "Unsupported AVAudioPCMBuffer format: \(buffer.format.commonFormat.rawValue)."
    )
}
