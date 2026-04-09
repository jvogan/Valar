import CryptoKit
import Foundation
import ValarAudio
import ValarMLX
import ValarModelKit
import ValarPersistence

public struct VoiceCreateRequest: Codable, Sendable, Equatable {
    public let label: String
    public let modelID: String?
    public let runtimeModelID: String?
    public let sourceAssetName: String?
    public let voicePrompt: String?

    public init(
        label: String,
        modelID: String? = nil,
        runtimeModelID: String? = nil,
        sourceAssetName: String? = nil,
        voicePrompt: String? = nil
    ) {
        self.label = label
        self.modelID = modelID
        self.runtimeModelID = runtimeModelID
        self.sourceAssetName = sourceAssetName
        self.voicePrompt = voicePrompt
    }
}

public struct VoiceCloneRequest: Sendable, Equatable {
    public let label: String
    public let referenceTranscript: String
    public let audioData: Data
    public let audioFileExtension: String
    public let sourceAssetName: String?
    /// Optional model override. When nil, Valar uses the default Qwen Base clone-prompt lane
    /// and only falls back to TADA when the Base lane is unavailable.
    public let modelID: ModelIdentifier?

    public init(
        label: String,
        referenceTranscript: String,
        audioData: Data,
        audioFileExtension: String,
        sourceAssetName: String? = nil,
        modelID: ModelIdentifier? = nil
    ) {
        self.label = label
        self.referenceTranscript = referenceTranscript
        self.audioData = audioData
        self.audioFileExtension = audioFileExtension
        self.sourceAssetName = sourceAssetName
        self.modelID = modelID
    }
}

public struct VoiceStabilizeRequest: Sendable, Equatable {
    public let sourceVoiceID: UUID
    public let label: String?
    public let anchorText: String?
    public let modelID: ModelIdentifier?

    public init(
        sourceVoiceID: UUID,
        label: String? = nil,
        anchorText: String? = nil,
        modelID: ModelIdentifier? = nil
    ) {
        self.sourceVoiceID = sourceVoiceID
        self.label = label
        self.anchorText = anchorText
        self.modelID = modelID
    }
}

public enum ValarVoiceError: LocalizedError, Equatable, Sendable {
    case emptyLabel
    case emptyTranscript
    case unsupportedFileType(String)
    case fileTooLarge(bytes: Int)
    case invalidAudioHeader(String)
    case clipTooShort(actual: Double)
    case clipTooLong(actual: Double)
    case sampleRateTooLow(actual: Double)
    case missingModel(String)
    case unsupportedInferenceBackend
    case immutablePresetVoice(String)
    case voiceNotFound(UUID)
    case unsupportedOperation(String)

    public var errorDescription: String? {
        switch self {
        case .emptyLabel:
            return "Voice label must not be empty."
        case .emptyTranscript:
            return "Reference transcript must not be empty."
        case .unsupportedFileType(let fileExtension):
            return "Unsupported audio file type '.\(fileExtension)'. Choose a WAV or M4A clip."
        case .fileTooLarge(let bytes):
            let megabytes = Double(bytes) / (1_024 * 1_024)
            return "Reference audio file is too large (\(megabytes.formatted(.number.precision(.fractionLength(1)))) MB). Maximum allowed size is 50 MB."
        case .invalidAudioHeader(let expected):
            return "The selected file does not appear to be a valid \(expected) audio file."
        case .clipTooShort(let actual):
            return "Reference audio must be at least 5 seconds. Selected clip is \(actual.formatted(.number.precision(.fractionLength(1)))) seconds."
        case .clipTooLong(let actual):
            return "Reference audio must be 30 seconds or less. Selected clip is \(actual.formatted(.number.precision(.fractionLength(1)))) seconds."
        case .sampleRateTooLow(let actual):
            return "Reference audio must be at least 16 kHz. Selected clip is \(actual.formatted(.number.precision(.fractionLength(0)))) Hz."
        case .missingModel(let identifier):
            return "Model '\(identifier)' is unavailable."
        case .unsupportedInferenceBackend:
            return "The active inference backend does not support speaker embedding extraction."
        case .immutablePresetVoice(let label):
            return "Voice '\(label)' is a model-declared preset and cannot be edited or deleted."
        case .voiceNotFound(let id):
            return "No saved voice exists with id \(id.uuidString)."
        case .unsupportedOperation(let message):
            return message
        }
    }
}

public extension ValarRuntime {
    static let defaultVoiceCreateModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    static let defaultVoiceDesignModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    static let defaultVoiceCloneProfileModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    static let defaultVoiceCloneRuntimeModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    static let catalogPresetVoiceCreatedAt = Date(timeIntervalSince1970: 0)
    static let defaultStableNarratorAnchorText = """
    The lantern light was steady, and the old house finally grew quiet. \
    Tonight I will speak clearly, at an even pace, and carry the same calm tone from the first sentence to the last. \
    If you are listening closely, you should hear a confident narrator with warm detail, precise rhythm, and a grounded voice that remains consistent throughout the passage.
    """

    func listVoices() async -> [VoiceLibraryRecord] {
        let presetVoices = await modelDeclaredPresetVoices()
        let savedVoices = (await voiceStore.list()).sorted {
            $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending
        }
        return presetVoices + savedVoices
    }

    func voiceRecord(id: UUID) async -> VoiceLibraryRecord? {
        let voices = await listVoices()
        return voices.first(where: { $0.id == id })
    }

    func voiceRecord(label: String) async -> VoiceLibraryRecord? {
        let normalizedLabel = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalizedLabel.isEmpty else { return nil }
        let voices = await listVoices()
        return voices.first { voice in
            voice.label.lowercased() == normalizedLabel
                || voice.backendVoiceID?.lowercased() == normalizedLabel
        }
    }

    func modelDeclaredPresetVoices() async -> [VoiceLibraryRecord] {
        guard let catalogModels = try? await modelCatalog.refresh() else {
            return []
        }

        return catalogModels
            .filter { $0.installState == .installed }
            .reduce(into: [VoiceLibraryRecord]()) { records, catalogModel in
                guard
                    let presetSpecs = SupportedModelCatalog.entry(for: catalogModel.id)?.manifest.presetVoices,
                    presetSpecs.isEmpty == false
                else {
                    return
                }

                records.append(
                    contentsOf: presetSpecs.map { preset in
                        let backendVoiceID = preset.name
                        return VoiceLibraryRecord(
                            id: Self.catalogPresetVoiceID(
                                modelID: catalogModel.id,
                                backendVoiceID: backendVoiceID
                            ),
                            label: Self.displayLabel(forPresetVoiceID: backendVoiceID),
                            modelID: catalogModel.id.rawValue,
                            runtimeModelID: catalogModel.id.rawValue,
                            backendVoiceID: backendVoiceID,
                            createdAt: Self.catalogPresetVoiceCreatedAt
                        )
                    }
                )
            }
            .sorted { lhs, rhs in
                if lhs.modelID == rhs.modelID {
                    return lhs.label.localizedCaseInsensitiveCompare(rhs.label) == .orderedAscending
                }
                return lhs.modelID.localizedCaseInsensitiveCompare(rhs.modelID) == .orderedAscending
            }
    }

    func createVoice(_ request: VoiceCreateRequest) async throws -> VoiceLibraryRecord {
        let label = request.label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !label.isEmpty else {
            throw ValarVoiceError.emptyLabel
        }

        let prompt = request.voicePrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPrompt = prompt?.isEmpty == false ? prompt : nil
        let requestedModelID = Self.normalizedModelIdentifier(
            request.modelID,
            fallback: resolvedPrompt == nil
                ? Self.defaultVoiceCreateModelID
                : Self.defaultVoiceDesignModelID
        )
        let resolvedModelID = if resolvedPrompt != nil,
                                 requestedModelID != Self.defaultVoiceDesignModelID {
            Self.defaultVoiceDesignModelID
        } else {
            requestedModelID
        }
        let resolvedRuntimeModelID = Self.normalizedModelIdentifier(
            request.runtimeModelID,
            fallback: resolvedModelID
        )
        let sourceAssetName = Self.normalizedOptionalString(request.sourceAssetName)

        return try await voiceStore.save(
            VoiceLibraryRecord(
                label: label,
                modelID: resolvedModelID.rawValue,
                runtimeModelID: resolvedRuntimeModelID.rawValue,
                sourceAssetName: sourceAssetName,
                voiceKind: resolvedPrompt == nil ? nil : VoiceKind.legacyPrompt.rawValue,
                voicePrompt: resolvedPrompt
            )
        )
    }

    func cloneVoice(_ request: VoiceCloneRequest) async throws -> VoiceLibraryRecord {
        let label = request.label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !label.isEmpty else {
            throw ValarVoiceError.emptyLabel
        }

        let transcript = request.referenceTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            throw ValarVoiceError.emptyTranscript
        }

        let fileExtension = request.audioFileExtension.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard VoiceCloneFileValidator.allowedExtensions.contains(fileExtension) else {
            throw ValarVoiceError.unsupportedFileType(fileExtension)
        }
        guard request.audioData.count <= VoiceCloneFileValidator.maximumFileSizeBytes else {
            throw ValarVoiceError.fileTooLarge(bytes: request.audioData.count)
        }

        try VoiceCloneFileValidator.validateFileHeader(request.audioData, hint: fileExtension)

        let decodedBuffer = try await audioPipeline.decode(request.audioData, hint: fileExtension)
        let assessment = try VoiceCloneAudioValidator.validate(decodedBuffer)
        let runtimeModelID = try await preferredCloneRuntimeModelID(request.modelID)
        let runtimeDescriptor = try await descriptor(for: runtimeModelID)
        let targetSampleRate = runtimeDescriptor.defaultSampleRate ?? assessment.normalizedBuffer.format.sampleRate
        let preparedBuffer = if assessment.normalizedBuffer.format.sampleRate == targetSampleRate {
            assessment.normalizedBuffer
        } else {
            try await audioPipeline.resample(assessment.normalizedBuffer, to: targetSampleRate)
        }

        let exportedAsset = try await audioPipeline.export(
            preparedBuffer,
            as: AudioFormatDescriptor(
                sampleRate: preparedBuffer.format.sampleRate,
                channelCount: 1,
                sampleFormat: .float32,
                interleaved: false,
                container: "wav"
            )
        )

        let voiceID = UUID()
        let assetURL = try paths.voiceAssetURL(voiceID: voiceID, fileExtension: "wav")
        try persistVoiceAsset(exportedAsset.data, to: assetURL)

        let monoSamples = preparedBuffer.channels.first ?? []

        if runtimeDescriptor.familyID == .tadaTTS {
            guard let conditioningBackend = inferenceBackend as? any VoiceConditioningInferenceBackend else {
                throw ValarVoiceError.unsupportedInferenceBackend
            }
            let conditioning = try await conditioningBackend.extractVoiceConditioning(
                VoiceConditioningExtractionRequest(
                    descriptor: runtimeDescriptor,
                    monoReferenceSamples: monoSamples,
                    sampleRate: preparedBuffer.format.sampleRate,
                    referenceTranscript: transcript
                )
            )
            let bundleURL = try paths.voiceConditioningAssetURL(voiceID: voiceID)
            _ = try TADAConditioningBundleIO.write(conditioning: conditioning, to: bundleURL)

            let record = VoiceLibraryRecord(
                id: voiceID,
                label: label,
                modelID: runtimeDescriptor.id.rawValue,
                runtimeModelID: runtimeDescriptor.id.rawValue,
                sourceAssetName: Self.normalizedOptionalString(request.sourceAssetName),
                referenceAudioAssetName: assetURL.lastPathComponent,
                referenceTranscript: transcript,
                referenceDurationSeconds: assessment.durationSeconds,
                referenceSampleRate: preparedBuffer.format.sampleRate,
                referenceChannelCount: assessment.originalChannelCount,
                conditioningFormat: VoiceLibraryRecord.tadaReferenceConditioningFormat,
                voiceKind: VoiceKind.tadaReference.rawValue
            )
            return try await voiceStore.save(record)
        } else {
            if runtimeDescriptor.familyID == .voxtralTTS {
                throw ValarVoiceError.unsupportedOperation("Voxtral is a preset-voice model and does not support voice cloning. Use --voice to select from its 20 preset voices.")
            }
            if runtimeDescriptor.familyID == .qwen3TTS {
                let extractedConditioning = try await optionalQwenConditioning(
                    descriptor: runtimeDescriptor,
                    monoReferenceSamples: monoSamples,
                    sampleRate: preparedBuffer.format.sampleRate,
                    referenceTranscript: transcript
                )
                let fallbackEmbedding = extractedConditioning == nil
                    ? try await optionalSpeakerEmbedding(
                        descriptor: runtimeDescriptor,
                        monoReferenceSamples: monoSamples
                    )
                    : nil
                let storedPayload = extractedConditioning?.payload ?? fallbackEmbedding
                let conditioningFormat = extractedConditioning?.format
                    ?? VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat
                let voiceKind = conditioningFormat == VoiceLibraryRecord.qwenClonePromptConditioningFormat
                    ? VoiceKind.clonePrompt.rawValue
                    : VoiceKind.embeddingOnly.rawValue
                let record = VoiceLibraryRecord(
                    id: voiceID,
                    label: label,
                    modelID: runtimeDescriptor.id.rawValue,
                    runtimeModelID: runtimeDescriptor.id.rawValue,
                    sourceAssetName: Self.normalizedOptionalString(request.sourceAssetName),
                    referenceAudioAssetName: assetURL.lastPathComponent,
                    referenceTranscript: transcript,
                    referenceDurationSeconds: assessment.durationSeconds,
                    referenceSampleRate: preparedBuffer.format.sampleRate,
                    referenceChannelCount: assessment.originalChannelCount,
                    speakerEmbedding: storedPayload,
                    conditioningFormat: conditioningFormat,
                    voiceKind: voiceKind
                )
                return try await voiceStore.save(record)
            }

            guard let embeddingBackend = inferenceBackend as? any SpeakerEmbeddingInferenceBackend else {
                throw ValarVoiceError.unsupportedInferenceBackend
            }
            let embedding = try await embeddingBackend.extractSpeakerEmbedding(
                descriptor: runtimeDescriptor,
                monoReferenceSamples: monoSamples
            )

            let record = VoiceLibraryRecord(
                id: voiceID,
                label: label,
                modelID: runtimeDescriptor.id.rawValue,
                runtimeModelID: runtimeDescriptor.id.rawValue,
                sourceAssetName: Self.normalizedOptionalString(request.sourceAssetName),
                referenceAudioAssetName: assetURL.lastPathComponent,
                referenceTranscript: transcript,
                referenceDurationSeconds: assessment.durationSeconds,
                referenceSampleRate: preparedBuffer.format.sampleRate,
                referenceChannelCount: assessment.originalChannelCount,
                speakerEmbedding: embedding,
                conditioningFormat: VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat,
                voiceKind: VoiceKind.embeddingOnly.rawValue
            )
            return try await voiceStore.save(record)
        }
    }

    func stabilizeVoice(_ request: VoiceStabilizeRequest) async throws -> VoiceLibraryRecord {
        guard let sourceVoice = await voiceRecord(id: request.sourceVoiceID) else {
            throw ValarVoiceError.voiceNotFound(request.sourceVoiceID)
        }
        guard !sourceVoice.isModelDeclaredPreset else {
            throw ValarVoiceError.immutablePresetVoice(sourceVoice.label)
        }

        let kind = sourceVoice.resolvedVoiceKind
        guard kind == .legacyPrompt || sourceVoice.voicePrompt?.isEmpty == false else {
            throw ValarVoiceError.unsupportedOperation(
                "Only expressive designed Qwen voices can be stabilized into a Base clone-prompt narrator."
            )
        }

        let anchorText = Self.normalizedOptionalString(request.anchorText) ?? Self.defaultStableNarratorAnchorText
        let resolvedLabel = Self.normalizedOptionalString(request.label) ?? "\(sourceVoice.label) Stable"
        let designModelID = sourceVoice.preferredRuntimeModelIdentifier ?? Self.defaultVoiceDesignModelID
        let cloneModelID = request.modelID ?? Self.defaultVoiceCloneRuntimeModelID

        let anchorChunk = try await synthesizeVoiceChunk(
            text: anchorText,
            modelID: designModelID,
            voiceRecord: sourceVoice,
            voiceBehavior: .expressive
        )
        let anchorData = try await exportVoiceChunkAsWAV(anchorChunk)
        let sourceAssetName = "stabilized-anchor-\(sourceVoice.id.uuidString).wav"

        return try await cloneVoice(
            VoiceCloneRequest(
                label: resolvedLabel,
                referenceTranscript: anchorText,
                audioData: anchorData,
                audioFileExtension: "wav",
                sourceAssetName: sourceAssetName,
                modelID: cloneModelID
            )
        )
    }

    func upgradeVoiceForSynthesisIfNeeded(_ voice: VoiceLibraryRecord) async throws -> VoiceLibraryRecord {
        guard !voice.isModelDeclaredPreset else {
            return voice
        }

        let kind = voice.resolvedVoiceKind
        guard voice.inferredFamilyID == "qwen3_tts",
              kind == .clonePrompt || kind == .embeddingOnly,
              !voice.hasReusableQwenClonePrompt,
              let assetName = voice.referenceAudioAssetName,
              let transcript = Self.normalizedOptionalString(voice.referenceTranscript)
        else {
            return voice
        }

        let runtimeModelID = Self.defaultVoiceCloneRuntimeModelID
        guard let runtimeDescriptor = try? await descriptor(for: runtimeModelID),
              runtimeDescriptor.familyID == .qwen3TTS else {
            return voice
        }

        let assetURL = try resolvedVoiceAssetURL(assetName)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            return voice
        }

        let assetData = try VoiceLibraryProtection.readProtectedFile(from: assetURL)
        let buffer = try await audioPipeline.decode(assetData, hint: assetURL.pathExtension)
        let monoSamples = buffer.channels.first ?? []
        guard !monoSamples.isEmpty else {
            return voice
        }

        guard let conditioning = try await optionalQwenConditioning(
            descriptor: runtimeDescriptor,
            monoReferenceSamples: monoSamples,
            sampleRate: buffer.format.sampleRate,
            referenceTranscript: transcript
        ), conditioning.format == VoiceLibraryRecord.qwenClonePromptConditioningFormat else {
            return voice
        }

        var upgradedVoice = voice
        upgradedVoice.modelID = runtimeModelID.rawValue
        upgradedVoice.runtimeModelID = runtimeModelID.rawValue
        upgradedVoice.speakerEmbedding = conditioning.payload
        upgradedVoice.conditioningFormat = conditioning.format
        upgradedVoice.voiceKind = VoiceKind.clonePrompt.rawValue

        let saved = try await voiceStore.save(upgradedVoice)
        await voicePromptCache.invalidate(voiceID: voice.id)
        return saved
    }

    func deleteVoice(_ id: UUID) async throws {
        guard let existingVoice = await voiceRecord(id: id) else {
            throw ValarVoiceError.voiceNotFound(id)
        }
        if existingVoice.isModelDeclaredPreset {
            throw ValarVoiceError.immutablePresetVoice(existingVoice.label)
        }

        if let assetName = existingVoice.referenceAudioAssetName {
            let assetURL = try resolvedVoiceAssetURL(assetName)
            try removeVoiceItemIfPresent(at: assetURL)
        }

        let conditioningURL = try paths.voiceConditioningAssetURL(voiceID: id)
        try removeVoiceItemIfPresent(at: conditioningURL)

        try await voiceStore.delete(id)
        await voicePromptCache.invalidate(voiceID: id)
    }
}

private extension ValarRuntime {
    static func displayLabel(forPresetVoiceID backendVoiceID: String) -> String {
        backendVoiceID
            .split(separator: "_")
            .map { component in
                let lowered = component.lowercased()
                return lowered.count <= 2 ? lowered.uppercased() : lowered.capitalized
            }
            .joined(separator: " ")
    }

    static func catalogPresetVoiceID(modelID: ModelIdentifier, backendVoiceID: String) -> UUID {
        let digest = SHA256.hash(data: Data("valar-preset:\(modelID.rawValue):\(backendVoiceID)".utf8))
        let bytes = Array(digest.prefix(16))
        let uuidBytes: [UInt8] = [
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            (bytes[6] & 0x0F) | 0x50,
            bytes[7],
            (bytes[8] & 0x3F) | 0x80,
            bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]
        return UUID(uuid: (
            uuidBytes[0], uuidBytes[1], uuidBytes[2], uuidBytes[3],
            uuidBytes[4], uuidBytes[5], uuidBytes[6], uuidBytes[7],
            uuidBytes[8], uuidBytes[9], uuidBytes[10], uuidBytes[11],
            uuidBytes[12], uuidBytes[13], uuidBytes[14], uuidBytes[15]
        ))
    }

    func descriptor(for identifier: ModelIdentifier) async throws -> ModelDescriptor {
        if let descriptor = await modelRegistry.descriptor(for: identifier) {
            return descriptor
        }

        if let descriptor = await capabilityRegistry.descriptor(for: identifier) {
            return descriptor
        }

        let supportedModels = try await modelCatalog.supportedModels()
        if let descriptor = supportedModels.first(where: { $0.id == identifier })?.descriptor {
            return descriptor
        }

        throw ValarVoiceError.missingModel(identifier.rawValue)
    }

    func persistVoiceAsset(_ data: Data, to url: URL) throws {
        try ValarAppPaths.validateContainment(url, within: paths.voiceLibraryDirectory)
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try VoiceLibraryProtection.writeProtectedFile(data, to: url)
    }

    func resolvedVoiceAssetURL(_ assetName: String) throws -> URL {
        try ValarAppPaths.validateRelativePath(assetName, label: "voice asset name")
        let assetURL = paths.voiceLibraryDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(assetURL, within: paths.voiceLibraryDirectory)
        return assetURL
    }

    func removeVoiceItemIfPresent(at url: URL) throws {
        try ValarAppPaths.validateContainment(url, within: paths.voiceLibraryDirectory)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try FileManager.default.removeItem(at: url)
    }

    static func normalizedOptionalString(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    static func normalizedModelIdentifier(_ value: String?, fallback: ModelIdentifier) -> ModelIdentifier {
        if let trimmed = normalizedOptionalString(value) {
            return ModelIdentifier(trimmed)
        }
        return fallback
    }

    func synthesizeVoiceChunk(
        text: String,
        modelID: ModelIdentifier,
        voiceRecord: VoiceLibraryRecord?,
        voiceBehavior: SpeechSynthesisVoiceBehavior
    ) async throws -> AudioChunk {
        let resolvedDescriptor = try await descriptor(for: modelID)
        let backendRuntime = BackendSelectionPolicy.Runtime(
            availableBackends: [inferenceBackend.backendKind]
        )
        let configuration = try BackendSelectionPolicy().runtimeConfiguration(
            for: resolvedDescriptor,
            residencyPolicy: .automatic,
            runtime: backendRuntime
        )

        let request = SpeechSynthesisRequest(
            model: modelID,
            text: text,
            voice: voiceRecord?.makeVoiceProfile(),
            sampleRate: resolvedDescriptor.defaultSampleRate ?? 24_000,
            responseFormat: "pcm_f32le",
            voiceBehavior: voiceBehavior
        )
        do {
            return try await withReservedTextToSpeechWorkflowSession(
                descriptor: resolvedDescriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.synthesize(request: request, in: reserved.session)
            }
        } catch let error as WorkflowReservationError {
            switch error {
            case .unsupportedTextToSpeech:
                throw ValarVoiceError.unsupportedOperation(
                    "Model '\(modelID.rawValue)' does not support speech synthesis for voice stabilization."
                )
            default:
                throw error
            }
        }
    }

    func exportVoiceChunkAsWAV(_ chunk: AudioChunk) async throws -> Data {
        let buffer = AudioPCMBuffer(
            mono: chunk.samples,
            sampleRate: chunk.sampleRate,
            container: "wav"
        )
        let exportedAsset = try await audioPipeline.export(
            buffer,
            as: AudioFormatDescriptor(
                sampleRate: chunk.sampleRate,
                channelCount: 1,
                sampleFormat: .float32,
                interleaved: false,
                container: "wav"
            )
        )
        return exportedAsset.data
    }

    func preferredCloneRuntimeModelID(_ explicitModelID: ModelIdentifier?) async throws -> ModelIdentifier {
        if let explicitModelID {
            return explicitModelID
        }

        let catalogModels = try await modelCatalog.refresh()
        let installedCloneModels = catalogModels.filter {
            $0.installState == .installed && $0.descriptor.capabilities.contains(.voiceCloning)
        }

        for preferredModelID in [
            Self.defaultVoiceCloneRuntimeModelID,
            TadaCatalog.tada3BModelIdentifier,
            TadaCatalog.tada1BModelIdentifier,
        ] {
            if installedCloneModels.contains(where: { $0.id == preferredModelID }) {
                return preferredModelID
            }
        }

        return Self.defaultVoiceCloneRuntimeModelID
    }

    func optionalSpeakerEmbedding(
        descriptor: ModelDescriptor,
        monoReferenceSamples: [Float]
    ) async throws -> Data? {
        guard let embeddingBackend = inferenceBackend as? any SpeakerEmbeddingInferenceBackend else {
            return nil
        }
        do {
            return try await embeddingBackend.extractSpeakerEmbedding(
                descriptor: descriptor,
                monoReferenceSamples: monoReferenceSamples
            )
        } catch {
            return nil
        }
    }

    func optionalQwenConditioning(
        descriptor: ModelDescriptor,
        monoReferenceSamples: [Float],
        sampleRate: Double,
        referenceTranscript: String
    ) async throws -> VoiceConditioning? {
        guard descriptor.familyID == .qwen3TTS else {
            return nil
        }
        guard let conditioningBackend = inferenceBackend as? any VoiceConditioningInferenceBackend else {
            return nil
        }
        do {
            return try await conditioningBackend.extractVoiceConditioning(
                VoiceConditioningExtractionRequest(
                    descriptor: descriptor,
                    monoReferenceSamples: monoReferenceSamples,
                    sampleRate: sampleRate,
                    referenceTranscript: referenceTranscript
                )
            )
        } catch {
            return nil
        }
    }
}

private struct VoiceCloneAssessment: Sendable, Equatable {
    let normalizedBuffer: AudioPCMBuffer
    let durationSeconds: Double
    let originalChannelCount: Int
}

private enum VoiceCloneAudioValidator {
    static func validate(_ buffer: AudioPCMBuffer) throws -> VoiceCloneAssessment {
        let duration = buffer.duration
        guard duration >= 5 else {
            throw ValarVoiceError.clipTooShort(actual: duration)
        }
        guard duration <= 30 else {
            throw ValarVoiceError.clipTooLong(actual: duration)
        }
        guard buffer.format.sampleRate >= 16_000 else {
            throw ValarVoiceError.sampleRateTooLow(actual: buffer.format.sampleRate)
        }

        let originalChannelCount = max(buffer.format.channelCount, buffer.channels.count)
        let normalizedBuffer = downmixToMono(buffer)

        return VoiceCloneAssessment(
            normalizedBuffer: normalizedBuffer,
            durationSeconds: duration,
            originalChannelCount: originalChannelCount
        )
    }

    private static func downmixToMono(_ buffer: AudioPCMBuffer) -> AudioPCMBuffer {
        guard buffer.channels.count > 1 else { return buffer }

        let frameCount = buffer.frameCount
        guard frameCount > 0 else {
            return AudioPCMBuffer(mono: [], sampleRate: buffer.format.sampleRate, container: buffer.format.container)
        }

        var mixed = Array(repeating: Float.zero, count: frameCount)
        let channels = buffer.channels.filter { !$0.isEmpty }
        guard !channels.isEmpty else { return buffer }

        for channel in channels {
            for index in 0 ..< min(channel.count, frameCount) {
                mixed[index] += channel[index]
            }
        }

        let divisor = Float(channels.count)
        mixed = mixed.map { $0 / divisor }
        return AudioPCMBuffer(mono: mixed, sampleRate: buffer.format.sampleRate, container: buffer.format.container)
    }
}

private enum VoiceCloneFileValidator {
    static let allowedExtensions: Set<String> = ["wav", "m4a"]
    static let maximumFileSizeBytes: Int = 50 * 1_024 * 1_024

    static func validateFileHeader(_ data: Data, hint: String) throws {
        guard data.count >= 12 else {
            throw ValarVoiceError.invalidAudioHeader(hint.uppercased())
        }

        switch hint.lowercased() {
        case "wav":
            let riff = data.prefix(4)
            let wave = data[data.startIndex + 8 ..< data.startIndex + 12]
            guard riff.elementsEqual("RIFF".utf8),
                  wave.elementsEqual("WAVE".utf8) else {
                throw ValarVoiceError.invalidAudioHeader("WAV")
            }
        case "m4a":
            let ftyp = data[data.startIndex + 4 ..< data.startIndex + 8]
            guard ftyp.elementsEqual("ftyp".utf8) else {
                throw ValarVoiceError.invalidAudioHeader("M4A")
            }
        default:
            throw ValarVoiceError.unsupportedFileType(hint)
        }
    }
}
