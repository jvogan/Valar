@preconcurrency import ArgumentParser
import Foundation
import ValarAudio
import ValarExport
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence

enum SpeakOutputFormat: String, CaseIterable, ExpressibleByArgument {
    case wav
    case oggOpus = "ogg_opus"
    case pcmF32le = "pcm_f32le"
}

extension SpeechSynthesisVoiceBehavior: @retroactive ExpressibleByArgument {}

private let validOpusRates: Set<Double> = [8_000, 12_000, 16_000, 24_000, 48_000]
private let voxtralPresetVoiceIDs = Set(VoxtralCatalog.presetVoices.map(\.name))
private let orpheusPresetVoiceIDs = Set(OrpheusCatalog.presetVoices.map(\.name))

struct SpeakCommand: AsyncParsableCommand {
    static let defaultModelAlias = "Qwen3-TTS-12Hz-1.7B-Base"
    static let defaultModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Synthesize text to an audio file (WAV, OGG/Opus, or raw PCM). Requires an installed TTS model — run 'valartts models install <id>' first."
    )

    @Argument(help: "The text to synthesize.")
    var inputText: String?

    @Option(name: .long, help: "Text to synthesize. Use this instead of the positional argument when scripting.")
    var text: String?

    @Option(name: .long, help: "Model identifier to load for synthesis. Defaults to the first installed speech model, or Qwen3-TTS-12Hz-1.7B-Base.")
    var model: String = Self.defaultModelAlias

    @Option(name: .long, help: "Voice UUID (from 'valartts voices list') or preset name (e.g. tara, neutral_female). Saved voices use full UUIDs; preset names depend on the selected model.")
    var voice: String?

    @Option(name: .long, help: "Qwen VoiceDesign instruction prompt for the target speaking style.")
    var instruct: String?

    @Option(name: .long, help: "Optional reference audio file for one-shot cloning (for example with TADA).")
    var referenceAudio: String?

    @Option(name: .long, help: "Optional transcript for the reference audio. Strongly recommended; saved-voice creation still requires it.")
    var referenceTranscript: String?

    @Option(name: .long, help: "Chatterbox emotion exaggeration (0.0-1.0).")
    var exaggeration: Float?

    @Option(name: .long, help: "Chatterbox classifier-free guidance weight (0.0-1.0).")
    var cfgWeight: Float?

    @Option(name: .long, help: "Language code for synthesis (e.g. EN, FR, DE, ES, NL, PT, IT, HI, AR). Defaults to auto-detection.")
    var language: String?

    @Option(name: .long, help: "Destination path for the rendered audio file. Defaults to 'speech.wav' in the current directory.")
    var output: String = "speech.wav"

    @Option(name: .long, help: "Output audio format. Supported values: wav, ogg_opus, pcm_f32le.")
    var format: SpeakOutputFormat = .wav

    @Option(name: .long, help: "Sampling temperature (0.0–2.0). Higher values increase randomness.")
    var temperature: Float?

    @Option(name: .long, help: "Nucleus sampling top-p (0.0–1.0, exclusive of 0). Filters tokens by cumulative probability.")
    var topP: Float?

    @Option(name: .long, help: "Repetition penalty (1.0–2.0). Values > 1 discourage repeated tokens.")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "Maximum tokens to generate (1–8192).")
    var maxTokens: Int?

    @Option(name: .long, help: "Optional Qwen long-form override: auto, expressive, or stableNarrator.")
    var voiceBehavior: SpeechSynthesisVoiceBehavior = .auto

    mutating func run() async throws {
        let synthesisText = try resolvedText()
        // Validate generation parameter ranges before doing any work.
        if let t = temperature, !(0...2).contains(t) {
            throw SpeakCommandError.invalidArgument("--temperature must be between 0.0 and 2.0 (got \(t)).")
        }
        if let p = topP, p <= 0 || p > 1 {
            throw SpeakCommandError.invalidArgument("--top-p must be in (0.0, 1.0] (got \(p)).")
        }
        if let r = repetitionPenalty, !(1...2).contains(r) {
            throw SpeakCommandError.invalidArgument("--repetition-penalty must be between 1.0 and 2.0 (got \(r)).")
        }
        if let m = maxTokens, !(1...8192).contains(m) {
            throw SpeakCommandError.invalidArgument("--max-tokens must be between 1 and 8192 (got \(m)).")
        }
        if let value = exaggeration, !(0...1).contains(value) {
            throw SpeakCommandError.invalidArgument("--exaggeration must be between 0.0 and 1.0 (got \(value)).")
        }
        if let value = cfgWeight, !(0...1).contains(value) {
            throw SpeakCommandError.invalidArgument("--cfg-weight must be between 0.0 and 1.0 (got \(value)).")
        }

        let runtime = try ValarRuntime()
        _ = await runtime.ensureStartupMaintenance()
        let provisionalDescriptor = try await resolveDescriptor(
            for: model,
            voiceRecord: nil,
            runtime: runtime
        )
        let shouldPreferPresetResolution = Self.prefersVibeVoicePresetResolution(
            rawIdentifier: selectedVoiceIdentifier,
            familyID: provisionalDescriptor.familyID
        )
        let voiceRecord = shouldPreferPresetResolution
            ? nil
            : try await resolveVoiceRecord(
                for: selectedVoiceIdentifier,
                runtime: runtime
            )
        if referenceAudio != nil, voiceRecord != nil {
            throw SpeakCommandError.invalidArgument(
                "--reference-audio cannot be combined with a stored voice UUID or label."
            )
        }
        let descriptor = shouldPreferPresetResolution
            ? provisionalDescriptor
            : try await resolveDescriptor(for: model, voiceRecord: voiceRecord, runtime: runtime)
        let resolvedRequestedVoice = try voiceRecord == nil
            ? resolveRequestedVoiceProfile(
                rawIdentifier: selectedVoiceIdentifier,
                descriptor: descriptor
            )
            : (voiceProfile: nil, vibeVoiceSelection: nil)
        let configuration = try BackendSelectionPolicy().runtimeConfiguration(
            for: descriptor,
            runtime: BackendSelectionPolicy.Runtime(
                availableBackends: [runtime.inferenceBackend.backendKind]
            )
        )
        if configuration.backendKind == .mlx, !Self.checkMetallibAvailable() {
            throw SpeakCommandError.missingInferenceAssets(
                "Local MLX inference assets are missing: expected default.metallib or mlx.metallib next to \(URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent). Run: bash scripts/build_metallib.sh apps/ValarCLI/.build"
            )
        }

        do {
            let outputURL = resolvedOutputURL(from: output)
            let inlineReferencePayload = try await requestTimeReferencePromptPayload(runtime: runtime)
            let promptPayload = try await referencePromptPayload(
                for: voiceRecord,
                runtime: runtime
            )
            let baseVoiceProfile = inlineReferencePayload == nil
                ? resolvedRequestedVoice.voiceProfile ?? voiceRecord.map { resolvedVoiceProfile(from: $0) }
                : resolvedRequestedVoice.voiceProfile
            let enrichedVoiceProfile: VoiceProfile? = try baseVoiceProfile.map { profile in
                guard let record = voiceRecord else { return profile }
                return try enrichWithTADAConditioning(profile, record: record, runtime: runtime)
            }
            let effectiveLanguage = resolvedRequestedVoice.vibeVoiceSelection?.effectiveLanguage ?? language
            let request = SpeechSynthesisRequest(
                model: descriptor.id,
                text: synthesisText,
                voice: enrichedVoiceProfile,
                language: effectiveLanguage,
                referenceAudioAssetName: inlineReferencePayload == nil ? voiceRecord?.referenceAudioAssetName : nil,
                referenceAudioPCMFloat32LE: inlineReferencePayload?.pcmData ?? promptPayload?.pcmData,
                referenceAudioSampleRate: inlineReferencePayload?.sampleRate ?? promptPayload?.sampleRate,
                referenceTranscript: inlineReferencePayload?.transcript ?? promptPayload?.transcript,
                instruct: normalizedInstruct(),
                exaggeration: exaggeration,
                cfgWeight: cfgWeight,
                sampleRate: descriptor.defaultSampleRate ?? 24_000,
                responseFormat: "pcm_f32le",
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                maxTokens: maxTokens,
                voiceBehavior: voiceBehavior
            )
            let chunk = try await runtime.withReservedTextToSpeechWorkflowSession(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.synthesize(request: request, in: reserved.session)
            }
            try await writeOutput(chunk: chunk, to: outputURL, format: format, runtime: runtime)
            let formatLabel: String
            switch format {
            case .wav: formatLabel = "WAV"
            case .oggOpus: formatLabel = "OGG/Opus"
            case .pcmF32le: formatLabel = "PCM F32LE"
            }
            let message = "Wrote \(formatLabel) to \(outputURL.path)"
            let resolvedVoiceIdentifier = resolvedRequestedVoice.vibeVoiceSelection?.effectiveVoice
                ?? resolvedRequestedVoice.voiceProfile?.label
                ?? voiceRecord?.id.uuidString
                ?? ""
            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("speak"),
                    data: SpeechSynthesisPayloadDTO(
                        message: message,
                        modelID: descriptor.id.rawValue,
                        outputPath: outputURL.path,
                        text: synthesisText,
                        voiceID: resolvedVoiceIdentifier,
                        effectiveVoiceID: resolvedRequestedVoice.vibeVoiceSelection?.effectiveVoice,
                        effectiveLanguage: resolvedRequestedVoice.vibeVoiceSelection?.effectiveLanguage,
                        voiceSelectionMode: resolvedRequestedVoice.vibeVoiceSelection?.selectionMode.rawValue
                    )
                )
            } else {
                print(message)
                if let vibeVoiceSelection = resolvedRequestedVoice.vibeVoiceSelection,
                   vibeVoiceSelection.selectionMode != .explicit {
                    print(vibeVoiceSelectionSummary(vibeVoiceSelection))
                }
            }
        } catch {
            if let reservationError = error as? WorkflowReservationError,
               case .unsupportedTextToSpeech = reservationError {
                throw SpeakCommandError.unsupportedWorkflow(modelID: descriptor.id.rawValue)
            }
            if let backendError = error as? MLXBackendError {
                throw SpeakCommandError.modelLoadFailed(
                    modelID: descriptor.id.rawValue,
                    underlying: backendError
                )
            }
            throw error
        }
    }

    private var selectedVoiceIdentifier: String? {
        voice
    }

    private func resolvedText() throws -> String {
        let positional = inputText?.trimmingCharacters(in: .whitespacesAndNewlines)
        let flagged = text?.trimmingCharacters(in: .whitespacesAndNewlines)

        if let positional, !positional.isEmpty, let flagged, !flagged.isEmpty {
            throw SpeakCommandError.invalidArgument("Provide either positional text or --text, not both.")
        }
        if let positional, !positional.isEmpty {
            return positional
        }
        if let flagged, !flagged.isEmpty {
            return flagged
        }
        throw SpeakCommandError.invalidArgument("Missing text. Provide positional text or --text.")
    }

    private func normalizedInstruct() -> String? {
        let trimmed = instruct?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func normalizedReferenceTranscript() -> String? {
        let trimmed = referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func resolveDescriptor(
        for rawIdentifier: String,
        voiceRecord: VoiceLibraryRecord?,
        runtime: ValarRuntime
    ) async throws -> ModelDescriptor {
        try await resolveDescriptor(
            for: resolvedModelIdentifier(from: rawIdentifier, voiceRecord: voiceRecord, runtime: runtime),
            runtime: runtime
        )
    }

    private func resolveDescriptor(
        for identifier: ModelIdentifier,
        runtime: ValarRuntime
    ) async throws -> ModelDescriptor {
        if let descriptor = await runtime.modelRegistry.descriptor(for: identifier) {
            return try validatedSpeechModel(descriptor)
        }

        if let descriptor = await runtime.capabilityRegistry.descriptor(for: identifier) {
            return try validatedSpeechModel(descriptor)
        }

        let normalizedQuery = Self.normalizedModelQuery(identifier.rawValue)
        if let installedAliasMatch = try? await runtime.modelCatalog.supportedModels()
            .first(where: {
                $0.installState == .installed
                    && $0.descriptor.capabilities.contains(.speechSynthesis)
                    && Self.modelMatchesAlias(descriptor: $0.descriptor, normalizedQuery: normalizedQuery)
            }) {
            return try validatedSpeechModel(installedAliasMatch.descriptor)
        }

        if let catalogModel = try await runtime.modelCatalog.model(for: identifier) {
            if catalogModel.installState == .installed {
                return try validatedSpeechModel(catalogModel.descriptor)
            }

            // Model files may be on disk but not yet registered in DB (e.g. after a DB reset).
            // Auto-register using metadata-only install so the user doesn't need to re-download.
            if let packDir = try? runtime.paths.modelPackDirectory(
                familyID: catalogModel.descriptor.familyID.rawValue,
                modelID: identifier.rawValue
            ), FileManager.default.fileExists(atPath: packDir.path),
               let manifest = try? await runtime.modelCatalog.installationManifest(for: identifier) {
                let sourceKind: ModelPackSourceKind = catalogModel.providerURL == nil ? .localFile : .remoteURL
                let sourceLocation = catalogModel.providerURL?.absoluteString ?? identifier.rawValue
                _ = try? await runtime.modelInstaller.install(
                    manifest: manifest,
                    sourceKind: sourceKind,
                    sourceLocation: sourceLocation,
                    notes: catalogModel.notes,
                    mode: .metadataOnly
                )
                return try validatedSpeechModel(catalogModel.descriptor)
            }

            throw SpeakCommandError.modelNotInstalled(modelID: identifier.rawValue)
        }

        if let catalogAliasMatch = try? await runtime.modelCatalog.supportedModels()
            .first(where: {
                $0.descriptor.capabilities.contains(.speechSynthesis)
                    && Self.modelMatchesAlias(descriptor: $0.descriptor, normalizedQuery: normalizedQuery)
            }) {
            if catalogAliasMatch.installState == .installed {
                return try validatedSpeechModel(catalogAliasMatch.descriptor)
            }
            throw SpeakCommandError.modelNotInstalled(modelID: catalogAliasMatch.id.rawValue)
        }

        throw SpeakCommandError.missingModel(modelID: identifier.rawValue)
    }

    private func resolvedModelIdentifier(
        from rawIdentifier: String,
        voiceRecord: VoiceLibraryRecord?,
        runtime: ValarRuntime
    ) async -> ModelIdentifier {
        let trimmedIdentifier = rawIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedIdentifier.isEmpty || trimmedIdentifier == Self.defaultModelAlias {
            if let preferredRuntimeModelIdentifier = voiceRecord?.preferredRuntimeModelIdentifier {
                return preferredRuntimeModelIdentifier
            }
            if let inferredModelQuery = Self.inferredSpeechModelQuery(
                fromExplicitVoiceIdentifier: selectedVoiceIdentifier
            ) {
                return ModelIdentifier(inferredModelQuery)
            }
            if let installedModel = try? await runtime.modelCatalog.supportedModels()
                .first(where: { $0.installState == .installed && $0.descriptor.capabilities.contains(.speechSynthesis) }) {
                return installedModel.id
            }

            return ModelIdentifier(Self.defaultModelIdentifier)
        }

        return ModelIdentifier(trimmedIdentifier)
    }

    private func validatedSpeechModel(_ descriptor: ModelDescriptor) throws -> ModelDescriptor {
        guard descriptor.capabilities.contains(.speechSynthesis) else {
            throw SpeakCommandError.unsupportedModel(modelID: descriptor.id.rawValue)
        }
        return descriptor
    }

    private func resolveVoiceRecord(
        for rawIdentifier: String?,
        runtime: ValarRuntime
    ) async throws -> VoiceLibraryRecord? {
        guard let rawIdentifier else {
            return nil
        }

        let trimmedIdentifier = rawIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedIdentifier.isEmpty == false else {
            return nil
        }

        if let voiceID = UUID(uuidString: trimmedIdentifier) {
            guard let record = await runtime.voiceRecord(id: voiceID) else {
                throw SpeakCommandError.missingVoice(voiceID: trimmedIdentifier)
            }
            return try await runtime.upgradeVoiceForSynthesisIfNeeded(record)
        }

        if let record = await runtime.voiceRecord(label: trimmedIdentifier) {
            return try await runtime.upgradeVoiceForSynthesisIfNeeded(record)
        }
        return nil
    }

    private func resolveRequestedVoiceProfile(
        rawIdentifier: String?,
        descriptor: ModelDescriptor
    ) throws -> (voiceProfile: VoiceProfile?, vibeVoiceSelection: VibeVoiceResolvedRequest?) {
        if descriptor.familyID == .vibevoiceRealtimeTTS {
            do {
                let resolvedRequest = try VibeVoiceRequestResolver.resolve(
                    voice: rawIdentifier,
                    language: language
                )
                return (
                    voiceProfile: VoiceProfile(
                        label: resolvedRequest.effectiveVoice,
                        backendVoiceID: resolvedRequest.effectiveVoice,
                        sourceModel: descriptor.id,
                        localeIdentifier: resolvedRequest.effectiveLanguage,
                        voiceKind: .preset
                    ),
                    vibeVoiceSelection: resolvedRequest
                )
            } catch let error as VibeVoiceRequestResolutionError {
                throw SpeakCommandError.invalidArgument(error.localizedDescription)
            }
        }

        return (
            voiceProfile: try resolveExplicitVoiceProfile(
                rawIdentifier: rawIdentifier,
                descriptor: descriptor
            ),
            vibeVoiceSelection: nil
        )
    }

    private func resolveExplicitVoiceProfile(
        rawIdentifier: String?,
        descriptor: ModelDescriptor
    ) throws -> VoiceProfile? {
        guard let rawIdentifier else {
            return nil
        }

        let trimmedIdentifier = rawIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedIdentifier.isEmpty == false else {
            return nil
        }

        if UUID(uuidString: trimmedIdentifier) != nil {
            return nil
        }

        let normalizedIdentifier = trimmedIdentifier.lowercased()
        switch descriptor.familyID {
        case .qwen3TTS where QwenCatalog.acceptsNamedSpeaker(descriptor.id):
            return VoiceProfile(
                label: trimmedIdentifier,
                backendVoiceID: trimmedIdentifier,
                sourceModel: descriptor.id,
                localeIdentifier: language,
                voiceKind: .namedSpeaker
            )
        case .voxtralTTS:
            guard let resolvedName = VoxtralCatalog.resolvePresetName(normalizedIdentifier)
                ?? (voxtralPresetVoiceIDs.contains(normalizedIdentifier) ? normalizedIdentifier : nil) else {
                throw SpeakCommandError.invalidVoiceID(trimmedIdentifier)
            }
            return VoiceProfile(
                label: resolvedName,
                backendVoiceID: resolvedName,
                sourceModel: descriptor.id,
                localeIdentifier: language,
                voiceKind: .preset
            )
        case .orpheus:
            guard orpheusPresetVoiceIDs.contains(normalizedIdentifier) else {
                throw SpeakCommandError.invalidVoiceID(trimmedIdentifier)
            }
            return VoiceProfile(
                label: normalizedIdentifier,
                backendVoiceID: normalizedIdentifier,
                sourceModel: descriptor.id,
                localeIdentifier: language,
                voiceKind: .preset
            )
        case .tadaTTS:
            throw SpeakCommandError.invalidArgument(
                "TADA models use saved voices, not named presets. " +
                "Create a voice with 'valartts voices create --reference-audio <file>' then use the UUID. " +
                "Or pass --reference-audio directly."
            )
        default:
            throw SpeakCommandError.invalidVoiceID(trimmedIdentifier)
        }
    }

    static func inferredSpeechModelQuery(fromExplicitVoiceIdentifier identifier: String?) -> String? {
        guard let identifier = identifier?.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty,
              UUID(uuidString: identifier) == nil else {
            return nil
        }
        if VibeVoiceCatalog.acceptsPresetIdentifier(identifier) {
            return ModelFamilyID.vibevoiceRealtimeTTS.rawValue
        }
        return nil
    }

    static func prefersVibeVoicePresetResolution(
        rawIdentifier: String?,
        familyID: ModelFamilyID
    ) -> Bool {
        guard familyID == .vibevoiceRealtimeTTS,
              let identifier = rawIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty else {
            return false
        }
        return VibeVoiceCatalog.acceptsPresetIdentifier(identifier)
    }

    private static func normalizedModelQuery(_ value: String) -> String {
        value.lowercased().replacingOccurrences(of: "_", with: "-")
    }

    private static func modelMatchesAlias(
        descriptor: ModelDescriptor,
        normalizedQuery: String
    ) -> Bool {
        let family = descriptor.familyID.rawValue.lowercased().replacingOccurrences(of: "_", with: "-")
        let identifier = descriptor.id.rawValue.lowercased()
        return family == normalizedQuery || identifier.contains(normalizedQuery)
    }

    private func referencePromptPayload(
        for voice: VoiceLibraryRecord?,
        runtime: ValarRuntime
    ) async throws -> (pcmData: Data, sampleRate: Double, transcript: String)? {
        guard let voice else {
            return nil
        }

        guard
            let assetName = voice.referenceAudioAssetName,
            let transcript = voice.referenceTranscript?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !transcript.isEmpty
        else {
            return nil
        }

        if voice.hasReusableQwenClonePrompt {
            return nil
        }

        // Check voice prompt cache first — skip decode on cache hit
        if let cached = await runtime.voicePromptCache.payload(for: voice.id) {
            return (
                pcmData: audioPCMFloat32LEData(from: cached.monoSamples),
                sampleRate: cached.sampleRate,
                transcript: cached.referenceTranscript
            )
        }

        let assetURL = try resolvedVoiceAssetURL(assetName: assetName, runtime: runtime)
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw SpeakCommandError.missingReferenceAudio(assetName)
        }

        let assetData = try VoiceLibraryProtection.readProtectedFile(from: assetURL)
        let buffer = try await runtime.audioPipeline.decode(assetData, hint: assetURL.pathExtension)
        let monoSamples = buffer.channels.first ?? []

        // Populate cache for next call with same voice
        await runtime.voicePromptCache.store(
            VoicePromptCache.Payload(
                monoSamples: monoSamples,
                sampleRate: buffer.format.sampleRate,
                referenceTranscript: transcript,
                normalizedVoicePrompt: voice.voicePrompt
            ),
            for: voice.id
        )

        return (
            pcmData: audioPCMFloat32LEData(from: monoSamples),
            sampleRate: buffer.format.sampleRate,
            transcript: transcript
        )
    }

    private func requestTimeReferencePromptPayload(
        runtime: ValarRuntime
    ) async throws -> (pcmData: Data, sampleRate: Double, transcript: String?)? {
        guard let rawPath = referenceAudio?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawPath.isEmpty else {
            return nil
        }

        let fileURL = URL(fileURLWithPath: rawPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw SpeakCommandError.invalidArgument("Reference audio file not found: \(fileURL.path)")
        }

        let assetData = try Data(contentsOf: fileURL)
        let buffer = try await runtime.audioPipeline.decode(assetData, hint: fileURL.pathExtension)
        let monoSamples = buffer.channels.first ?? []
        return (
            pcmData: audioPCMFloat32LEData(from: monoSamples),
            sampleRate: buffer.format.sampleRate,
            transcript: normalizedReferenceTranscript()
        )
    }

    private func resolvedVoiceProfile(from record: VoiceLibraryRecord) -> VoiceProfile {
        record.makeVoiceProfile()
    }

    private func enrichWithTADAConditioning(
        _ profile: VoiceProfile,
        record: VoiceLibraryRecord,
        runtime: ValarRuntime
    ) throws -> VoiceProfile {
        guard record.conditioningFormat == VoiceLibraryRecord.tadaReferenceConditioningFormat,
              let assetName = record.conditioningAssetName else {
            return profile
        }
        let bundleURL = runtime.paths.voiceLibraryDirectory
            .appendingPathComponent(assetName, isDirectory: true)
        guard let _ = try? ValarAppPaths.validateContainment(bundleURL, within: runtime.paths.voiceLibraryDirectory) else {
            return profile
        }
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            return profile  // Bundle missing; WAV fallback will apply.
        }
        let loaded = try TADAConditioningBundleIO.load(from: bundleURL)
        return VoiceProfile(
            id: profile.id,
            label: profile.label,
            backendVoiceID: profile.backendVoiceID,
            sourceModel: profile.sourceModel,
            runtimeModel: profile.runtimeModel,
            referenceAudioAssetName: profile.referenceAudioAssetName,
            referenceTranscript: profile.referenceTranscript,
            speakerEmbedding: profile.speakerEmbedding,
            conditioningFormat: profile.conditioningFormat,
            conditioningAssets: loaded.assetFiles,
            conditioningMetadata: loaded.metadata,
            voiceKind: profile.voiceKind,
            isLegacyExpressive: profile.isLegacyExpressive
        )
    }

    private func resolvedVoiceAssetURL(
        assetName: String,
        runtime: ValarRuntime
    ) throws -> URL {
        try ValarAppPaths.validateRelativePath(assetName, label: "voice asset name")
        let assetURL = runtime.paths.voiceLibraryDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(
            assetURL,
            within: runtime.paths.voiceLibraryDirectory
        )
        return assetURL
    }

    private func resolvedOutputURL(from rawPath: String) -> URL {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return URL(fileURLWithPath: trimmedPath, relativeTo: currentDirectory).standardizedFileURL
    }

    private func writeOutput(
        chunk: AudioChunk,
        to outputURL: URL,
        format: SpeakOutputFormat,
        runtime: ValarRuntime
    ) async throws {
        let buffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
        switch format {
        case .wav:
            let audioFormat = AudioFormatDescriptor(
                sampleRate: chunk.sampleRate,
                channelCount: 1,
                sampleFormat: .float32,
                interleaved: false,
                container: "wav"
            )
            _ = try await runtime.audioPipeline.export(buffer, as: audioFormat, to: outputURL)
        case .oggOpus:
            var oggBuffer = buffer
            if !validOpusRates.contains(oggBuffer.format.sampleRate) {
                oggBuffer = try await runtime.audioPipeline.resample(oggBuffer, to: 24_000)
            }
            let oggData = try await ChannelAudioExporter().encode(oggBuffer)
            try oggData.write(to: outputURL)
        case .pcmF32le:
            let rawData = audioPCMFloat32LEData(from: chunk.samples)
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try rawData.write(to: outputURL)
        }
    }

    private func audioPCMFloat32LEData(from samples: [Float]) -> Data {
        var data = Data()
        data.reserveCapacity(samples.count * MemoryLayout<Float>.size)
        for sample in samples {
            var bits = sample.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { rawBuffer in
                data.append(contentsOf: rawBuffer)
            }
        }
        return data
    }

    private func vibeVoiceSelectionSummary(_ selection: VibeVoiceResolvedRequest) -> String {
        "Resolved VibeVoice preset \(selection.effectiveVoice) (\(selection.selectionMode.rawValue), language \(selection.effectiveLanguage))."
    }

    private static func checkMetallibAvailable() -> Bool {
        let binaryURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let binaryDir = binaryURL.deletingLastPathComponent()
        let candidates = [
            binaryDir.appendingPathComponent("default.metallib"),
            binaryDir.appendingPathComponent("mlx.metallib"),
        ]
        return candidates.contains { FileManager.default.fileExists(atPath: $0.path) }
    }
}

private enum SpeakCommandError: LocalizedError, CLIExitCodeProviding {
    case invalidVoiceID(String)
    case missingModel(modelID: String)
    case modelNotInstalled(modelID: String)
    case unsupportedModel(modelID: String)
    case missingVoice(voiceID: String)
    case missingReferenceAudio(String)
    case unsupportedWorkflow(modelID: String)
    case modelLoadFailed(modelID: String, underlying: Error)
    case missingInferenceAssets(String)
    case invalidArgument(String)

    var cliExitCode: CLIExitCode {
        switch self {
        case .invalidVoiceID,
             .missingModel,
             .modelNotInstalled,
             .unsupportedModel,
             .missingVoice,
             .missingInferenceAssets,
             .invalidArgument:
            return .usageError
        case .missingReferenceAudio,
             .unsupportedWorkflow,
             .modelLoadFailed:
            return .failure
        }
    }

    var errorDescription: String? {
        switch self {
        case let .invalidVoiceID(value):
            return "'\(value)' is not a valid voice ID or supported preset voice. Saved voices use UUIDs (for example 550e8400-e29b-41d4-a716-446655440000); preset voices depend on the selected model."
        case let .missingModel(modelID):
            if let hiddenReason = CatalogVisibilityPolicy.currentProcess().hiddenReason(for: ModelIdentifier(modelID)) {
                return hiddenReason
            }
            return "Model '\(modelID)' was not found in the catalog.\nRun: valartts models list to see available speech synthesis models."
        case let .modelNotInstalled(modelID):
            return "Model '\(modelID)' is not installed.\nRun: valartts models install \(modelID)\nTo see all available models, run: valartts models list"
        case let .unsupportedModel(modelID):
            return "Model '\(modelID)' does not support speech synthesis — use a TTS model.\nRun: valartts models list to see available speech synthesis models."
        case let .missingVoice(voiceID):
            return "Voice '\(voiceID)' was not found in the voice library. Verify the voice ID is correct, or create the voice first."
        case let .missingReferenceAudio(assetName):
            return "Voice reference audio '\(assetName)' is missing from the voice library. The voice record may be corrupt — re-create the voice to fix this."
        case let .missingInferenceAssets(message):
            return message
        case let .unsupportedWorkflow(modelID):
            return "Model '\(modelID)' loaded but is not a text-to-speech workflow — use a TTS model.\nRun: valartts models list to see available speech synthesis models."
        case let .modelLoadFailed(modelID, underlying):
            return "Failed to load model '\(modelID)': \(underlying.localizedDescription)\nIf the shared cache may be stale or corrupt, reinstall with: valartts models install \(modelID) --refresh-cache --allow-download"
        case let .invalidArgument(message):
            return message
        }
    }
}
