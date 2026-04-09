import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import ValarAudio
import ValarExport
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence

private let maxMultipartBodyBytes = 15_000_000
private let maxSpeechRequestBodyBytes = 15_000_000
private let validOpusRates: Set<Double> = [8_000, 12_000, 16_000, 24_000, 48_000]
private let voxtralPresetVoiceIDs = Set(VoxtralCatalog.presetVoices.map(\.name))

extension ValarDaemonRouter {
    static func registerAudioRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime
    ) {
        let audio = router.group("audio")

        audio.post("speech") { request, context async throws -> Response in
            await handleSpeechRequest(request, context: context, runtime: runtime)
        }

        audio.post("transcriptions") { request, _ async throws -> Response in
            await handleTranscriptionRequest(request, runtime: runtime)
        }

        audio.post("transcriptions/stream") { request, _ async throws -> Response in
            await handleTranscriptionStreamRequest(request, runtime: runtime)
        }

        audio.post("speech/stream") { request, _ async throws -> Response in
            await handleSpeechStreamRequest(request, runtime: runtime)
        }
    }

    static func registerAlignmentRoutes(
        on router: HTTPRouteGroup,
        runtime: ValarRuntime
    ) {
        router.post("alignments") { request, _ async throws -> Response in
            await handleAlignmentRequest(request, runtime: runtime)
        }
    }

    private static func handleSpeechRequest(
        _ request: Request,
        context: ValarDaemonRouter.Context,
        runtime: ValarRuntime
    ) async -> Response {
        var request = request
        _ = await runtime.ensureStartupMaintenance()
        let payload: SpeechRequest
        let parsedRequestTimeReferenceAudio: RequestTimeReferenceAudio?
        do {
            let body = try await request.collectBody(upTo: maxSpeechRequestBodyBytes)
            payload = try JSONDecoder().decode(
                SpeechRequest.self,
                from: body
            )
            parsedRequestTimeReferenceAudio = try parseRequestTimeReferenceAudio(
                encodedValue: payload.reference_audio,
                providedTranscript: payload.reference_transcript
            )
        } catch is DecodingError {
            return errorResponse(
                "Invalid JSON request body.",
                status: .badRequest
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        }

        let input = payload.input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard input.isEmpty == false else {
            return errorResponse(
                "Field 'input' is required.",
                status: .badRequest
            )
        }

        let responseFormat = trimmedOrNil(payload.response_format) ?? "wav"
        guard responseFormat == "wav" || responseFormat == "ogg_opus" || responseFormat == "pcm_f32le" else {
            return errorResponse(
                "Invalid response_format '\(responseFormat)'. Supported values: wav, ogg_opus, pcm_f32le.",
                status: .badRequest
            )
        }

        if let t = payload.temperature, !(0...2).contains(t) {
            return errorResponse("Field 'temperature' must be in range 0...2.", status: .badRequest)
        }
        if let p = payload.top_p, !(0...1).contains(p) || p == 0 {
            return errorResponse("Field 'top_p' must be in range (0, 1].", status: .badRequest)
        }
        if let r = payload.repetition_penalty, !(1...2).contains(r) {
            return errorResponse("Field 'repetition_penalty' must be in range 1...2.", status: .badRequest)
        }
        if let m = payload.max_tokens, !(1...8192).contains(m) {
            return errorResponse("Field 'max_tokens' must be in range 1...8192.", status: .badRequest)
        }
        if let rc = payload.repetition_context_size, !(1...4096).contains(rc) {
            return errorResponse("Field 'repetition_context_size' must be in range 1...4096.", status: .badRequest)
        }
        if let value = payload.exaggeration, !(0...1).contains(value) {
            return errorResponse("Field 'exaggeration' must be in range 0...1.", status: .badRequest)
        }
        if let value = payload.cfg_weight, !(0...1).contains(value) {
            return errorResponse("Field 'cfg_weight' must be in range 0...1.", status: .badRequest)
        }

        let rawVoiceIdentifier = trimmedOrNil(payload.voice) ?? trimmedOrNil(payload.voice_id)

        let provisionalDescriptor: ModelDescriptor
        do {
            provisionalDescriptor = try await resolveSpeechDescriptor(
                requestedModel: effectiveSpeechModel(
                    requested: payload.model,
                    explicitVoiceIdentifier: rawVoiceIdentifier,
                    voice: nil
                ),
                runtime: runtime
            )
        } catch {
            return errorResponse(error)
        }

        let shouldPreferPresetResolution = prefersVibeVoicePresetResolution(
            rawIdentifier: rawVoiceIdentifier,
            familyID: provisionalDescriptor.familyID
        )

        let voiceRecord: VoiceLibraryRecord?
        do {
            voiceRecord = shouldPreferPresetResolution ? nil : try await resolveVoiceRecord(
                identifier: rawVoiceIdentifier,
                runtime: runtime
            )
        } catch {
            return errorResponse(error)
        }

        if parsedRequestTimeReferenceAudio != nil, voiceRecord != nil {
            return errorResponse(
                "Field 'reference_audio' cannot be combined with a stored voice UUID.",
                status: .badRequest
            )
        }

        let descriptor: ModelDescriptor
        do {
            descriptor = shouldPreferPresetResolution
                ? provisionalDescriptor
                : try await resolveSpeechDescriptor(
                    requestedModel: effectiveSpeechModel(
                        requested: payload.model,
                        explicitVoiceIdentifier: rawVoiceIdentifier,
                        voice: voiceRecord
                    ),
                    runtime: runtime
                )
        } catch {
            return errorResponse(error)
        }

        let resolvedRequestedVoice: (voiceProfile: VoiceProfile?, vibeVoiceSelection: VibeVoiceResolvedRequest?)
        do {
            resolvedRequestedVoice = voiceRecord == nil
                ? try resolveRequestedVoiceProfile(
                    rawIdentifier: rawVoiceIdentifier,
                    descriptor: descriptor,
                    language: payload.language
                )
                : (voiceProfile: nil, vibeVoiceSelection: nil)
        } catch {
            return errorResponse(error)
        }

        let configuration: ModelRuntimeConfiguration
        do {
            configuration = try BackendSelectionPolicy().runtimeConfiguration(
                for: descriptor,
                runtime: BackendSelectionPolicy.Runtime(
                    availableBackends: [runtime.inferenceBackend.backendKind]
                )
            )
        } catch {
            return errorResponse(
                "Failed to prepare model '\(descriptor.id.rawValue)': \(error.localizedDescription)",
                status: .internalServerError
            )
        }

        do {
            let inlineReferencePayload = try await requestTimeReferencePromptPayload(
                parsedRequestTimeReferenceAudio,
                descriptor: descriptor,
                language: payload.language,
                runtime: runtime
            )
            let promptPayload = try await referencePromptPayload(for: voiceRecord, runtime: runtime)
            let baseVoiceProfile = inlineReferencePayload == nil
                ? resolvedRequestedVoice.voiceProfile ?? voiceRecord.map { voiceProfile(from: $0) }
                : resolvedRequestedVoice.voiceProfile
            let enrichedVoiceProfile: VoiceProfile? = baseVoiceProfile.map { profile in
                guard let record = voiceRecord else { return profile }
                return Self.enrichWithTADAConditioning(profile, record: record, runtime: runtime)
            }
            let effectiveLanguage = resolvedRequestedVoice.vibeVoiceSelection?.effectiveLanguage ?? payload.language
            let synthesisRequest = SpeechSynthesisRequest(
                model: descriptor.id,
                text: input,
                voice: enrichedVoiceProfile,
                language: effectiveLanguage,
                referenceAudioAssetName: inlineReferencePayload == nil ? voiceRecord?.referenceAudioAssetName : nil,
                referenceAudioPCMFloat32LE: inlineReferencePayload?.pcmData ?? promptPayload?.pcmData,
                referenceAudioSamples: inlineReferencePayload?.monoSamples ?? promptPayload?.monoSamples,
                referenceAudioSampleRate: inlineReferencePayload?.sampleRate ?? promptPayload?.sampleRate,
                referenceTranscript: inlineReferencePayload?.transcript ?? promptPayload?.transcript,
                instruct: trimmedOrNil(payload.instruct),
                exaggeration: payload.exaggeration,
                cfgWeight: payload.cfg_weight,
                sampleRate: descriptor.defaultSampleRate ?? 24_000,
                responseFormat: "pcm_f32le",
                temperature: payload.temperature,
                topP: payload.top_p,
                repetitionPenalty: payload.repetition_penalty,
                repetitionContextSize: payload.repetition_context_size,
                maxTokens: payload.max_tokens,
                voiceBehavior: payload.voice_behavior ?? .auto
            )
            let requestID = UUID()
            await runtime.activeSynthesisTracker.begin(
                requestID: requestID,
                modelID: descriptor.id.rawValue,
                voiceBehavior: synthesisRequest.voiceBehavior.rawValue,
                executionMode: SynthesisExecutionMode.oneShot.rawValue
            )
            let chunk: AudioChunk
            do {
                let observer = Self.synthesisObserver(runtime: runtime, requestID: requestID)
                chunk = try await withCancellationOnClientDisconnect(
                    channel: context.channel
                ) {
                    try await SynthesisExecutionObserverContext.$observer.withValue(observer) {
                        try await runtime.withReservedTextToSpeechWorkflowSession(
                            descriptor: descriptor,
                            configuration: configuration
                        ) { reserved in
                            try await reserved.workflow.synthesize(
                                request: synthesisRequest,
                                in: reserved.session
                            )
                        }
                    }
                }
                await runtime.activeSynthesisTracker.finish(
                    requestID: requestID,
                    terminalState: .completed
                )
            } catch let error as WorkflowReservationError {
                await runtime.activeSynthesisTracker.finish(
                    requestID: requestID,
                    terminalState: .failed,
                    message: error.localizedDescription
                )
                if case .unsupportedTextToSpeech = error {
                    return errorResponse(
                        "Model '\(descriptor.id.rawValue)' does not support speech synthesis.",
                        status: .badRequest
                    )
                }
                return errorResponse(error.localizedDescription, status: .internalServerError)
            } catch {
                await runtime.activeSynthesisTracker.finish(
                    requestID: requestID,
                    terminalState: Self.synthesisTerminalState(for: error),
                    message: error.localizedDescription
                )
                return errorResponse(error.localizedDescription, status: .internalServerError)
            }

            var body = ByteBuffer()
            if responseFormat == "ogg_opus" {
                var buffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
                buffer = await runtime.audioPipeline.peakNormalize(buffer)
                if !validOpusRates.contains(buffer.format.sampleRate) {
                    buffer = try await runtime.audioPipeline.resample(buffer, to: 24_000)
                }
                let oggData = try await ChannelAudioExporter().encode(buffer)
                body.writeBytes(oggData)
                return Response(
                    status: .ok,
                    headers: audioResponseHeaders(
                        contentType: "audio/ogg; codecs=opus",
                        vibeVoiceSelection: resolvedRequestedVoice.vibeVoiceSelection
                    ),
                    body: .init(byteBuffer: body)
                )
            } else if responseFormat == "pcm_f32le" {
                var pcmBuffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
                pcmBuffer = await runtime.audioPipeline.peakNormalize(pcmBuffer)
                let pcmData = Self.audioPCMFloat32LEData(from: pcmBuffer.channels[0])
                body.writeBytes(pcmData)
                return Response(
                    status: .ok,
                    headers: audioResponseHeaders(
                        contentType: "application/octet-stream",
                        vibeVoiceSelection: resolvedRequestedVoice.vibeVoiceSelection
                    ),
                    body: .init(byteBuffer: body)
                )
            } else {
                let wavData = try await renderWAV(chunk: chunk, runtime: runtime)
                body.writeBytes(wavData)
                return Response(
                    status: .ok,
                    headers: audioResponseHeaders(
                        contentType: "audio/wav",
                        vibeVoiceSelection: resolvedRequestedVoice.vibeVoiceSelection
                    ),
                    body: .init(byteBuffer: body)
                )
            }
        } catch {
            return errorResponse(error)
        }
    }

    private static func handleTranscriptionRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: maxMultipartBodyBytes)
            body = Data(buffer: buffer)
        } catch {
            return errorResponse(
                "Failed to read request body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        let multipart: MultipartFormData
        do {
            multipart = try MultipartFormData.parse(
                body: body,
                contentType: request.headers[.contentType] ?? ""
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        }

        guard let file = multipart.files["file"] else {
            return errorResponse(
                "Multipart field 'file' is required.",
                status: .badRequest
            )
        }

        let responseFormat = TranscriptionResponseFormat(
            apiValue: trimmedOrNil(multipart.fields["response_format"]) ?? TranscriptionResponseFormat.text.rawValue
        )
        guard let responseFormat else {
            return errorResponse(
                "Invalid response_format '\(trimmedOrNil(multipart.fields["response_format"]) ?? "")'. Supported values: \(TranscriptionResponseFormat.supportedValuesDescription).",
                status: .badRequest
            )
        }

        let descriptor: ModelDescriptor
        do {
            descriptor = try await resolveTranscriptionDescriptor(
                requestedModel: multipart.fields["model"],
                runtime: runtime
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: status(for: error)
            )
        }

        let chunk: AudioChunk
        do {
            chunk = try await decodeAudioChunk(
                data: file.data,
                hint: file.filename.flatMap(pathExtension(from:)),
                targetSampleRate: descriptor.defaultSampleRate,
                runtime: runtime
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        }

        do {
            let result = try await runtime.transcribe(
                SpeechRecognitionRequest(
                    model: descriptor.id,
                    audio: chunk,
                    languageHint: trimmedOrNil(multipart.fields["language"])
                )
            )
            return try transcriptionResponse(result, format: responseFormat)
        } catch let error as ValarRuntime.TranscriptionError {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .internalServerError
            )
        }
    }

    private static func handleAlignmentRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: maxMultipartBodyBytes)
            body = Data(buffer: buffer)
        } catch {
            return errorResponse(
                "Failed to read request body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        let multipart: MultipartFormData
        do {
            multipart = try MultipartFormData.parse(
                body: body,
                contentType: request.headers[.contentType] ?? ""
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        }

        guard let file = multipart.files["file"] else {
            return errorResponse(
                "Multipart field 'file' is required.",
                status: .badRequest
            )
        }

        guard let transcript = trimmedOrNil(multipart.fields["transcript"]) else {
            return errorResponse(
                "Multipart field 'transcript' is required.",
                status: .badRequest
            )
        }
        let languageHint = trimmedOrNil(multipart.fields["language"])

        let descriptor: ModelDescriptor
        do {
            descriptor = try await resolveAlignmentDescriptor(
                requestedModel: multipart.fields["model"],
                runtime: runtime
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: status(for: error)
            )
        }

        let chunk: AudioChunk
        do {
            chunk = try await decodeAudioChunk(
                data: file.data,
                hint: file.filename.flatMap(pathExtension(from:)),
                targetSampleRate: descriptor.defaultSampleRate,
                runtime: runtime
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        }

        do {
            let result = try await runtime.align(
                ForcedAlignmentRequest(
                    model: descriptor.id,
                    audio: chunk,
                    transcript: transcript,
                    languageHint: languageHint
                )
            )
            return try jsonResponse(result)
        } catch let error as ValarRuntime.AlignmentError {
            return errorResponse(
                error.localizedDescription,
                status: .badRequest
            )
        } catch {
            return errorResponse(
                error.localizedDescription,
                status: .internalServerError
            )
        }
    }

    private static func effectiveSpeechModel(
        requested: String?,
        explicitVoiceIdentifier: String?,
        voice: VoiceLibraryRecord?
    ) -> String? {
        if let requested = trimmedOrNil(requested) {
            return requested
        }
        if let runtimeModelID = voice?.preferredRuntimeModelID {
            return runtimeModelID
        }
        if let explicitVoiceIdentifier = trimmedOrNil(explicitVoiceIdentifier),
           UUID(uuidString: explicitVoiceIdentifier) == nil,
           VibeVoiceCatalog.acceptsPresetIdentifier(explicitVoiceIdentifier) {
                return ModelFamilyID.vibevoiceRealtimeTTS.rawValue
        }
        return nil
    }

    private static func prefersVibeVoicePresetResolution(
        rawIdentifier: String?,
        familyID: ModelFamilyID
    ) -> Bool {
        guard familyID == .vibevoiceRealtimeTTS,
              let identifier = trimmedOrNil(rawIdentifier) else {
            return false
        }
        return VibeVoiceCatalog.acceptsPresetIdentifier(identifier)
    }

    private static func resolveSpeechDescriptor(
        requestedModel: String?,
        runtime: ValarRuntime
    ) async throws -> ModelDescriptor {
        try await resolveDescriptor(
            requestedModel: requestedModel,
            capability: .speechSynthesis,
            runtime: runtime
        ) { descriptor in
            descriptor.capabilities.contains(.speechSynthesis)
        }
    }

    private static func resolveTranscriptionDescriptor(
        requestedModel: String?,
        runtime: ValarRuntime
    ) async throws -> ModelDescriptor {
        try await resolveDescriptor(
            requestedModel: requestedModel,
            capability: .speechRecognition,
            runtime: runtime
        ) { descriptor in
            descriptor.capabilities.contains(.speechRecognition)
                && descriptor.capabilities.contains(.forcedAlignment) == false
        }
    }

    private static func resolveAlignmentDescriptor(
        requestedModel: String?,
        runtime: ValarRuntime
    ) async throws -> ModelDescriptor {
        try await resolveDescriptor(
            requestedModel: requestedModel,
            capability: .forcedAlignment,
            runtime: runtime
        ) { descriptor in
            descriptor.capabilities.contains(.forcedAlignment)
        }
    }

    private static func resolveDescriptor(
        requestedModel: String?,
        capability: ModelCapability,
        runtime: ValarRuntime,
        validate: (ModelDescriptor) -> Bool
    ) async throws -> ModelDescriptor {
        let catalogModels = try await runtime.modelCatalog.models(supporting: capability)
        let installedCatalogModels = catalogModels.filter { $0.installState == .installed }

        if let requestedModel = trimmedOrNil(requestedModel) {
            if let exactDescriptor = await runtime.modelRegistry.descriptor(
                for: ModelIdentifier(requestedModel)
            ),
               validate(exactDescriptor) {
                return exactDescriptor
            }

            if let exactDescriptor = await runtime.capabilityRegistry.descriptor(
                for: ModelIdentifier(requestedModel)
            ),
               validate(exactDescriptor) {
                return exactDescriptor
            }

            if let exactCatalogModel = installedCatalogModels.first(where: {
                $0.id == ModelIdentifier(requestedModel) && validate($0.descriptor)
            }) {
                return exactCatalogModel.descriptor
            }

            let normalizedQuery = normalizedModelQuery(requestedModel)
            if let aliasDescriptor = installedCatalogModels.first(where: { model in
                validate(model.descriptor)
                    && modelMatchesAlias(
                        descriptor: model.descriptor,
                        normalizedQuery: normalizedQuery
                    )
            }) {
                return aliasDescriptor.descriptor
            }

            if let exactCatalogModel = try await runtime.modelCatalog.model(
                for: ModelIdentifier(requestedModel)
            ),
               validate(exactCatalogModel.descriptor) {
                throw DaemonRequestError.modelNotInstalled(
                    requestedModel,
                    hint: exactCatalogModel.id.rawValue
                )
            }

            if let aliasDescriptor = catalogModels.first(where: { model in
                validate(model.descriptor)
                    && modelMatchesAlias(
                        descriptor: model.descriptor,
                        normalizedQuery: normalizedQuery
                    )
            }) {
                throw DaemonRequestError.modelNotInstalled(
                    requestedModel,
                    hint: aliasDescriptor.id.rawValue
                )
            }

            throw DaemonRequestError.missingModel(requestedModel)
        }

        if let defaultDescriptor = installedCatalogModels.first(where: { validate($0.descriptor) }) {
            return defaultDescriptor.descriptor
        }

        let hint = (catalogModels.first(where: { $0.isRecommended && validate($0.descriptor) })
            ?? catalogModels.first(where: { validate($0.descriptor) }))?.id.rawValue
        throw DaemonRequestError.missingDefaultModel(capability, hint: hint)
    }

    private static func resolveVoiceRecord(
        identifier: String?,
        runtime: ValarRuntime
    ) async throws -> VoiceLibraryRecord? {
        guard let identifier = trimmedOrNil(identifier) else {
            return nil
        }
        if let voiceID = UUID(uuidString: identifier) {
            guard let record = await runtime.voiceRecord(id: voiceID) else {
                throw DaemonRequestError.missingVoice(identifier)
            }
            return try await runtime.upgradeVoiceForSynthesisIfNeeded(record)
        }

        if let record = await runtime.voiceRecord(label: identifier) {
            return try await runtime.upgradeVoiceForSynthesisIfNeeded(record)
        }

        return nil
    }

    private static func referencePromptPayload(
        for voice: VoiceLibraryRecord?,
        runtime: ValarRuntime
    ) async throws -> (monoSamples: [Float]?, pcmData: Data?, sampleRate: Double, transcript: String)? {
        guard
            let voice,
            let assetName = voice.referenceAudioAssetName,
            let transcript = voice.referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
            transcript.isEmpty == false
        else {
            return nil
        }

        if voice.hasReusableQwenClonePrompt {
            return nil
        }

        // Check voice prompt cache first — return [Float] directly to avoid encode/decode round-trip
        if let cached = await runtime.voicePromptCache.payload(for: voice.id) {
            return (
                monoSamples: cached.monoSamples,
                pcmData: nil,
                sampleRate: cached.sampleRate,
                transcript: cached.referenceTranscript
            )
        }

        try ValarAppPaths.validateRelativePath(assetName, label: "voice asset name")
        let assetURL = runtime.paths.voiceLibraryDirectory.appendingPathComponent(
            assetName,
            isDirectory: false
        )
        try ValarAppPaths.validateContainment(
            assetURL,
            within: runtime.paths.voiceLibraryDirectory
        )

        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw DaemonRequestError.missingReferenceAudio(assetName)
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: assetURL.path)
        if let size = attrs[.size] as? Int, size > 50_000_000 {
            throw DaemonRequestError.referenceAudioTooLarge(size)
        }

        let assetData = try VoiceLibraryProtection.readProtectedFile(from: assetURL)
        let buffer = try await runtime.audioPipeline.decode(
            assetData,
            hint: assetURL.pathExtension
        )
        let monoSamples = buffer.channels.first ?? []

        // Populate cache for subsequent requests with same voice
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
            monoSamples: nil,
            pcmData: audioPCMFloat32LEData(from: monoSamples),
            sampleRate: buffer.format.sampleRate,
            transcript: transcript
        )
    }

    private static func voiceProfile(from record: VoiceLibraryRecord) -> VoiceProfile {
        record.makeVoiceProfile()
    }

    private static func enrichWithTADAConditioning(
        _ profile: VoiceProfile,
        record: VoiceLibraryRecord,
        runtime: ValarRuntime
    ) -> VoiceProfile {
        guard record.conditioningFormat == VoiceLibraryRecord.tadaReferenceConditioningFormat,
              let assetName = record.conditioningAssetName else {
            return profile
        }
        let bundleURL = runtime.paths.voiceLibraryDirectory
            .appendingPathComponent(assetName, isDirectory: true)
        guard let _ = try? ValarAppPaths.validateContainment(bundleURL, within: runtime.paths.voiceLibraryDirectory) else {
            return profile
        }
        guard FileManager.default.fileExists(atPath: bundleURL.path),
              let loaded = try? TADAConditioningBundleIO.load(from: bundleURL) else {
            return profile  // Bundle missing or unreadable; WAV fallback will apply.
        }
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

    private static func resolveRequestedVoiceProfile(
        rawIdentifier: String?,
        descriptor: ModelDescriptor,
        language: String?
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
                throw DaemonRequestError(vibeVoiceResolutionError: error)
            }
        }

        return (
            voiceProfile: try resolveExplicitVoiceProfile(
                rawIdentifier: rawIdentifier,
                descriptor: descriptor,
                language: language
            ),
            vibeVoiceSelection: nil
        )
    }

    private static func resolveExplicitVoiceProfile(
        rawIdentifier: String?,
        descriptor: ModelDescriptor,
        language: String?
    ) throws -> VoiceProfile? {
        guard let rawIdentifier else {
            return nil
        }
        guard UUID(uuidString: rawIdentifier) == nil else {
            return nil
        }
        if descriptor.familyID == .tadaTTS {
            throw DaemonRequestError.unsupportedOperation(
                "TADA models use saved voices, not named presets. " +
                "Create a voice with 'valartts voices create --reference-audio <file>' then use the UUID. " +
                "Or pass --reference-audio directly."
            )
        }

        if descriptor.familyID == .qwen3TTS, QwenCatalog.acceptsNamedSpeaker(descriptor.id) {
            return VoiceProfile(
                label: rawIdentifier,
                backendVoiceID: rawIdentifier,
                sourceModel: descriptor.id,
                localeIdentifier: language,
                voiceKind: .namedSpeaker
            )
        }

        guard descriptor.familyID == .voxtralTTS else {
            throw DaemonRequestError.invalidVoice(rawIdentifier)
        }
        // Resolve aliases (e.g. "emma" → "neutral_female") and "random"
        let resolvedName = VoxtralCatalog.resolvePresetName(rawIdentifier)
            ?? (voxtralPresetVoiceIDs.contains(rawIdentifier) ? rawIdentifier : nil)
        guard let resolvedName else {
            throw DaemonRequestError.invalidVoice(rawIdentifier)
        }
        return VoiceProfile(
            label: resolvedName.lowercased(),
            backendVoiceID: resolvedName,
            sourceModel: descriptor.id,
            localeIdentifier: language,
            voiceKind: .preset
        )
    }

    private static func decodeAudioChunk(
        data: Data,
        hint: String?,
        targetSampleRate: Double?,
        runtime: ValarRuntime
    ) async throws -> AudioChunk {
        let hintLower = hint?.lowercased()
        let decoded: AudioPCMBuffer
        if hintLower == "ogg" || hintLower == "oga" {
            decoded = try ChannelAudioImporter().decode(data)
        } else {
            decoded = try await runtime.audioPipeline.decode(data, hint: hint)
        }
        let workingBuffer: AudioPCMBuffer

        if let targetSampleRate,
           abs(decoded.format.sampleRate - targetSampleRate) > 0.5 {
            workingBuffer = try await runtime.audioPipeline.resample(
                decoded,
                to: targetSampleRate
            )
        } else {
            workingBuffer = decoded
        }

        return AudioChunk(
            samples: workingBuffer.channels.first ?? [],
            sampleRate: workingBuffer.format.sampleRate
        )
    }

    private static func requestTimeReferencePromptPayload(
        _ requestTimeReferenceAudio: RequestTimeReferenceAudio?,
        descriptor: ModelDescriptor,
        language: String?,
        runtime: ValarRuntime
    ) async throws -> (monoSamples: [Float]?, pcmData: Data?, sampleRate: Double, transcript: String?)? {
        guard let requestTimeReferenceAudio else {
            return nil
        }

        if requestTimeReferenceAudio.data.count > 50_000_000 {
            throw DaemonRequestError.referenceAudioTooLarge(requestTimeReferenceAudio.data.count)
        }

        let audioChunk = try await decodeAudioChunk(
            data: requestTimeReferenceAudio.data,
            hint: requestTimeReferenceAudio.hint,
            targetSampleRate: nil,
            runtime: runtime
        )
        let transcript = try await resolvedRequestTimeReferenceTranscript(
            descriptor: descriptor,
            referenceAudio: audioChunk,
            providedTranscript: requestTimeReferenceAudio.transcript,
            language: language,
            runtime: runtime
        )

        return (
            monoSamples: audioChunk.samples,
            pcmData: audioPCMFloat32LEData(from: audioChunk.samples),
            sampleRate: audioChunk.sampleRate,
            transcript: transcript
        )
    }

    private static func resolvedRequestTimeReferenceTranscript(
        descriptor: ModelDescriptor,
        referenceAudio: AudioChunk,
        providedTranscript: String?,
        language: String?,
        runtime: ValarRuntime
    ) async throws -> String? {
        if let providedTranscript {
            return providedTranscript
        }

        guard isTADAModel(descriptor) else {
            return nil
        }

        let nonEnglishReference = isExplicitNonEnglishLanguage(language)

        do {
            if let transcript = try await autoTranscribeReferenceAudio(
                referenceAudio,
                languageHint: language,
                runtime: runtime
            ) {
                return transcript
            }
        } catch {
            if nonEnglishReference {
                throw error
            }
        }

        if nonEnglishReference {
            throw DaemonRequestError.transcriptRequiredForNonEnglishTADACloning
        }

        return nil
    }

    private static func autoTranscribeReferenceAudio(
        _ referenceAudio: AudioChunk,
        languageHint: String?,
        runtime: ValarRuntime
    ) async throws -> String? {
        guard let modelID = try await installedASRModelIdentifier(runtime: runtime) else {
            return nil
        }

        let result = try await runtime.transcribe(
            SpeechRecognitionRequest(
                model: modelID,
                audio: referenceAudio,
                languageHint: languageHint
            )
        )
        let transcript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return transcript.isEmpty ? nil : transcript
    }

    private static func installedASRModelIdentifier(runtime: ValarRuntime) async throws -> ModelIdentifier? {
        let models = try await runtime.modelCatalog.models(supporting: .speechRecognition)
        let installed = models.filter {
            $0.installState == .installed && $0.descriptor.capabilities.contains(.forcedAlignment) == false
        }
        return (installed.first(where: \.isRecommended) ?? installed.first)?.id
    }

    private static func parseRequestTimeReferenceAudio(
        encodedValue: String?,
        providedTranscript: String?
    ) throws -> RequestTimeReferenceAudio? {
        let transcript = trimmedOrNil(providedTranscript)

        guard let encodedValue = trimmedOrNil(encodedValue) else {
            if transcript != nil {
                throw DaemonRequestError.invalidReferenceAudio(
                    "Field 'reference_transcript' requires 'reference_audio'."
                )
            }
            return nil
        }

        if encodedValue.hasPrefix("data:") {
            return try requestTimeReferenceAudio(fromDataURL: encodedValue, transcript: transcript)
        }

        if let decoded = Data(base64Encoded: encodedValue), !decoded.isEmpty {
            return RequestTimeReferenceAudio(data: decoded, hint: nil, transcript: transcript)
        }

        throw DaemonRequestError.invalidReferenceAudio(
            "Field 'reference_audio' must be a valid base64 audio payload or data URL."
        )
    }

    private static func requestTimeReferenceAudio(
        fromDataURL value: String,
        transcript: String?
    ) throws -> RequestTimeReferenceAudio {
        guard let separator = value.firstIndex(of: ",") else {
            throw DaemonRequestError.invalidReferenceAudio("Field 'reference_audio' is not a valid data URL.")
        }

        let header = String(value[..<separator])
        let payload = String(value[value.index(after: separator)...])
        guard let data = Data(base64Encoded: payload), !data.isEmpty else {
            throw DaemonRequestError.invalidReferenceAudio("Field 'reference_audio' is not valid base64 audio data.")
        }

        let rawHint: String?
        if let slash = header.firstIndex(of: "/"),
           let semicolon = header.firstIndex(of: ";"),
           slash < semicolon {
            let extStart = header.index(after: slash)
            rawHint = String(header[extStart..<semicolon])
        } else {
            rawHint = nil
        }

        let allowedMIMESubtypes: Set<String> = [
            "wav", "mp3", "ogg", "opus", "m4a", "flac", "aac", "pcm", "mpeg", "mp4", "webm",
        ]
        let hint: String? = rawHint.flatMap { allowedMIMESubtypes.contains($0.lowercased()) ? $0 : nil }

        return RequestTimeReferenceAudio(data: data, hint: hint, transcript: transcript)
    }

    private static func isTADAModel(_ descriptor: ModelDescriptor) -> Bool {
        let family = descriptor.familyID.rawValue.lowercased()
        let identifier = descriptor.id.rawValue.lowercased()
        return family == "tada" || family == "tada_tts" || identifier.contains("tada")
    }

    private static func isExplicitNonEnglishLanguage(_ language: String?) -> Bool {
        guard let normalizedLanguage = trimmedOrNil(language)?.lowercased() else {
            return false
        }
        return isEnglishLanguage(normalizedLanguage) == false
    }

    private static func isEnglishLanguage(_ language: String) -> Bool {
        language == "en" || language == "english" || language.hasPrefix("en-") || language.hasPrefix("en_")
    }

    private static func renderWAV(
        chunk: AudioChunk,
        runtime: ValarRuntime
    ) async throws -> Data {
        var buffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
        buffer = await runtime.audioPipeline.peakNormalize(buffer)
        let exported = try await runtime.audioPipeline.transcode(buffer, container: "wav")
        return exported.data
    }

    private static func errorResponse(
        _ message: String,
        status: HTTPResponse.Status
    ) -> Response {
        daemonErrorResponse(message: message, status: status, kind: "audio_error")
    }

    private static func errorResponse(_ error: Error) -> Response {
        let status = status(for: error)
        let kind = (error as? DaemonRequestError)?.errorKind ?? "audio_error"
        return daemonErrorResponse(
            message: error.localizedDescription,
            status: status,
            kind: kind
        )
    }

    private static func audioResponseHeaders(
        contentType: String,
        vibeVoiceSelection: VibeVoiceResolvedRequest?
    ) -> HTTPFields {
        var headers = HTTPFields()
        headers[.contentType] = contentType
        applyVoiceSelectionHeaders(to: &headers, vibeVoiceSelection: vibeVoiceSelection)
        return headers
    }

    private static func applyVoiceSelectionHeaders(
        to headers: inout HTTPFields,
        vibeVoiceSelection: VibeVoiceResolvedRequest?
    ) {
        guard let vibeVoiceSelection else {
            return
        }

        if let effectiveVoiceHeader = HTTPField.Name("x-valar-effective-voice"),
           let effectiveLanguageHeader = HTTPField.Name("x-valar-effective-language"),
           let selectionModeHeader = HTTPField.Name("x-valar-voice-selection-mode"),
           let selectionKindHeader = HTTPField.Name("x-valar-voice-selection-kind") {
            headers[effectiveVoiceHeader] = vibeVoiceSelection.effectiveVoice
            headers[effectiveLanguageHeader] = vibeVoiceSelection.effectiveLanguage
            headers[selectionModeHeader] = vibeVoiceSelection.selectionMode.rawValue
            headers[selectionKindHeader] = voiceSelectionKind(for: vibeVoiceSelection)
        }
    }

    private static func voiceSelectionKind(for selection: VibeVoiceResolvedRequest) -> String {
        switch selection.selectionMode {
        case .autoDefault:
            return "missing_voice_auto_selected"
        case .explicit:
            return "explicit_voice"
        case .random:
            return "random_voice"
        }
    }

    private static func status(for error: Error) -> HTTPResponse.Status {
        guard let requestError = error as? DaemonRequestError else {
            return .badRequest
        }
        return requestError.httpStatus
    }

    private static func trimmedOrNil(_ value: String?) -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              value.isEmpty == false else {
            return nil
        }
        return value
    }

    private static func normalizedModelQuery(_ value: String) -> String {
        value.lowercased().replacingOccurrences(of: "_", with: "-")
    }

    private static func modelMatchesAlias(
        descriptor: ModelDescriptor,
        normalizedQuery: String
    ) -> Bool {
        let family = descriptor.familyID.rawValue
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        let identifier = descriptor.id.rawValue.lowercased()
        return family == normalizedQuery || identifier.contains(normalizedQuery)
    }

    // MARK: - Speech Stream

    private static func handleSpeechStreamRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        _ = await runtime.ensureStartupMaintenance()
        let payload: SpeechRequest
        let parsedRequestTimeReferenceAudio: RequestTimeReferenceAudio?
        do {
            let body = try await request.body.collect(upTo: maxSpeechRequestBodyBytes)
            payload = try JSONDecoder().decode(
                SpeechRequest.self,
                from: body
            )
            parsedRequestTimeReferenceAudio = try parseRequestTimeReferenceAudio(
                encodedValue: payload.reference_audio,
                providedTranscript: payload.reference_transcript
            )
        } catch is DecodingError {
            return errorResponse("Invalid JSON request body.", status: .badRequest)
        } catch {
            return errorResponse(error.localizedDescription, status: .badRequest)
        }

        let input = payload.input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard input.isEmpty == false else {
            return errorResponse("Field 'input' is required.", status: .badRequest)
        }

        let responseFormat = trimmedOrNil(payload.response_format) ?? "pcm_f32le"
        guard responseFormat == "pcm_f32le" else {
            return errorResponse(
                "Invalid response_format '\(responseFormat)' for streaming. Supported values: pcm_f32le.",
                status: .badRequest
            )
        }

        if let t = payload.temperature, !(0...2).contains(t) {
            return errorResponse("Field 'temperature' must be in range 0...2.", status: .badRequest)
        }
        if let p = payload.top_p, !(0...1).contains(p) || p == 0 {
            return errorResponse("Field 'top_p' must be in range (0, 1].", status: .badRequest)
        }
        if let r = payload.repetition_penalty, !(1...2).contains(r) {
            return errorResponse("Field 'repetition_penalty' must be in range 1...2.", status: .badRequest)
        }
        if let m = payload.max_tokens, !(1...8192).contains(m) {
            return errorResponse("Field 'max_tokens' must be in range 1...8192.", status: .badRequest)
        }
        if let rc = payload.repetition_context_size, !(1...4096).contains(rc) {
            return errorResponse("Field 'repetition_context_size' must be in range 1...4096.", status: .badRequest)
        }
        if let value = payload.exaggeration, !(0...1).contains(value) {
            return errorResponse("Field 'exaggeration' must be in range 0...1.", status: .badRequest)
        }
        if let value = payload.cfg_weight, !(0...1).contains(value) {
            return errorResponse("Field 'cfg_weight' must be in range 0...1.", status: .badRequest)
        }

        let rawVoiceIdentifier = trimmedOrNil(payload.voice) ?? trimmedOrNil(payload.voice_id)

        let provisionalDescriptor: ModelDescriptor
        do {
            provisionalDescriptor = try await resolveSpeechDescriptor(
                requestedModel: effectiveSpeechModel(
                    requested: payload.model,
                    explicitVoiceIdentifier: rawVoiceIdentifier,
                    voice: nil
                ),
                runtime: runtime
            )
        } catch {
            return errorResponse(error)
        }

        let shouldPreferPresetResolution = prefersVibeVoicePresetResolution(
            rawIdentifier: rawVoiceIdentifier,
            familyID: provisionalDescriptor.familyID
        )

        let voiceRecord: VoiceLibraryRecord?
        do {
            voiceRecord = shouldPreferPresetResolution ? nil : try await resolveVoiceRecord(
                identifier: rawVoiceIdentifier,
                runtime: runtime
            )
        } catch {
            return errorResponse(error)
        }

        if parsedRequestTimeReferenceAudio != nil, voiceRecord != nil {
            return errorResponse(
                "Field 'reference_audio' cannot be combined with a stored voice UUID.",
                status: .badRequest
            )
        }

        let descriptor: ModelDescriptor
        do {
            descriptor = shouldPreferPresetResolution
                ? provisionalDescriptor
                : try await resolveSpeechDescriptor(
                    requestedModel: effectiveSpeechModel(
                        requested: payload.model,
                        explicitVoiceIdentifier: rawVoiceIdentifier,
                        voice: voiceRecord
                    ),
                    runtime: runtime
                )
        } catch {
            return errorResponse(error)
        }

        let resolvedRequestedVoice: (voiceProfile: VoiceProfile?, vibeVoiceSelection: VibeVoiceResolvedRequest?)
        do {
            resolvedRequestedVoice = voiceRecord == nil
                ? try resolveRequestedVoiceProfile(
                    rawIdentifier: rawVoiceIdentifier,
                    descriptor: descriptor,
                    language: payload.language
                )
                : (voiceProfile: nil, vibeVoiceSelection: nil)
        } catch {
            return errorResponse(error)
        }

        let promptPayload: (monoSamples: [Float]?, pcmData: Data?, sampleRate: Double, transcript: String?)?
        do {
            let inlineReferencePayload = try await requestTimeReferencePromptPayload(
                parsedRequestTimeReferenceAudio,
                descriptor: descriptor,
                language: payload.language,
                runtime: runtime
            )
            let savedPromptPayload = try await referencePromptPayload(for: voiceRecord, runtime: runtime)
            promptPayload = inlineReferencePayload ?? savedPromptPayload
        } catch {
            return errorResponse(error.localizedDescription, status: .internalServerError)
        }

        let configuration: ModelRuntimeConfiguration
        do {
            configuration = try BackendSelectionPolicy().runtimeConfiguration(
                for: descriptor,
                runtime: BackendSelectionPolicy.Runtime(
                    availableBackends: [runtime.inferenceBackend.backendKind]
                )
            )
        } catch {
            return errorResponse(
                "Failed to prepare model '\(descriptor.id.rawValue)': \(error.localizedDescription)",
                status: .internalServerError
            )
        }

        let baseStreamVoiceProfile = resolvedRequestedVoice.voiceProfile ?? voiceRecord.map { voiceProfile(from: $0) }
        let enrichedStreamVoiceProfile: VoiceProfile? = baseStreamVoiceProfile.map { profile in
            guard let record = voiceRecord else { return profile }
            return Self.enrichWithTADAConditioning(profile, record: record, runtime: runtime)
        }
        let effectiveLanguage = resolvedRequestedVoice.vibeVoiceSelection?.effectiveLanguage ?? payload.language
        let synthesisRequest = SpeechSynthesisRequest(
            model: descriptor.id,
            text: input,
            voice: enrichedStreamVoiceProfile,
            language: effectiveLanguage,
            referenceAudioAssetName: parsedRequestTimeReferenceAudio == nil
                ? voiceRecord?.referenceAudioAssetName
                : nil,
            referenceAudioPCMFloat32LE: promptPayload?.pcmData,
            referenceAudioSamples: promptPayload?.monoSamples,
            referenceAudioSampleRate: promptPayload?.sampleRate,
            referenceTranscript: promptPayload?.transcript,
            instruct: trimmedOrNil(payload.instruct),
            exaggeration: payload.exaggeration,
            cfgWeight: payload.cfg_weight,
            sampleRate: descriptor.defaultSampleRate ?? 24_000,
            responseFormat: "pcm_f32le",
            temperature: payload.temperature,
            topP: payload.top_p,
            repetitionPenalty: payload.repetition_penalty,
            repetitionContextSize: payload.repetition_context_size,
            maxTokens: payload.max_tokens,
            voiceBehavior: payload.voice_behavior ?? .auto
        )
        let requestID = UUID()
        await runtime.activeSynthesisTracker.begin(
            requestID: requestID,
            modelID: descriptor.id.rawValue,
            voiceBehavior: synthesisRequest.voiceBehavior.rawValue,
            executionMode: SynthesisExecutionMode.oneShot.rawValue
        )

        let sseBody = ResponseBody(contentLength: nil) { writer in
            let encoder = JSONEncoder()
            var chunkIndex = 0
            var outputSampleRate = descriptor.defaultSampleRate ?? 24_000
            var terminalState: ActiveSynthesisTerminalState = .completed
            var terminalMessage: String?
            await withTaskCancellationHandler {
                do {
                    let observer = Self.synthesisObserver(runtime: runtime, requestID: requestID)
                    let audioStream = try await SynthesisExecutionObserverContext.$observer.withValue(observer) {
                        try await runtime.withReservedTextToSpeechWorkflowSessionStream(
                            descriptor: descriptor,
                            configuration: configuration
                        ) { reserved in
                            try await reserved.workflow.synthesizeStream(request: synthesisRequest)
                        }
                    }

                    let startedEvent = SpeechStreamStartedEvent(
                        model: descriptor.id.rawValue,
                        sampleRate: Int(outputSampleRate),
                        effectiveVoiceID: resolvedRequestedVoice.vibeVoiceSelection?.effectiveVoice,
                        effectiveLanguage: resolvedRequestedVoice.vibeVoiceSelection?.effectiveLanguage,
                        voiceSelectionMode: resolvedRequestedVoice.vibeVoiceSelection?.selectionMode.rawValue
                    )
                    try await writer.write(sseEventBuffer(
                        event: "started",
                        payload: try encoder.encode(startedEvent)
                    ))

                    for try await chunk in audioStream {
                        outputSampleRate = chunk.sampleRate
                        let pcmData = audioPCMFloat32LEData(from: chunk.samples)
                        let event = SpeechChunkEvent(
                            index: chunkIndex,
                            data: pcmData.base64EncodedString(),
                            sampleRate: Int(chunk.sampleRate)
                        )
                        try await writer.write(sseEventBuffer(
                            event: "chunk",
                            payload: try encoder.encode(event)
                        ))
                        chunkIndex += 1
                    }

                    let completeEvent = SpeechStreamCompleteEvent(
                        model: descriptor.id.rawValue,
                        sampleRate: Int(outputSampleRate),
                        totalChunks: chunkIndex
                    )
                    try await writer.write(sseEventBuffer(
                        event: "complete",
                        payload: try encoder.encode(completeEvent)
                    ))
                } catch {
                    let msg = error.localizedDescription
                        .replacingOccurrences(of: "\\", with: "\\\\")
                        .replacingOccurrences(of: "\r", with: "\\r")
                        .replacingOccurrences(of: "\n", with: "\\n")
                        .replacingOccurrences(of: "\"", with: "'")
                    var errBuf = ByteBuffer()
                    errBuf.writeString("event: error\ndata: {\"message\":\"\(msg)\"}\n\n")
                    try? await writer.write(errBuf)
                    terminalState = Self.synthesisTerminalState(for: error)
                    terminalMessage = error.localizedDescription
                }

                await runtime.activeSynthesisTracker.finish(
                    requestID: requestID,
                    terminalState: terminalState,
                    message: terminalMessage
                )
                try? await writer.finish(nil)
            } onCancel: {
                Task {
                    await runtime.activeSynthesisTracker.finish(
                        requestID: requestID,
                        terminalState: .cancelled,
                        message: "Speech stream body cancelled."
                    )
                }
            }
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream; charset=utf-8"
        headers[.cacheControl] = "no-cache"
        applyVoiceSelectionHeaders(to: &headers, vibeVoiceSelection: resolvedRequestedVoice.vibeVoiceSelection)
        // X-Accel-Buffering: no — omitted (requires HTTPTypes custom name init)
        return Response(status: .ok, headers: headers, body: sseBody)
    }

    // MARK: - Transcription Stream

    private static func handleTranscriptionStreamRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        // Parse body
        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: maxMultipartBodyBytes)
            body = Data(buffer: buffer)
        } catch {
            return errorResponse(
                "Failed to read request body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        let multipart: MultipartFormData
        do {
            multipart = try MultipartFormData.parse(
                body: body,
                contentType: request.headers[.contentType] ?? ""
            )
        } catch {
            return errorResponse(error.localizedDescription, status: .badRequest)
        }

        guard let file = multipart.files["file"] else {
            return errorResponse("Multipart field 'file' is required.", status: .badRequest)
        }

        let languageHint = trimmedOrNil(multipart.fields["language"])

        let descriptor: ModelDescriptor
        do {
            descriptor = try await resolveTranscriptionDescriptor(
                requestedModel: multipart.fields["model"],
                runtime: runtime
            )
        } catch {
            return errorResponse(error.localizedDescription, status: status(for: error))
        }

        let audioChunk: AudioChunk
        do {
            audioChunk = try await decodeAudioChunk(
                data: file.data,
                hint: file.filename.flatMap(pathExtension(from:)),
                targetSampleRate: descriptor.defaultSampleRate,
                runtime: runtime
            )
        } catch {
            return errorResponse(error.localizedDescription, status: .badRequest)
        }

        // Build streaming SSE response. All transcription work happens inside
        // the ResponseBody closure so headers are sent promptly.
        let sseBody = ResponseBody { writer in
            let encoder = JSONEncoder()
            let durationSeconds = audioChunk.sampleRate > 0
                ? Double(audioChunk.samples.count) / audioChunk.sampleRate
                : 0.0

            do {
                if durationSeconds < 10.0 {
                    // --- Single-chunk path: one transcription call, emit final + complete ---
                    let result = try await runtime.transcribe(
                        SpeechRecognitionRequest(
                            model: descriptor.id,
                            audio: audioChunk,
                            languageHint: languageHint
                        )
                    )
                    let finalSeg = TranscriptionSegment(
                        text: result.text,
                        startTime: result.segments.first?.startTime,
                        endTime: result.segments.last?.endTime,
                        confidence: 1.0,
                        isFinal: true
                    )
                    try await writer.write(sseEventBuffer(
                        event: "final",
                        payload: try encoder.encode(finalSeg)
                    ))
                    try await writer.write(sseEventBuffer(
                        event: "complete",
                        payload: try encoder.encode(result)
                    ))
                } else {
                    // --- Multi-chunk path: VAD → ASRChunkScheduler → TranscriptionMerger ---
                    let asrRate = Int(audioChunk.sampleRate)
                    let vadChunkSize: Int
                    let speechProbabilities: [Float]

                    // Try Silero VAD; fall back to all-speech if model unavailable.
                    do {
                        let vad = try await SileroVADSession()
                        let vadRate = await vad.sampleRate
                        let rawVadChunk = await vad.chunkSize

                        let vadSamples: [Float]
                        if asrRate != vadRate {
                            let resampled = try await runtime.audioPipeline.resample(
                                AudioPCMBuffer(mono: audioChunk.samples, sampleRate: audioChunk.sampleRate),
                                to: Double(vadRate)
                            )
                            vadSamples = resampled.channels.first ?? []
                        } else {
                            vadSamples = audioChunk.samples
                        }

                        var probs: [Float] = []
                        var pos = 0
                        while pos < vadSamples.count {
                            let end = min(pos + rawVadChunk, vadSamples.count)
                            let slice = Array(vadSamples[pos..<end])
                            let vadResult = try await vad.process(chunk: slice)
                            probs.append(vadResult.speechProbability)
                            pos += rawVadChunk
                        }
                        speechProbabilities = probs
                        vadChunkSize = asrRate == vadRate
                            ? rawVadChunk
                            : rawVadChunk * asrRate / vadRate
                    } catch {
                        // VAD unavailable — treat all audio as speech and chunk by duration.
                        let fallbackChunk = 4096 // ~256 ms at 16 kHz
                        let count = max(1, (audioChunk.samples.count + fallbackChunk - 1) / fallbackChunk)
                        speechProbabilities = [Float](repeating: 1.0, count: count)
                        vadChunkSize = fallbackChunk
                    }

                    let scheduler = ASRChunkScheduler()
                    var asrChunks = scheduler.schedule(
                        audio: audioChunk.samples,
                        speechProbabilities: speechProbabilities,
                        sampleRate: asrRate,
                        vadChunkSize: vadChunkSize
                    )

                    // Fully silent audio → treat the whole buffer as one chunk.
                    if asrChunks.isEmpty {
                        asrChunks = [
                            ASRChunk(
                                index: 0,
                                samples: audioChunk.samples,
                                overlapStartSample: 0,
                                contentStartSample: 0,
                                contentEndSample: audioChunk.samples.count
                            )
                        ]
                    }

                    let scheduledChunks = asrChunks
                    let configuration = try BackendSelectionPolicy().runtimeConfiguration(
                        for: descriptor,
                        runtime: BackendSelectionPolicy.Runtime(
                            availableBackends: [runtime.inferenceBackend.backendKind]
                        )
                    )
                    do {
                        let stream = try await runtime.withReservedSpeechToTextWorkflowSessionStream(
                            descriptor: descriptor,
                            configuration: configuration
                        ) { reservedWorkflow in
                            AsyncThrowingStream { streamContinuation in
                                Task {
                                    do {
                                        let merger = TranscriptionMerger()
                                        var lastMeta: BackendMetadata?

                                        for asrChunk in scheduledChunks {
                                            try Task.checkCancellation()
                                            let chunkAudio = AudioChunk(
                                                samples: asrChunk.samples,
                                                sampleRate: audioChunk.sampleRate
                                            )
                                            let chunkReq = SpeechRecognitionRequest(
                                                model: descriptor.id,
                                                audio: chunkAudio,
                                                languageHint: languageHint
                                            )
                                            let chunkResult = try await reservedWorkflow.workflow.transcribe(
                                                request: chunkReq,
                                                in: reservedWorkflow.session
                                            )
                                            lastMeta = chunkResult.backendMetadata

                                            let events = await merger.merge(
                                                chunk: asrChunk,
                                                result: chunkResult,
                                                sampleRate: asrRate
                                            )
                                            for event in events {
                                                streamContinuation.yield(event)
                                            }
                                        }

                                        let meta = lastMeta ?? BackendMetadata(
                                            modelId: descriptor.id.rawValue,
                                            backendKind: .mlx
                                        )
                                        let completedEvent = await merger.finalize(
                                            language: languageHint,
                                            backendMetadata: meta
                                        )
                                        streamContinuation.yield(completedEvent)
                                        streamContinuation.finish()
                                    } catch is CancellationError {
                                        streamContinuation.finish(throwing: CancellationError())
                                    } catch {
                                        streamContinuation.finish(throwing: error)
                                    }
                                }
                            }
                        }
                        for try await event in stream {
                            if let buf = try sseBuffer(for: event, encoder: encoder) {
                                try await writer.write(buf)
                            }
                        }
                    } catch let error as WorkflowReservationError {
                        if case .unsupportedSpeechToText = error {
                            throw TranscriptionStreamError.modelNotSpeechToText(descriptor.id.rawValue)
                        }
                        throw error
                    }
                }
            } catch {
                let msg = error.localizedDescription
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\r", with: "\\r")
                    .replacingOccurrences(of: "\n", with: "\\n")
                    .replacingOccurrences(of: "\"", with: "'")
                var errBuf = ByteBuffer()
                errBuf.writeString("event: error\ndata: {\"message\":\"\(msg)\"}\n\n")
                try? await writer.write(errBuf)
            }
            try await writer.finish(nil)
        }

        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream; charset=utf-8"
        headers[.cacheControl] = "no-cache"
        // X-Accel-Buffering: no — omitted (requires HTTPTypes custom name init)
        return Response(status: .ok, headers: headers, body: sseBody)
    }

    /// Format a single SSE event frame as a ByteBuffer.
    private static func sseEventBuffer(event: String, payload: Data) -> ByteBuffer {
        var buf = ByteBuffer()
        buf.writeString("event: \(event)\ndata: ")
        buf.writeBytes(payload)
        buf.writeString("\n\n")
        return buf
    }

    /// Map a `SpeechRecognitionEvent` to an SSE ByteBuffer.
    /// Returns `nil` for event types not exposed on this endpoint (metrics, warning).
    private static func sseBuffer(
        for event: SpeechRecognitionEvent,
        encoder: JSONEncoder
    ) throws -> ByteBuffer? {
        switch event {
        case .partial(let seg):
            return sseEventBuffer(event: "partial", payload: try encoder.encode(seg))
        case .finalSegment(let seg):
            return sseEventBuffer(event: "final", payload: try encoder.encode(seg))
        case .completed(let result):
            return sseEventBuffer(event: "complete", payload: try encoder.encode(result))
        case .metrics, .warning:
            return nil
        }
    }

    private static func synthesisObserver(
        runtime: ValarRuntime,
        requestID: UUID
    ) -> @Sendable (SynthesisExecutionEvent) -> Void {
        { event in
            Task {
                await runtime.activeSynthesisTracker.heartbeat(
                    requestID: requestID,
                    executionMode: event.executionMode.rawValue,
                    segmentIndex: event.segmentIndex,
                    segmentCount: event.segmentCount,
                    usesAnchorConditioning: event.usesAnchorConditioning,
                    chunkCharacterCount: event.chunkCharacterCount,
                    generatedTokenCount: event.generatedTokenCount,
                    maxTokenCount: event.maxTokenCount,
                    prefillTokenCount: event.prefillTokenCount,
                    segmentPrefillTimeSeconds: event.segmentPrefillTimeSeconds,
                    segmentDecodeTimeSeconds: event.segmentDecodeTimeSeconds,
                    anchorSegmentDecodeTimeSeconds: event.anchorSegmentDecodeTimeSeconds,
                    continuationSegmentDecodeTimeSeconds: event.continuationSegmentDecodeTimeSeconds,
                    samplingTimeSeconds: event.samplingTimeSeconds,
                    evalTimeSeconds: event.evalTimeSeconds,
                    tokenMaterializationTimeSeconds: event.tokenMaterializationTimeSeconds,
                    embeddingAssemblyTimeSeconds: event.embeddingAssemblyTimeSeconds,
                    talkerForwardTimeSeconds: event.talkerForwardTimeSeconds,
                    codePredictorTimeSeconds: event.codePredictorTimeSeconds,
                    segmentWallTimeSeconds: event.segmentWallTimeSeconds,
                    segmentAudioDurationSeconds: event.segmentAudioDurationSeconds,
                    continuationOutlier: event.continuationOutlier
                )
            }
        }
    }

    private static func synthesisTerminalState(for error: Error) -> ActiveSynthesisTerminalState {
        if error is CancellationError {
            return .cancelled
        }
        if let mlxError = error as? MLXBackendError,
           case .inferenceError(let message) = mlxError,
           message.localizedCaseInsensitiveContains("stalled with no progress") {
            return .stalled
        }
        if error.localizedDescription.localizedCaseInsensitiveContains("stalled with no progress") {
            return .stalled
        }
        return .failed
    }

    private static func withCancellationOnClientDisconnect<Value: Sendable>(
        channel: any Channel,
        operation: @Sendable @escaping () async throws -> Value
    ) async throws -> Value {
        let disconnectFuture = await ClientInputCloseRegistry.shared.future(for: channel) ?? channel.closeFuture
        let task = Task {
            try await operation()
        }
        disconnectFuture.whenComplete { _ in
            task.cancel()
        }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    private static func audioPCMFloat32LEData(from samples: [Float]) -> Data {
        guard !samples.isEmpty else { return Data() }
        return samples.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.size)
        }
    }

    private static func transcriptionResponse(
        _ result: RichTranscriptionResult,
        format: TranscriptionResponseFormat
    ) throws -> Response {
        switch format {
        case .json:
            return try jsonResponse(TranscriptionResponse(text: result.text))
        case .verbose_json:
            return try jsonResponse(result)
        case .text, .srt, .vtt:
            var body = ByteBuffer()
            body.writeString(try format.render(result))
            return Response(
                status: .ok,
                headers: [.contentType: format.contentType],
                body: .init(byteBuffer: body)
            )
        }
    }

    private static func pathExtension(from filename: String) -> String? {
        let ext = URL(fileURLWithPath: filename).pathExtension
        return ext.isEmpty ? nil : ext
    }
}

private struct RequestTimeReferenceAudio {
    let data: Data
    let hint: String?
    let transcript: String?
}

private struct SpeechRequest: Decodable {
    let input: String
    let model: String?
    let voice: String?
    let voice_id: String?
    let voice_behavior: SpeechSynthesisVoiceBehavior?
    let language: String?
    let instruct: String?
    let exaggeration: Float?
    let cfg_weight: Float?
    let response_format: String?
    let temperature: Float?
    let top_p: Float?
    let repetition_penalty: Float?
    let repetition_context_size: Int?
    let max_tokens: Int?
    let reference_audio: String?
    let reference_transcript: String?
}

private struct TranscriptionResponse: Encodable {
    let text: String
}

private struct SpeechChunkEvent: Encodable {
    let index: Int
    /// Base64-encoded PCM float32 LE audio samples.
    let data: String
    let sampleRate: Int
}

private struct SpeechStreamStartedEvent: Encodable {
    let model: String
    let sampleRate: Int
    let effectiveVoiceID: String?
    let effectiveLanguage: String?
    let voiceSelectionMode: String?
}

private struct SpeechStreamCompleteEvent: Encodable {
    let model: String
    let sampleRate: Int
    let totalChunks: Int
}

private struct MultipartFormData {
    let fields: [String: String]
    let files: [String: MultipartFile]

    static func parse(body: Data, contentType: String) throws -> MultipartFormData {
        guard let boundary = multipartBoundary(from: contentType) else {
            throw DaemonRequestError.invalidMultipart("Missing multipart boundary.")
        }

        let delimiter = Data(("--" + boundary).utf8)
        let terminal = Data("--".utf8)
        let parts = split(body, by: delimiter)
        var fields: [String: String] = [:]
        var files: [String: MultipartFile] = [:]

        for rawPart in parts.dropFirst() {
            var part = trimLeadingCRLF(from: rawPart)
            if part.starts(with: terminal) {
                continue
            }
            part = trimTrailingCRLF(from: part)
            guard part.isEmpty == false else {
                continue
            }

            let separator = Data("\r\n\r\n".utf8)
            guard let separatorRange = part.range(of: separator) else {
                throw DaemonRequestError.invalidMultipart("Malformed multipart section.")
            }

            let headerData = part.subdata(in: part.startIndex ..< separatorRange.lowerBound)
            let valueData = part.subdata(in: separatorRange.upperBound ..< part.endIndex)
            guard let headerText = String(data: headerData, encoding: .utf8) else {
                throw DaemonRequestError.invalidMultipart("Multipart headers are not valid UTF-8.")
            }

            var dispositionAttributes: [String: String] = [:]
            var contentTypeHeader: String?

            for line in headerText.components(separatedBy: "\r\n") {
                guard let separatorIndex = line.firstIndex(of: ":") else {
                    continue
                }
                let name = line[..<separatorIndex]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                let value = line[line.index(after: separatorIndex)...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if name == "content-disposition" {
                    dispositionAttributes = parseContentDisposition(value)
                } else if name == "content-type" {
                    contentTypeHeader = value
                }
            }

            guard let fieldName = dispositionAttributes["name"] else {
                throw DaemonRequestError.invalidMultipart("Multipart field is missing a name.")
            }

            if let filename = dispositionAttributes["filename"] {
                files[fieldName] = MultipartFile(
                    filename: filename,
                    contentType: contentTypeHeader,
                    data: valueData
                )
            } else {
                let text = String(data: valueData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                fields[fieldName] = text
            }
        }

        return MultipartFormData(fields: fields, files: files)
    }

    private static func multipartBoundary(from contentType: String) -> String? {
        for segment in contentType.components(separatedBy: ";") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("boundary=") else {
                continue
            }
            let value = String(trimmed.dropFirst("boundary=".count))
            let boundary = value.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            guard boundary.count <= 70 else {
                return nil  // RFC 2046 limits boundary to 70 chars
            }
            return boundary
        }
        return nil
    }

    private static func parseContentDisposition(_ value: String) -> [String: String] {
        var attributes: [String: String] = [:]
        for segment in value.components(separatedBy: ";") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let separator = trimmed.firstIndex(of: "=") else {
                continue
            }
            let key = trimmed[..<separator]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            let rawValue = trimmed[trimmed.index(after: separator)...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            attributes[key] = rawValue.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return attributes
    }

    private static func split(_ data: Data, by separator: Data) -> [Data] {
        guard separator.isEmpty == false else {
            return [data]
        }

        var parts: [Data] = []
        var searchStart = data.startIndex
        while let range = data.range(of: separator, options: [], in: searchStart ..< data.endIndex) {
            parts.append(data.subdata(in: searchStart ..< range.lowerBound))
            searchStart = range.upperBound
        }
        parts.append(data.subdata(in: searchStart ..< data.endIndex))
        return parts
    }

    private static func trimLeadingCRLF(from data: Data) -> Data {
        if data.starts(with: Data("\r\n".utf8)) {
            return data.subdata(in: data.index(data.startIndex, offsetBy: 2) ..< data.endIndex)
        }
        return data
    }

    private static func trimTrailingCRLF(from data: Data) -> Data {
        if data.count >= 2 && data.suffix(2) == Data("\r\n".utf8) {
            return data.subdata(in: data.startIndex ..< data.index(data.endIndex, offsetBy: -2))
        }
        return data
    }
}

private struct MultipartFile {
    let filename: String?
    let contentType: String?
    let data: Data
}

private enum TranscriptionStreamError: LocalizedError {
    case modelNotSpeechToText(String)

    var errorDescription: String? {
        switch self {
        case .modelNotSpeechToText(let id):
            return "Model '\(id)' does not support speech-to-text transcription."
        }
    }
}

private enum DaemonRequestError: LocalizedError {
    case invalidMultipart(String)
    case invalidReferenceAudio(String)
    case missingModel(String)
    case modelNotInstalled(String, hint: String?)
    case missingDefaultModel(ModelCapability, hint: String?)
    case invalidVoice(String)
    case unsupportedVibeVoiceLanguage(String)
    case explicitVoiceLanguageMismatch(
        voice: String,
        requestedLanguage: String,
        presetLanguage: String,
        suggestedVoice: String?
    )
    case missingVoice(String)
    case missingReferenceAudio(String)
    case referenceAudioTooLarge(Int)
    case transcriptRequiredForNonEnglishTADACloning
    case unsupportedOperation(String)

    var httpStatus: HTTPResponse.Status {
        switch self {
        case .invalidMultipart,
             .invalidReferenceAudio,
             .missingModel,
             .modelNotInstalled,
             .missingDefaultModel,
             .invalidVoice,
             .unsupportedVibeVoiceLanguage,
             .explicitVoiceLanguageMismatch,
             .missingVoice,
             .missingReferenceAudio,
             .referenceAudioTooLarge,
             .transcriptRequiredForNonEnglishTADACloning,
             .unsupportedOperation:
            return .badRequest
        }
    }

    var errorKind: String {
        switch self {
        case .explicitVoiceLanguageMismatch:
            return "explicit_voice_language_mismatch"
        default:
            return "audio_error"
        }
    }

    var errorDescription: String? {
        switch self {
        case .invalidMultipart(let message):
            return message
        case .invalidReferenceAudio(let message):
            return message
        case .missingModel(let modelID):
            if let hiddenReason = CatalogVisibilityPolicy.currentProcess().hiddenReason(for: ModelIdentifier(modelID)) {
                return hiddenReason
            }
            return "Model '\(modelID)' was not found in the runtime catalog."
        case .modelNotInstalled(let requestedModel, let hint):
            let resolvedHint = hint ?? requestedModel
            return "Model '\(requestedModel)' is supported but not installed. Run: valartts models install \(resolvedHint)"
        case .missingDefaultModel(let capability, let hint):
            let base = "No \(capability.rawValue) model is installed."
            if let hint {
                return "\(base) Install one with: valartts models install \(hint)"
            }
            return "\(base) Run: valartts models list to see available models."
        case .invalidVoice(let identifier):
            return "Voice '\(identifier)' not found. Use a stored voice UUID or a preset supported by the selected model."
        case .unsupportedVibeVoiceLanguage(let language):
            let supported = VibeVoiceCatalog.supportedLanguageCodes.joined(separator: ", ")
            return "Language '\(language)' is not supported by VibeVoice. Supported values: \(supported)."
        case let .explicitVoiceLanguageMismatch(voice, requestedLanguage, presetLanguage, suggestedVoice):
            let suggestion: String
            if let suggestedVoice {
                suggestion = " Try '\(suggestedVoice)' for language '\(requestedLanguage)'."
            } else {
                suggestion = ""
            }
            return "Voice '\(voice)' is a \(presetLanguage) preset and cannot be used with language '\(requestedLanguage)'.\(suggestion)"
        case .missingVoice(let identifier):
            return "Voice '\(identifier)' was not found in the voice library."
        case .missingReferenceAudio(let assetName):
            return "Voice reference audio '\(assetName)' is missing from the voice library."
        case .referenceAudioTooLarge(let size):
            return "Voice reference audio is too large (\(size) bytes); maximum is 50 MB."
        case .transcriptRequiredForNonEnglishTADACloning:
            return "Transcript required for non-English TADA cloning. Provide --reference-transcript or install a speech recognition model."
        case .unsupportedOperation(let message):
            return message
        }
    }
}

private extension DaemonRequestError {
    init(vibeVoiceResolutionError: VibeVoiceRequestResolutionError) {
        switch vibeVoiceResolutionError {
        case .invalidVoice(let voice):
            self = .invalidVoice(voice)
        case .unsupportedLanguage(let language):
            self = .unsupportedVibeVoiceLanguage(language)
        case let .explicitVoiceLanguageMismatch(voice, requestedLanguage, presetLanguage, suggestedVoice):
            self = .explicitVoiceLanguageMismatch(
                voice: voice,
                requestedLanguage: requestedLanguage,
                presetLanguage: presetLanguage,
                suggestedVoice: suggestedVoice
            )
        }
    }
}
