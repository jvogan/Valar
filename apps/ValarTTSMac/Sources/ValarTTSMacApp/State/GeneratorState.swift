import os
import SwiftUI
import ValarAudio
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence

@Observable
@MainActor
final class GeneratorState {
    private static let logger = Logger(subsystem: "com.valar.tts", category: "Generator")
    private static let maxAccumulatedFrames = 24_000 * 3_600

    var text: String = ""
    var selectedModelID: ModelIdentifier?
    var selectedVoiceID: UUID?
    var selectedLanguage: String = "auto"
    var selectedReferenceAudioURL: URL?
    var referenceTranscript: String = ""
    var isGenerating: Bool = false
    var generationProgress: Double = 0
    var availableModels: [RuntimeModelPickerOption] = []
    var availableVoices: [VoiceLibraryRecord] = []
    var voiceBehavior: SpeechSynthesisVoiceBehavior = .auto

    // Audio
    var hasAudio: Bool = false
    var audioDuration: Double = 0
    var playbackPosition: Double = 0
    var isPlaying: Bool = false
    var isPlaybackBuffering: Bool = false
    var waveformSamples: [Float] = []

    // Error
    var errorMessage: String?

    // Generation params
    var temperature: Double = 0.7
    var topP: Double = 0.95
    var topK: Int = 50
    var repetitionPenalty: Double = 1.0
    var maxTokens: Int = 8192

    // Draft history
    var drafts: [DraftEntry] = []

    // Undo/redo presentation
    private(set) var canUndo = false
    private(set) var canRedo = false
    private(set) var undoMenuTitle = "Undo"
    private(set) var redoMenuTitle = "Redo"

    // Computed
    var wordCount: Int { text.wordCount }
    var canGenerate: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isGenerating
            && (selectedModelID != nil || selectedVoice?.preferredRuntimeModelIdentifier != nil)
            && (!requiresInlineReferenceAudio || selectedReferenceAudioURL != nil)
    }
    var canSaveGeneratedAudio: Bool { (generatedAudioBuffer?.frameCount ?? 0) > 0 }
    var suggestedSaveAudioFilename: String { "\(sanitizedAudioFileStem()).wav" }
    var onSelectionChange: (@MainActor (ModelIdentifier?) -> Void)?

    struct DraftEntry: Identifiable, Equatable {
        let id = UUID()
        let text: String
        let modelID: ModelIdentifier?
        let timestamp: Date
    }

    private let services: ValarServiceHub
    private let modelDiscoveryServices: ValarServiceHub
    private let playbackPollInterval: Duration
    private let textUndoCoalescingWindow: Duration
    private var generatedAudioBuffer: AudioPCMBuffer?
    private var playbackMonitorTask: Task<Void, Never>?
    private var playbackCommandID: UInt64 = 0
    private let editorUndoManager = UndoManager()
    private var isTextUndoSessionActive = false
    private var textUndoResetTask: Task<Void, Never>?
    // Ring buffer used as a staging area for streaming audio chunks.
    // Capacity covers any single chunk a TTS model is likely to emit
    // (~2.7 s at 24 kHz); if a chunk exceeds this, samples are fed directly.
    private let streamingRingBuffer: SPSCFloatRingBuffer

    init(
        services: ValarServiceHub,
        modelDiscoveryServices: ValarServiceHub? = nil,
        playbackPollInterval: Duration = .milliseconds(50),
        textUndoCoalescingWindow: Duration = .milliseconds(500)
    ) {
        self.services = services
        self.modelDiscoveryServices = modelDiscoveryServices ?? services
        self.playbackPollInterval = playbackPollInterval
        self.textUndoCoalescingWindow = textUndoCoalescingWindow
        self.streamingRingBuffer = SPSCFloatRingBuffer(minimumCapacity: 65_536)
        updateUndoPresentation()
    }

    var undoManager: UndoManager {
        editorUndoManager
    }

    func reloadRuntimeOptions(selectedModelID: ModelIdentifier?) async {
        let availableModels = await modelDiscoveryServices.runtime.generationModelOptions()
        let runtimeVoices = await services.runtime.listVoices()
        let availableModelIDs = Set(availableModels.map(\.id))
        let selectedVoice = self.selectedVoiceID.flatMap { voiceID in
            runtimeVoices.first(where: { $0.id == voiceID })
        }

        let resolvedSelectedModelID: ModelIdentifier?
        if let selectedVoice,
           let runtimeModelID = selectedVoice.preferredRuntimeModelIdentifier {
            resolvedSelectedModelID = runtimeModelID
        } else if let selectedModelID,
                  availableModelIDs.contains(selectedModelID) {
            resolvedSelectedModelID = selectedModelID
        } else {
            resolvedSelectedModelID = availableModels.first?.id
        }

        let availableVoices = Self.supplementedVoices(
            from: runtimeVoices,
            availableModels: availableModels
        )

        self.availableModels = availableModels
        self.availableVoices = availableVoices

        if let selectedVoiceID,
           !availableVoices.contains(where: { $0.id == selectedVoiceID }) {
            self.selectedVoiceID = nil
        }

        self.selectedModelID = resolvedSelectedModelID
        notifySelectionChange()
    }

    func generate() async {
        guard canGenerate else { return }
        clearUndoHistory()
        errorMessage = nil
        isGenerating = true
        generationProgress = 0
        playbackCommandID &+= 1
        cancelPlaybackMonitoring()
        await services.audioPlayer.stop()
        resetGeneratedAudio()

        // Save draft
        drafts.insert(DraftEntry(text: text, modelID: selectedModelID, timestamp: .now), at: 0)
        if drafts.count > 20 { drafts = Array(drafts.prefix(20)) }
        defer { isGenerating = false }

        let activeModelID = selectedVoice?.preferredRuntimeModelIdentifier ?? selectedModelID
        guard let activeModelID else {
            errorMessage = "No model selected. Please choose a model before generating."
            resetGeneratedAudio()
            return
        }

        if requiresInlineReferenceAudio, selectedReferenceAudioURL == nil {
            errorMessage = "Reference audio is required for the selected TADA model."
            resetGeneratedAudio()
            return
        }

        do {
            let preparedSelectedVoice = try await services.voiceReadyForSynthesis(selectedVoice)
            let descriptor = try await services.descriptor(for: activeModelID)
            let runtime = BackendSelectionPolicy.Runtime(
                availableBackends: [services.inferenceBackend.backendKind]
            )
            let configuration = try BackendSelectionPolicy().runtimeConfiguration(
                for: descriptor,
                residencyPolicy: .automatic,
                runtime: runtime
            )
            generationProgress = 0.35

            let voicePromptPayload: (pcmData: Data, sampleRate: Double, transcript: String)? = if let preparedSelectedVoice {
                try await services.referencePromptPayload(for: preparedSelectedVoice)
            } else {
                nil
            }
            let inlinePromptPayload: (pcmData: Data, sampleRate: Double, transcript: String?)? = if let selectedReferenceAudioURL,
                preparedSelectedVoice == nil {
                try await services.inlineReferencePromptPayload(
                    audioURL: selectedReferenceAudioURL,
                    transcript: referenceTranscript
                )
            } else {
                nil
            }
            let voiceProfile: VoiceProfile?
            if let preparedSelectedVoice {
                let profile = preparedSelectedVoice.makeVoiceProfile(localeIdentifier: normalizedSelectedLanguage)
                let enrichedProfile = try enrichWithTADAConditioning(profile, record: preparedSelectedVoice)
                try enrichedProfile.validateCompatibility(with: descriptor.id, familyID: descriptor.familyID)
                voiceProfile = enrichedProfile
            } else {
                voiceProfile = nil
            }

            let request = SpeechSynthesisRequest(
                model: activeModelID,
                text: text,
                voice: voiceProfile,
                language: normalizedSelectedLanguage,
                referenceAudioAssetName: inlinePromptPayload == nil
                    ? preparedSelectedVoice?.referenceAudioAssetName
                    : selectedReferenceAudioURL?.lastPathComponent,
                referenceAudioPCMFloat32LE: inlinePromptPayload?.pcmData ?? voicePromptPayload?.pcmData,
                referenceAudioSampleRate: inlinePromptPayload?.sampleRate ?? voicePromptPayload?.sampleRate,
                referenceTranscript: inlinePromptPayload?.transcript ?? voicePromptPayload?.transcript,
                sampleRate: descriptor.defaultSampleRate ?? 24_000,
                responseFormat: "pcm_f32le",
                temperature: Float(temperature),
                topP: Float(topP),
                repetitionPenalty: Float(repetitionPenalty),
                maxTokens: maxTokens,
                voiceBehavior: voiceBehavior
            )
            var accumulatedSamples: [Float] = []
            var totalGeneratedFrames = 0
            var streamSampleRate = request.sampleRate
            var receivedChunkCount = 0
            var didReachAccumulatedSamplesCap = false
            streamingRingBuffer.reset()

            let stream = try await services.runtime.withReservedTextToSpeechWorkflowSessionStream(
                descriptor: descriptor,
                configuration: configuration
            ) { reserved in
                try await reserved.workflow.synthesizeStream(request: request)
            }
            for try await chunk in stream {
                let chunkSamples = chunk.samples
                let chunkSampleRate = chunk.sampleRate
                let waveformBuffer = AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)

                // Stage through the ring buffer and feed via feedSamples.
                let written = streamingRingBuffer.write(chunk.samples)
                if written == chunk.samples.count {
                    let staged = streamingRingBuffer.read(count: written)
                    try await services.audioPlayer.feedSamples(staged, sampleRate: chunk.sampleRate)
                } else {
                    // Ring buffer overflow (chunk larger than capacity): drain and feed directly.
                    streamingRingBuffer.discard(count: streamingRingBuffer.availableToRead)
                    try await services.audioPlayer.feedSamples(chunk.samples, sampleRate: chunk.sampleRate)
                }

                guard !chunkSamples.isEmpty else { continue }

                receivedChunkCount += 1
                totalGeneratedFrames += chunkSamples.count
                streamSampleRate = chunkSampleRate

                let remainingAccumulatedFrames = Self.maxAccumulatedFrames - accumulatedSamples.count
                if remainingAccumulatedFrames > 0 {
                    accumulatedSamples.append(contentsOf: chunkSamples.prefix(remainingAccumulatedFrames))
                    if accumulatedSamples.count == Self.maxAccumulatedFrames,
                       !didReachAccumulatedSamplesCap {
                        didReachAccumulatedSamplesCap = true
                        Self.logger.warning(
                            "Reached accumulated sample cap (\(Self.maxAccumulatedFrames, privacy: .public) frames); retaining only the earliest buffered stream samples."
                        )
                    }
                } else if !didReachAccumulatedSamplesCap {
                    didReachAccumulatedSamplesCap = true
                    Self.logger.warning(
                        "Reached accumulated sample cap (\(Self.maxAccumulatedFrames, privacy: .public) frames); retaining only the earliest buffered stream samples."
                    )
                }

                // Incremental waveform: ~10 buckets per second of audio
                let chunkSeconds = Double(chunkSamples.count) / streamSampleRate
                let chunkBuckets = max(1, Int(chunkSeconds * 10))
                waveformSamples.append(
                    contentsOf: DSWaveformRenderer().waveformSamples(from: waveformBuffer, bucketCount: chunkBuckets)
                )

                hasAudio = true
                audioDuration = Double(totalGeneratedFrames) / streamSampleRate
                generationProgress = min(0.95, 0.35 + (Double(receivedChunkCount) * 0.1))

                if playbackMonitorTask == nil {
                    startPlaybackMonitoring(for: playbackCommandID)
                }
            }

            await services.audioPlayer.finishStream()

            // Build final buffer once from accumulated samples
            if !accumulatedSamples.isEmpty {
                generatedAudioBuffer = AudioPCMBuffer(mono: accumulatedSamples, sampleRate: streamSampleRate)
            }
            if let fullBuffer = generatedAudioBuffer {
                waveformSamples = DSWaveformRenderer().waveformSamples(from: fullBuffer, bucketCount: 200)
            }
            hasAudio = generatedAudioBuffer?.frameCount ?? 0 > 0
            audioDuration = generatedAudioBuffer?.duration ?? audioDuration

            guard hasAudio else {
                cancelPlaybackMonitoring()
                isPlaying = false
                isPlaybackBuffering = false
                generationProgress = 1
                return
            }

            generationProgress = 1
        } catch {
            errorMessage = PathRedaction.redactMessage(error.localizedDescription)
            cancelPlaybackMonitoring()
            await services.audioPlayer.stop()
            resetGeneratedAudio()
            generationProgress = 0
        }
    }

    func togglePlayback() {
        guard let generatedAudioBuffer, hasAudio else { return }

        if isPlaying || isPlaybackBuffering {
            playbackCommandID &+= 1
            Task {
                await services.audioPlayer.stop()
            }
            cancelPlaybackMonitoring()
            isPlaying = false
            isPlaybackBuffering = false
            playbackPosition = 0
            return
        }

        playbackCommandID &+= 1
        let commandID = playbackCommandID
        Task {
            do {
                guard commandID == playbackCommandID else { return }
                try await services.audioPlayer.play(generatedAudioBuffer)
                guard commandID == playbackCommandID else { return }
                await services.audioPlayer.finishStream()
            } catch {
                guard commandID == playbackCommandID else { return }
                Self.logger.error("togglePlayback failed: \(error.localizedDescription, privacy: .private)")
                cancelPlaybackMonitoring()
                isPlaying = false
                isPlaybackBuffering = false
                playbackPosition = 0
            }
        }
        startPlaybackMonitoring(for: commandID)
        playbackPosition = 0
        isPlaying = true
        isPlaybackBuffering = false
    }

    func saveGeneratedAudio(to destinationURL: URL) async throws {
        guard let generatedAudioBuffer, generatedAudioBuffer.frameCount > 0 else {
            throw GeneratorError.noGeneratedAudioToSave
        }

        let exportFormat = AudioFormatDescriptor(
            sampleRate: generatedAudioBuffer.format.sampleRate,
            channelCount: max(generatedAudioBuffer.format.channelCount, 1),
            sampleFormat: .float32,
            interleaved: false,
            container: "wav"
        )

        _ = try await AVFoundationAudioExporter().export(
            generatedAudioBuffer,
            as: exportFormat,
            to: normalizedSaveAudioURL(destinationURL),
            chapterMarkers: []
        )
    }

    func selectVoice(_ voiceID: UUID?) {
        performUndoableMutation(actionName: "Change Voice") {
            selectedVoiceID = voiceID
            if let runtimeModelID = selectedVoice?.preferredRuntimeModelIdentifier {
                selectedModelID = runtimeModelID
            }
        }
    }

    func selectModel(_ modelID: ModelIdentifier?) {
        performUndoableMutation(actionName: "Change Model") {
            selectedVoiceID = nil
            selectedModelID = modelID
        }
    }

    func selectReferenceAudio(_ url: URL?) throws {
        if let url {
            try VoiceCloneFileValidator.validateFileSelection(url)
        }
        selectedReferenceAudioURL = url
    }

    func clearReferenceAudio() {
        selectedReferenceAudioURL = nil
    }

    func restoreDraft(_ draft: DraftEntry) {
        performUndoableMutation(actionName: "Restore Draft") {
            text = draft.text
            selectedModelID = draft.modelID
        }
    }

    func applyTextEdit(_ newText: String) {
        guard newText != text else { return }

        let previousSnapshot = captureUndoSnapshot()
        text = newText

        if !isTextUndoSessionActive {
            isTextUndoSessionActive = true
            registerUndo(from: previousSnapshot, actionName: "Typing")
        } else {
            undoManager.setActionName("Typing")
        }

        scheduleTextUndoReset()
        updateUndoPresentation()
    }

    func performUndo() {
        guard undoManager.canUndo else { return }
        finishTextUndoCoalescing()
        undoManager.undo()
        updateUndoPresentation()
    }

    func performRedo() {
        guard undoManager.canRedo else { return }
        finishTextUndoCoalescing()
        undoManager.redo()
        updateUndoPresentation()
    }

    func clearUndoHistory() {
        finishTextUndoCoalescing()
        undoManager.removeAllActions()
        updateUndoPresentation()
    }

    private func resetGeneratedAudio() {
        generatedAudioBuffer = nil
        hasAudio = false
        audioDuration = 0
        playbackPosition = 0
        isPlaying = false
        isPlaybackBuffering = false
        waveformSamples = []
        streamingRingBuffer.reset()
    }

    private func normalizedSaveAudioURL(_ destinationURL: URL) -> URL {
        guard destinationURL.pathExtension.lowercased() != "wav" else {
            return destinationURL
        }

        return destinationURL.appendingPathExtension("wav")
    }

    private func sanitizedAudioFileStem() -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let collapsed = text
            .prefix(80)
            .unicodeScalars
            .map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { partialResult, character in
                partialResult.append(character)
            }
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let stem = collapsed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
            .lowercased()

        return stem.isEmpty ? "generated-audio" : stem
    }

    private func startPlaybackMonitoring(for commandID: UInt64) {
        cancelPlaybackMonitoring()

        playbackMonitorTask = Task { @MainActor [services, playbackPollInterval] in
            while !Task.isCancelled {
                guard commandID == playbackCommandID else { return }
                let snapshot = await services.audioPlayer.playbackSnapshot()
                guard commandID == playbackCommandID, !Task.isCancelled else { return }
                applyPlaybackSnapshot(snapshot)

                if snapshot.didFinish {
                    playbackPosition = audioDuration
                    if commandID == playbackCommandID {
                        playbackMonitorTask = nil
                    }
                    return
                }

                do {
                    try await Task.sleep(for: playbackPollInterval)
                } catch {
                    if commandID == playbackCommandID {
                        playbackMonitorTask = nil
                    }
                    return
                }
            }

            if commandID == playbackCommandID {
                playbackMonitorTask = nil
            }
        }
    }

    private func cancelPlaybackMonitoring() {
        playbackMonitorTask?.cancel()
        playbackMonitorTask = nil
    }

    private func applyPlaybackSnapshot(_ snapshot: AudioPlaybackSnapshot) {
        if audioDuration > 0 {
            playbackPosition = min(snapshot.position, audioDuration)
        } else {
            playbackPosition = snapshot.position
        }

        if snapshot.didFinish {
            isPlaying = false
            isPlaybackBuffering = false
            return
        }

        isPlaying = snapshot.isPlaying
        isPlaybackBuffering = snapshot.isBuffering
    }

    var selectedVoice: VoiceLibraryRecord? {
        guard let selectedVoiceID else { return nil }
        return availableVoices.first(where: { $0.id == selectedVoiceID })
    }

    var selectedModelOption: RuntimeModelPickerOption? {
        let activeModelID = selectedVoice?.preferredRuntimeModelIdentifier ?? selectedModelID
        guard let activeModelID else { return nil }
        return availableModels.first(where: { $0.id == activeModelID })
    }

    var showsInlineReferenceAudioControls: Bool {
        guard selectedVoice == nil else { return false }
        return selectedModelOption?.supportsReferenceAudio == true
    }

    var requiresInlineReferenceAudio: Bool {
        guard selectedVoice == nil else { return false }
        return selectedModelOption?.familyID == .tadaTTS
    }

    var normalizedSelectedLanguage: String? {
        guard selectedLanguage != "auto" else { return nil }
        return selectedLanguage.lowercased()
    }

    private struct UndoSnapshot: Equatable {
        let text: String
        let selectedModelID: ModelIdentifier?
        let selectedVoiceID: UUID?
    }

    private func captureUndoSnapshot() -> UndoSnapshot {
        UndoSnapshot(
            text: text,
            selectedModelID: selectedModelID,
            selectedVoiceID: selectedVoiceID
        )
    }

    private func apply(_ snapshot: UndoSnapshot) {
        text = snapshot.text
        selectedModelID = snapshot.selectedModelID
        selectedVoiceID = snapshot.selectedVoiceID
        notifySelectionChange()
    }

    private func performUndoableMutation(actionName: String, _ mutation: () -> Void) {
        finishTextUndoCoalescing()

        let previousSnapshot = captureUndoSnapshot()
        mutation()

        let currentSnapshot = captureUndoSnapshot()
        guard currentSnapshot != previousSnapshot else {
            updateUndoPresentation()
            return
        }

        if currentSnapshot.selectedModelID != previousSnapshot.selectedModelID
            || currentSnapshot.selectedVoiceID != previousSnapshot.selectedVoiceID {
            notifySelectionChange()
        }

        registerUndo(from: previousSnapshot, actionName: actionName)
        updateUndoPresentation()
    }

    private func restoreUndoSnapshot(_ snapshot: UndoSnapshot, actionName: String) {
        finishTextUndoCoalescing()

        let redoSnapshot = captureUndoSnapshot()
        apply(snapshot)
        registerUndo(from: redoSnapshot, actionName: actionName)
        updateUndoPresentation()
    }

    private func registerUndo(from snapshot: UndoSnapshot, actionName: String) {
        let openedGroup = undoManager.groupingLevel == 0
        if openedGroup {
            undoManager.beginUndoGrouping()
        }
        undoManager.registerUndo(withTarget: self) { target in
            target.restoreUndoSnapshot(snapshot, actionName: actionName)
        }
        undoManager.setActionName(actionName)
        if openedGroup {
            undoManager.endUndoGrouping()
        }
    }

    private func scheduleTextUndoReset() {
        textUndoResetTask?.cancel()
        textUndoResetTask = Task { @MainActor [weak self, textUndoCoalescingWindow] in
            do {
                try await Task.sleep(for: textUndoCoalescingWindow)
            } catch {
                return
            }

            guard !Task.isCancelled else { return }
            self?.finishTextUndoCoalescing()
        }
    }

    private func finishTextUndoCoalescing() {
        textUndoResetTask?.cancel()
        textUndoResetTask = nil
        isTextUndoSessionActive = false
    }

    private func updateUndoPresentation() {
        canUndo = undoManager.canUndo
        canRedo = undoManager.canRedo
        undoMenuTitle = canUndo ? undoManager.undoMenuItemTitle : "Undo"
        redoMenuTitle = canRedo ? undoManager.redoMenuItemTitle : "Redo"
    }

    private func notifySelectionChange() {
        onSelectionChange?(selectedModelID)
    }

    private static let voxtralRandomBackendVoiceID = "random"
    private static let voxtralRandomVoiceLabel = "Random"

    private static func supplementedVoices(
        from voices: [VoiceLibraryRecord],
        availableModels: [RuntimeModelPickerOption]
    ) -> [VoiceLibraryRecord] {
        let voxtralModelIDs = availableModels
            .filter { $0.familyID == .voxtralTTS }
            .map(\.id)
        guard !voxtralModelIDs.isEmpty else {
            return voices
        }

        let randomVoices = voxtralModelIDs
            .sorted { $0.rawValue.localizedCaseInsensitiveCompare($1.rawValue) == .orderedAscending }
            .map { makeVoxtralRandomVoice(for: $0) }

        let filteredVoices = voices.filter { voice in
            !voxtralModelIDs.contains(where: { modelID in
                isVoxtralRandomVoice(voice, modelID: modelID)
            })
        }

        return randomVoices + filteredVoices
    }

    private static func makeVoxtralRandomVoice(for modelID: ModelIdentifier) -> VoiceLibraryRecord {
        VoiceLibraryRecord(
            id: stableSyntheticVoiceIdentifier(seed: "voxtral-random:\(modelID.rawValue)"),
            label: voxtralRandomVoiceLabel,
            modelID: modelID.rawValue,
            runtimeModelID: modelID.rawValue,
            backendVoiceID: voxtralRandomBackendVoiceID
        )
    }

    private static func isVoxtralRandomVoice(
        _ voice: VoiceLibraryRecord,
        modelID: ModelIdentifier
    ) -> Bool {
        guard voice.modelID == modelID.rawValue,
              voice.runtimeModelID == modelID.rawValue,
              voice.backendVoiceID == voxtralRandomBackendVoiceID
        else {
            return false
        }
        return voice.isModelDeclaredPreset
    }

    private static func stableSyntheticVoiceIdentifier(seed: String) -> UUID {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in seed.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }

        let suffix = String(format: "%012llx", hash & 0x000f_ffff_ffff_ffff)
        let candidate = "00000000-0000-0000-0000-\(suffix)"
        return UUID(uuidString: candidate) ?? UUID()
    }

    private func enrichWithTADAConditioning(
        _ profile: VoiceProfile,
        record: VoiceLibraryRecord
    ) throws -> VoiceProfile {
        guard record.conditioningFormat == VoiceLibraryRecord.tadaReferenceConditioningFormat,
              let assetName = record.conditioningAssetName else {
            return profile
        }
        let bundleURL = services.appPaths.voiceLibraryDirectory
            .appendingPathComponent(assetName, isDirectory: true)
        guard let _ = try? ValarAppPaths.validateContainment(bundleURL, within: services.appPaths.voiceLibraryDirectory) else {
            return profile
        }
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            return profile
        }
        let loaded = try TADAConditioningBundleIO.load(from: bundleURL)
        return VoiceProfile(
            id: profile.id,
            label: profile.label,
            backendVoiceID: profile.backendVoiceID,
            sourceModel: profile.sourceModel,
            localeIdentifier: profile.localeIdentifier,
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
}

private enum GeneratorError: LocalizedError {
    case unsupportedWorkflow(ModelIdentifier)
    case noGeneratedAudioToSave

    var errorDescription: String? {
        switch self {
        case let .unsupportedWorkflow(modelIdentifier):
            return "The selected model '\(modelIdentifier.rawValue)' does not support speech synthesis."
        case .noGeneratedAudioToSave:
            return "Generate audio before trying to save it."
        }
    }
}

extension String {
    var wordCount: Int {
        split(whereSeparator: \.isWhitespace).count
    }
}
