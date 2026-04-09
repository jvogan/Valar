import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

struct ProjectRenderCommandAction {
    private let handler: @MainActor @Sendable () -> Void

    init(_ handler: @escaping @MainActor @Sendable () -> Void) {
        self.handler = handler
    }

    @MainActor
    func callAsFunction() {
        handler()
    }
}

private struct ProjectRenderChapterActionKey: FocusedValueKey {
    typealias Value = ProjectRenderCommandAction
}

private struct ProjectCanRenderChapterKey: FocusedValueKey {
    typealias Value = Bool
}

private struct ProjectRenderChapterDisabledReasonKey: FocusedValueKey {
    typealias Value = String
}

private struct ProjectRenderProjectActionKey: FocusedValueKey {
    typealias Value = ProjectRenderCommandAction
}

private struct ProjectCanRenderProjectKey: FocusedValueKey {
    typealias Value = Bool
}

private struct ProjectRenderProjectDisabledReasonKey: FocusedValueKey {
    typealias Value = String
}

private struct ProjectRenderModelIDKey: FocusedValueKey {
    typealias Value = ModelIdentifier
}

private struct ProjectRenderSynthesisOptionsKey: FocusedValueKey {
    typealias Value = RenderSynthesisOptions
}

extension FocusedValues {
    var projectRenderChapterAction: ProjectRenderCommandAction? {
        get { self[ProjectRenderChapterActionKey.self] }
        set { self[ProjectRenderChapterActionKey.self] = newValue }
    }

    var projectCanRenderChapter: Bool? {
        get { self[ProjectCanRenderChapterKey.self] }
        set { self[ProjectCanRenderChapterKey.self] = newValue }
    }

    var projectRenderChapterDisabledReason: String? {
        get { self[ProjectRenderChapterDisabledReasonKey.self] }
        set { self[ProjectRenderChapterDisabledReasonKey.self] = newValue }
    }

    var projectRenderProjectAction: ProjectRenderCommandAction? {
        get { self[ProjectRenderProjectActionKey.self] }
        set { self[ProjectRenderProjectActionKey.self] = newValue }
    }

    var projectCanRenderProject: Bool? {
        get { self[ProjectCanRenderProjectKey.self] }
        set { self[ProjectCanRenderProjectKey.self] = newValue }
    }

    var projectRenderProjectDisabledReason: String? {
        get { self[ProjectRenderProjectDisabledReasonKey.self] }
        set { self[ProjectRenderProjectDisabledReasonKey.self] = newValue }
    }

    var projectRenderModelID: ModelIdentifier? {
        get { self[ProjectRenderModelIDKey.self] }
        set { self[ProjectRenderModelIDKey.self] = newValue }
    }

    var projectRenderSynthesisOptions: RenderSynthesisOptions? {
        get { self[ProjectRenderSynthesisOptionsKey.self] }
        set { self[ProjectRenderSynthesisOptionsKey.self] = newValue }
    }
}

enum ChapterRenderStatus: Sendable, Equatable {
    case idle
    case queued
    case rendering(progress: Double)
    case complete
    case failed

    var isRendering: Bool {
        if case .rendering = self { return true }
        return false
    }

    var progress: Double? {
        switch self {
        case .rendering(let p): return p
        case .complete: return 1.0
        default: return nil
        }
    }
}

struct RenderModelOption: Identifiable, Equatable {
    let id: ModelIdentifier
    let displayName: String
}

struct ProjectUtilityModelOption: Identifiable, Equatable {
    let id: ModelIdentifier
    let displayName: String
}

struct SpeakerEntry: Identifiable, Equatable {
    let id: UUID
    var name: String
    var voiceModelID: ModelIdentifier?
    var language: String = "auto"

    init(
        id: UUID = UUID(),
        name: String,
        voiceModelID: ModelIdentifier?,
        language: String = "auto"
    ) {
        self.id = id
        self.name = name
        self.voiceModelID = voiceModelID
        self.language = language
    }

    init(record: ProjectSpeakerRecord) {
        self.init(
            id: record.id,
            name: record.name,
            voiceModelID: record.voiceModelID.map { ModelIdentifier($0) },
            language: record.language
        )
    }

    func record(projectID: UUID) -> ProjectSpeakerRecord {
        ProjectSpeakerRecord(
            id: id,
            projectID: projectID,
            name: name,
            voiceModelID: voiceModelID?.rawValue,
            language: language
        )
    }
}

@Observable
@MainActor
final class ProjectWorkspaceState {
    var project: ProjectRecord?
    var chapters: [ChapterRecord] = [] {
        didSet {
            guard chapters.count != oldValue.count else { return }
            onChapterCountChange?(chapters.count)
        }
    }
    var selectedChapterID: UUID?
    var speakers: [SpeakerEntry] = []
    var renderJobs: [RenderJob] = []
    var availableRenderModels: [RenderModelOption] = []
    var availableRecognitionModels: [ProjectUtilityModelOption] = []
    var availableAlignmentModels: [ProjectUtilityModelOption] = []
    var selectedRenderModelID: ModelIdentifier?
    var selectedRecognitionModelID: ModelIdentifier?
    var selectedAlignmentModelID: ModelIdentifier?
    var language: String = "auto"
    var chapterAudioLanguageHint: String = ""
    var temperature: Double = 0.7
    var topP: Double = 0.9
    var repetitionPenalty: Double = 1.0
    var maxTokens: Int = 8_192
    var voiceBehavior: SpeechSynthesisVoiceBehavior = .auto
    var isAttachingSourceAudio = false
    var isTranscribingSelectedChapter = false
    var isAligningSelectedChapter = false
    var errorMessage: String?

    private let services: ValarServiceHub
    @ObservationIgnored private let onChapterCountChange: (@MainActor @Sendable (Int) -> Void)?
    private var renderUpdatesTask: Task<Void, Never>?

    init(
        services: ValarServiceHub,
        onChapterCountChange: (@MainActor @Sendable (Int) -> Void)? = nil
    ) {
        self.services = services
        self.onChapterCountChange = onChapterCountChange
    }

    var selectedChapter: ChapterRecord? {
        chapters.first { $0.id == selectedChapterID }
    }

    var selectedChapterTranscription: RichTranscriptionResult? {
        guard let chapter = selectedChapter else { return nil }
        return decodedTranscription(from: chapter)
    }

    var selectedChapterAlignment: ForcedAlignmentResult? {
        guard let chapter = selectedChapter else { return nil }
        return decodedAlignment(from: chapter)
    }

    var selectedRecognitionModelName: String? {
        availableRecognitionModels.first(where: { $0.id == selectedRecognitionModelID })?.displayName
    }

    var selectedAlignmentModelName: String? {
        availableAlignmentModels.first(where: { $0.id == selectedAlignmentModelID })?.displayName
    }

    var isBusyWithChapterAudio: Bool {
        isAttachingSourceAudio || isTranscribingSelectedChapter || isAligningSelectedChapter
    }

    var attachSourceAudioDisabledReason: String? {
        guard selectedChapter != nil else {
            return "Select a chapter before attaching source audio."
        }
        guard !isBusyWithChapterAudio else {
            return "Wait for the current chapter audio task to finish."
        }
        return nil
    }

    var transcribeDisabledReason: String? {
        guard let chapter = selectedChapter else {
            return "Select a chapter before transcribing source audio."
        }
        guard !isBusyWithChapterAudio else {
            return "Wait for the current chapter audio task to finish."
        }
        guard chapter.hasSourceAudio else {
            return "Attach source audio before transcribing."
        }
        guard selectedRecognitionModelID != nil else {
            return "Install a speech recognition model to transcribe chapter audio."
        }
        return nil
    }

    var alignDisabledReason: String? {
        guard let chapter = selectedChapter else {
            return "Select a chapter before aligning source audio."
        }
        guard !isBusyWithChapterAudio else {
            return "Wait for the current chapter audio task to finish."
        }
        guard chapter.hasSourceAudio else {
            return "Attach source audio before aligning."
        }
        guard selectedAlignmentModelID != nil else {
            return "Install a forced-alignment model to align chapter audio."
        }
        guard alignmentTranscript(for: chapter) != nil else {
            return "Enter chapter text or transcribe the source audio before aligning."
        }
        return nil
    }

    var canStartRender: Bool {
        project != nil
            && !chapters.isEmpty
            && selectedRenderModelID != nil
            && !hasActiveRender
    }

    var hasActiveRender: Bool {
        renderJobs.contains { $0.state == .queued || $0.state == .running }
    }

    var projectHasRenderableContent: Bool {
        project != nil
            && chapters.contains(where: { !$0.script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
    }

    var overallRenderProgress: Double? {
        guard !renderJobs.isEmpty else { return nil }
        let terminalCount = renderJobs.filter { job in
            job.state == .completed || job.state == .cancelled || job.state == .failed
        }.count
        let runningProgress = renderJobs
            .filter { $0.state == .running }
            .reduce(0) { $0 + $1.progress }
        return min(1, (Double(terminalCount) + runningProgress) / Double(renderJobs.count))
    }

    var renderSummary: String {
        guard !renderJobs.isEmpty else { return "No render jobs queued" }

        let completedCount = renderJobs.filter { $0.state == .completed }.count
        let cancelledCount = renderJobs.filter { $0.state == .cancelled }.count
        let failedCount = renderJobs.filter { $0.state == .failed }.count
        let queuedCount = renderJobs.filter { $0.state == .queued }.count

        if let runningJob = renderJobs.first(where: { $0.state == .running }) {
            let chapterLabel = runningJob.title ?? runningJob.outputFileName
            return "Rendering \(chapterLabel) (\(completedCount + 1) of \(renderJobs.count))"
        }
        if queuedCount > 0 {
            return "Queued \(queuedCount) chapter render\(queuedCount == 1 ? "" : "s")"
        }
        if failedCount > 0 {
            return "Finished with \(failedCount) failed chapter render\(failedCount == 1 ? "" : "s")"
        }
        if cancelledCount > 0 {
            return "Stopped after \(completedCount) completed chapter render\(completedCount == 1 ? "" : "s")"
        }
        return "Completed \(completedCount) chapter render\(completedCount == 1 ? "" : "s")"
    }

    var canRenderSelectedChapter: Bool {
        renderChapterDisabledReason == nil
    }

    var renderSynthesisOptions: RenderSynthesisOptions {
        RenderSynthesisOptions(
            language: language,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens,
            voiceBehavior: voiceBehavior
        )
    }

    var renderChapterDisabledReason: String? {
        guard project != nil else {
            return "Open a project workspace to render a chapter."
        }
        guard !chapters.isEmpty else {
            return "Add a chapter before rendering."
        }
        guard let selectedChapter else {
            return "Select a chapter to render."
        }
        guard !selectedChapter.script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Add script text before rendering the selected chapter."
        }
        guard selectedRenderModelID != nil else {
            return "Choose a render model before rendering the selected chapter."
        }
        guard !hasActiveRender else {
            return "Stop the current render before starting another one."
        }
        return nil
    }

    var renderProjectDisabledReason: String? {
        guard project != nil else {
            return "Open a project workspace to render the project."
        }
        guard !chapters.isEmpty else {
            return "Add at least one chapter before rendering the project."
        }
        guard chapters.contains(where: { !$0.script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            return "Add script text to at least one chapter before rendering the project."
        }
        guard selectedRenderModelID != nil else {
            return "Choose a render model before rendering the project."
        }
        guard !hasActiveRender else {
            return "Stop the current render before starting another one."
        }
        return nil
    }

    func load(
        project: ProjectRecord,
        preferredModelID: ModelIdentifier?,
        preferredSynthesisOptions: RenderSynthesisOptions = RenderSynthesisOptions()
    ) async {
        self.project = project
        chapters = await services.projectStore.chapters(for: project.id)
        speakers = await services.projectStore.speakers(for: project.id).map(SpeakerEntry.init(record:))
        selectedChapterID = chapters.first?.id
        renderJobs = await services.renderQueue.jobs(matching: nil).filter { $0.projectID == project.id }
        applyRenderSynthesisOptions(preferredSynthesisOptions)
        await loadRenderModels(preferredModelID: preferredModelID)
        await loadChapterAudioModels()
        subscribeToRenderQueue(projectID: project.id)
    }

    func addChapter() async {
        guard let project else { return }
        let chapter = ChapterRecord(
            projectID: project.id,
            index: chapters.count,
            title: "Chapter \(chapters.count + 1)",
            script: ""
        )
        await services.projectStore.addChapter(chapter)
        chapters = await services.projectStore.chapters(for: project.id)
        selectedChapterID = chapter.id
    }

    func updateChapter(_ chapter: ChapterRecord) async {
        await services.projectStore.updateChapter(chapter)
        replaceChapter(chapter)
    }

    func attachSourceAudio(from url: URL) async {
        guard attachSourceAudioDisabledReason == nil,
              let chapter = selectedChapter else { return }

        isAttachingSourceAudio = true
        defer { isAttachingSourceAudio = false }

        do {
            let updated = try await services.attachChapterSourceAudio(chapter: chapter, from: url)
            replaceChapter(updated)
        } catch {
            errorMessage = "Could not attach chapter source audio: \(userFacingErrorMessage(for: error))"
        }
    }

    func transcribeSelectedChapter() async {
        guard transcribeDisabledReason == nil,
              let chapter = selectedChapter,
              let modelID = selectedRecognitionModelID else { return }

        isTranscribingSelectedChapter = true
        defer { isTranscribingSelectedChapter = false }

        do {
            let updated = try await services.transcribeChapterSourceAudio(
                chapter: chapter,
                modelID: modelID,
                languageHint: normalizedChapterAudioLanguageHint
            )
            replaceChapter(updated)
        } catch {
            errorMessage = "Could not transcribe chapter audio: \(userFacingErrorMessage(for: error))"
        }
    }

    func alignSelectedChapter() async {
        guard alignDisabledReason == nil,
              let chapter = selectedChapter,
              let modelID = selectedAlignmentModelID,
              let transcript = alignmentTranscript(for: chapter) else { return }

        isAligningSelectedChapter = true
        defer { isAligningSelectedChapter = false }

        do {
            let updated = try await services.alignChapterSourceAudio(
                chapter: chapter,
                transcript: transcript,
                modelID: modelID,
                languageHint: normalizedChapterAudioLanguageHint
            )
            replaceChapter(updated)
        } catch {
            errorMessage = "Could not align chapter audio: \(userFacingErrorMessage(for: error))"
        }
    }

    func addSpeaker(name: String, voiceModelID: ModelIdentifier?, language: String) async {
        guard let project else { return }
        let entry = SpeakerEntry(name: name, voiceModelID: voiceModelID, language: language)
        speakers.append(entry)
        await services.projectStore.addSpeaker(entry.record(projectID: project.id))
    }

    func removeSpeaker(_ id: UUID) async {
        guard let project else { return }
        speakers.removeAll { $0.id == id }
        await services.projectStore.removeSpeaker(id, from: project.id)
    }

    func moveChapter(from fromIndex: Int, to toIndex: Int) async {
        guard fromIndex != toIndex,
              chapters.indices.contains(fromIndex),
              chapters.indices.contains(toIndex) else { return }

        let chapter = chapters.remove(at: fromIndex)
        chapters.insert(chapter, at: toIndex)

        for i in chapters.indices {
            chapters[i].index = i
        }
        for chapter in chapters {
            await services.projectStore.updateChapter(chapter)
        }
    }

    func moveChapters(from source: IndexSet, to destination: Int) {
        chapters.move(fromOffsets: source, toOffset: destination)
        for (newIndex, _) in chapters.enumerated() {
            chapters[newIndex].index = newIndex
        }
        Task {
            for chapter in chapters {
                await services.projectStore.updateChapter(chapter)
            }
        }
    }

    func moveRenderJobs(from source: IndexSet, to destination: Int) {
        renderJobs.move(fromOffsets: source, toOffset: destination)
        guard let project else { return }
        let queuedIDs = renderJobs.filter { $0.state == .queued }.map(\.id)
        Task {
            await services.renderQueue.reorderQueuedJobs(for: project.id, orderedJobIDs: queuedIDs)
        }
    }

    func ingestScriptFile(_ url: URL, delimiter: String = "---") async {
        let content: String
        do {
            content = try readSecurityScopedText(at: url, encoding: .utf8)
        } catch {
            errorMessage = "Could not import \(url.lastPathComponent): \(userFacingErrorMessage(for: error))"
            return
        }
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let chunks: [String]
        if delimiter.isEmpty {
            chunks = [trimmed]
        } else {
            chunks = trimmed.components(separatedBy: delimiter)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }

        for chunk in chunks {
            let firstLine = chunk.components(separatedBy: .newlines).first ?? "Imported Chapter"
            let title = String(firstLine.prefix(60)).trimmingCharacters(in: .whitespaces)
            await addChapter()
            if var latest = chapters.last {
                latest.title = title
                latest.script = chunk
                await updateChapter(latest)
            }
        }
    }

    func queueRender(modelID: ModelIdentifier) async {
        guard let project else { return }
        selectedRenderModelID = modelID
        _ = await services.projectRenderService.enqueueProjectRender(
            project: project,
            modelID: modelID,
            synthesisOptions: renderSynthesisOptions
        )
    }

    func startRender() async {
        guard let modelID = selectedRenderModelID else { return }
        await queueRender(modelID: modelID)
    }

    func cancelRender(_ jobID: UUID) async {
        await services.projectRenderService.cancel(jobID)
    }

    func stopAllRenders() async {
        for job in renderJobs where job.state == .queued || job.state == .running {
            await cancelRender(job.id)
        }
    }

    func renderStatus(for chapterID: UUID) -> ChapterRenderStatus {
        let chapterJobs = renderJobs.filter { $0.chapterIDs.contains(chapterID) }
        if let running = chapterJobs.first(where: { $0.state == .running }) {
            return .rendering(progress: running.progress)
        }
        if chapterJobs.contains(where: { $0.state == .queued }) {
            return .queued
        }
        if chapterJobs.contains(where: { $0.state == .completed }) {
            return .complete
        }
        if chapterJobs.contains(where: { $0.state == .failed }) {
            return .failed
        }
        return .idle
    }

    func teardown() {
        renderUpdatesTask?.cancel()
        renderUpdatesTask = nil
    }

    func dismissError() {
        errorMessage = nil
    }

    func startSelectedChapterRender() async {
        guard let project,
              let chapter = selectedChapter,
              let modelID = selectedRenderModelID,
              !hasActiveRender else { return }

        _ = await services.projectRenderService.enqueueChapterRender(
            project: project,
            chapter: chapter,
            modelID: modelID,
            synthesisOptions: renderSynthesisOptions
        )
    }

    private func loadRenderModels(preferredModelID: ModelIdentifier?) async {
        let installedModels: [CatalogModel]
        do {
            installedModels = try await services.modelCatalog.installedModels()
        } catch {
            availableRenderModels = []
            selectedRenderModelID = nil
            errorMessage = "Could not load installed speech models: \(userFacingErrorMessage(for: error))"
            return
        }

        availableRenderModels = installedModels
            .filter { $0.descriptor.capabilities.contains(.speechSynthesis) }
            .map { RenderModelOption(id: $0.id, displayName: $0.descriptor.displayName) }

        if let preferredModelID,
           availableRenderModels.contains(where: { $0.id == preferredModelID }) {
            selectedRenderModelID = preferredModelID
            return
        }

        if let selectedRenderModelID,
           availableRenderModels.contains(where: { $0.id == selectedRenderModelID }) {
            return
        }

        selectedRenderModelID = availableRenderModels.first?.id
    }

    private func applyRenderSynthesisOptions(_ options: RenderSynthesisOptions) {
        language = options.language
        temperature = options.temperature ?? 0.7
        topP = options.topP ?? 0.9
        repetitionPenalty = options.repetitionPenalty ?? 1.0
        maxTokens = options.maxTokens ?? 8_192
        voiceBehavior = options.voiceBehavior
    }

    private func loadChapterAudioModels() async {
        let installedModels: [CatalogModel]
        do {
            installedModels = try await services.modelCatalog.installedModels()
        } catch {
            availableRecognitionModels = []
            availableAlignmentModels = []
            selectedRecognitionModelID = nil
            selectedAlignmentModelID = nil
            errorMessage = "Could not load speech analysis models: \(userFacingErrorMessage(for: error))"
            return
        }

        availableRecognitionModels = installedModels
            .filter { $0.descriptor.capabilities.contains(.speechRecognition) }
            .map { ProjectUtilityModelOption(id: $0.id, displayName: $0.descriptor.displayName) }

        availableAlignmentModels = installedModels
            .filter { $0.descriptor.capabilities.contains(.forcedAlignment) }
            .map { ProjectUtilityModelOption(id: $0.id, displayName: $0.descriptor.displayName) }

        if let selectedRecognitionModelID,
           availableRecognitionModels.contains(where: { $0.id == selectedRecognitionModelID }) {
            // Preserve current selection.
        } else {
            selectedRecognitionModelID = availableRecognitionModels.first?.id
        }

        if let selectedAlignmentModelID,
           availableAlignmentModels.contains(where: { $0.id == selectedAlignmentModelID }) {
            // Preserve current selection.
        } else {
            selectedAlignmentModelID = availableAlignmentModels.first?.id
        }
    }

    private func subscribeToRenderQueue(projectID: UUID) {
        renderUpdatesTask?.cancel()
        renderUpdatesTask = Task { [weak self] in
            guard let self else { return }
            let stream = await self.services.renderQueue.updates()
            for await jobs in stream {
                if Task.isCancelled {
                    return
                }
                self.renderJobs = jobs.filter { $0.projectID == projectID }
            }
        }
    }

    private var normalizedChapterAudioLanguageHint: String? {
        let trimmed = chapterAudioLanguageHint.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func alignmentTranscript(for chapter: ChapterRecord) -> String? {
        let trimmedScript = chapter.script.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedScript.isEmpty {
            return trimmedScript
        }
        guard let transcription = decodedTranscription(from: chapter)?.text
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !transcription.isEmpty
        else {
            return nil
        }
        return transcription
    }

    private func replaceChapter(_ chapter: ChapterRecord) {
        guard let idx = chapters.firstIndex(where: { $0.id == chapter.id }) else { return }
        chapters[idx] = chapter
    }

    private func decodedTranscription(from chapter: ChapterRecord) -> RichTranscriptionResult? {
        guard let transcriptionJSON = chapter.transcriptionJSON,
              let data = transcriptionJSON.data(using: .utf8)
        else {
            return nil
        }

        if let rich = try? JSONDecoder().decode(RichTranscriptionResult.self, from: data) {
            return rich
        }

        if let flat = try? JSONDecoder().decode(TranscriptionResult.self, from: data) {
            return RichTranscriptionResult(
                text: flat.text,
                segments: flat.segments.map { TranscriptionSegment(text: $0) },
                backendMetadata: BackendMetadata(
                    modelId: chapter.transcriptionModelID ?? "unknown",
                    backendKind: .mlx
                )
            )
        }

        return nil
    }

    private func decodedAlignment(from chapter: ChapterRecord) -> ForcedAlignmentResult? {
        guard let alignmentJSON = chapter.alignmentJSON,
              let data = alignmentJSON.data(using: .utf8)
        else {
            return nil
        }

        return try? JSONDecoder().decode(ForcedAlignmentResult.self, from: data)
    }

    private func userFacingErrorMessage(for error: any Error) -> String {
        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        return PathRedaction.redactMessage(message)
    }

    private func readSecurityScopedText(
        at url: URL,
        encoding: String.Encoding = .utf8
    ) throws -> String {
        let needsScopedAccess = url.startAccessingSecurityScopedResource()
        defer {
            if needsScopedAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        return try String(contentsOf: url, encoding: encoding)
    }
}
