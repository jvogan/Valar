import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ValarPersistence

struct SegmentEditorView: View {
    @Bindable var state: ProjectWorkspaceState

    @State private var editTitle = ""
    @State private var editScript = ""
    @State private var editSpeaker = ""
    @State private var editingChapterID: UUID?
    @State private var pendingSaveTask: Task<Void, Never>?
    @State private var hasPendingEdits = false
    @State private var isSyncingFromState = false

    var body: some View {
        Group {
            if let chapter = state.selectedChapter {
                editorContent(chapter)
            } else {
                EmptyStateView(
                    icon: "text.alignleft",
                    title: "No chapter selected",
                    message: "Select or create a chapter to begin editing."
                )
            }
        }
        .onChange(of: state.selectedChapterID) {
            flushPendingEdits()
            syncFromChapter()
        }
        .onChange(of: editTitle) {
            scheduleAutoSave()
        }
        .onChange(of: editScript) {
            scheduleAutoSave()
        }
        .onChange(of: editSpeaker) {
            scheduleAutoSave()
        }
        .onAppear {
            syncFromChapter()
        }
        .onDisappear {
            flushPendingEdits()
        }
    }

    @ViewBuilder
    private func editorContent(_ chapter: ChapterRecord) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            // Chapter title bar
            HStack(spacing: 12) {
                TextField("Chapter title", text: $editTitle)
                    .textFieldStyle(.plain)
                    .font(.title3.bold())
                    .accessibilityLabel("Chapter title")

                Spacer()

                Picker("Speaker", selection: $editSpeaker) {
                    Text("No speaker").tag("")
                    ForEach(state.speakers) { speaker in
                        Text(speaker.name).tag(speaker.name)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 160)
                .accessibilityLabel("Speaker")

                if hasPendingEdits {
                    Circle()
                        .fill(.orange)
                        .frame(width: 6, height: 6)
                        .accessibilityLabel("Unsaved changes")
                }

                Button("Save") {
                    commitEdits()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .accessibilityLabel("Save chapter")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.surfaceRecessed)

            Divider()

            chapterAudioTools(chapter)

            Divider()

            // Script editor
            TextEditor(text: $editScript)
                .font(.body.monospaced())
                .scrollContentBackground(.hidden)
                .padding(16)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .accessibilityLabel("Chapter script")

            // Footer
            HStack {
                Text("\(editScript.wordCount) words")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Spacer()
                if let dur = chapter.estimatedDurationSeconds {
                    Text("~\(Int(dur))s")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(.surfaceRecessed)
        }
    }

    @ViewBuilder
    private func chapterAudioTools(_ chapter: ChapterRecord) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Label {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(chapter.sourceAudioAssetName ?? "No source audio attached")
                            .font(.subheadline.weight(.medium))
                            .lineLimit(1)
                        if let metadata = sourceAudioMetadata(for: chapter) {
                            Text(metadata)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                } icon: {
                    Image(systemName: chapter.hasSourceAudio ? "waveform" : "waveform.badge.plus")
                        .foregroundStyle(chapter.hasSourceAudio ? Color.accentColor : Color.secondary)
                }

                Spacer()

                Button(chapter.hasSourceAudio ? "Replace Audio…" : "Attach Audio…") {
                    guard let sourceURL = chooseSourceAudioURL() else { return }
                    Task { await state.attachSourceAudio(from: sourceURL) }
                }
                .buttonStyle(.bordered)
                .disabled(state.attachSourceAudioDisabledReason != nil)
                .help(state.attachSourceAudioDisabledReason ?? "Attach source audio to this chapter.")
            }

            HStack(spacing: 10) {
                Text("Language Hint")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                TextField("Auto", text: $state.chapterAudioLanguageHint)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)

                Text("Optional language hint for transcription and alignment.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)

                Spacer()
            }

            HStack(spacing: 8) {
                Button("Transcribe Audio") {
                    commitEdits()
                    Task { await state.transcribeSelectedChapter() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.transcribeDisabledReason != nil)
                .help(state.transcribeDisabledReason ?? "Transcribe the attached source audio.")

                Button("Align to Transcript") {
                    commitEdits()
                    Task { await state.alignSelectedChapter() }
                }
                .buttonStyle(.bordered)
                .disabled(state.alignDisabledReason != nil)
                .help(state.alignDisabledReason ?? "Align the chapter transcript to the attached source audio.")

                if state.isBusyWithChapterAudio {
                    ProgressView()
                        .controlSize(.small)
                }

                Spacer()
            }

            if let transcription = state.selectedChapterTranscription {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Transcription")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(transcription.text)
                        .font(.caption)
                        .foregroundStyle(.primary)
                        .lineLimit(3)
                    if let modelLabel = state.selectedRecognitionModelName ?? chapter.transcriptionModelID {
                        Text("Model: \(modelLabel)")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }

            if let alignment = state.selectedChapterAlignment {
                HStack(spacing: 6) {
                    Text("Alignment")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text("\(alignment.tokens.count) token\(alignment.tokens.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.primary)
                    if let modelLabel = state.selectedAlignmentModelName ?? chapter.alignmentModelID {
                        Text("via \(modelLabel)")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.surfaceRecessed)
    }

    private func syncFromChapter() {
        pendingSaveTask?.cancel()
        pendingSaveTask = nil

        isSyncingFromState = true
        defer { isSyncingFromState = false }

        guard let chapter = state.selectedChapter else {
            editingChapterID = nil
            editTitle = ""
            editScript = ""
            editSpeaker = ""
            hasPendingEdits = false
            return
        }

        editingChapterID = chapter.id
        editTitle = chapter.title
        editScript = chapter.script
        editSpeaker = chapter.speakerLabel ?? ""
        hasPendingEdits = false
    }

    private func commitEdits() {
        pendingSaveTask?.cancel()
        pendingSaveTask = nil
        guard let updated = currentUpdatedChapter() else {
            hasPendingEdits = false
            return
        }
        persist(updated)
    }

    private func scheduleAutoSave() {
        guard !isSyncingFromState else { return }
        guard editingChapter != nil else { return }

        pendingSaveTask?.cancel()

        guard isDirty else {
            hasPendingEdits = false
            pendingSaveTask = nil
            return
        }

        hasPendingEdits = true
        pendingSaveTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            pendingSaveTask = nil
            await persistCurrentEdits()
        }
    }

    private func flushPendingEdits() {
        pendingSaveTask?.cancel()
        pendingSaveTask = nil

        guard let updated = currentUpdatedChapter() else {
            hasPendingEdits = false
            return
        }

        persist(updated)
    }

    private func persist(_ updated: ChapterRecord) {
        applyWorkspaceUpdate(updated)
        Task { @MainActor in
            await state.updateChapter(updated)
            if editingChapterID == updated.id, isDirtyCompared(to: updated) {
                scheduleAutoSave()
            } else if editingChapterID == updated.id {
                hasPendingEdits = false
            }
        }
    }

    private func persistCurrentEdits() async {
        guard let chapter = editingChapter else {
            hasPendingEdits = false
            return
        }

        guard isDirty else {
            hasPendingEdits = false
            return
        }

        let updated = updatedChapter(from: chapter)
        applyWorkspaceUpdate(updated)
        await state.updateChapter(updated)

        if editingChapterID == updated.id, isDirtyCompared(to: updated) {
            scheduleAutoSave()
        } else {
            hasPendingEdits = false
        }
    }

    private var editingChapter: ChapterRecord? {
        guard let editingChapterID else { return nil }
        return state.chapters.first { $0.id == editingChapterID }
    }

    private var isDirty: Bool {
        guard let chapter = editingChapter else { return false }
        return isDirtyCompared(to: chapter)
    }

    private func isDirtyCompared(to chapter: ChapterRecord) -> Bool {
        chapter.title != editTitle
            || chapter.script != editScript
            || (chapter.speakerLabel ?? "") != editSpeaker
    }

    private func updatedChapter(from chapter: ChapterRecord) -> ChapterRecord {
        var updated = chapter
        updated.title = editTitle
        updated.script = editScript
        updated.speakerLabel = editSpeaker.isEmpty ? nil : editSpeaker
        return updated
    }

    private func currentUpdatedChapter() -> ChapterRecord? {
        guard let chapter = editingChapter, isDirtyCompared(to: chapter) else { return nil }
        return updatedChapter(from: chapter)
    }

    private func applyWorkspaceUpdate(_ updated: ChapterRecord) {
        guard let index = state.chapters.firstIndex(where: { $0.id == updated.id }) else { return }
        state.chapters[index] = updated
    }

    private func chooseSourceAudioURL() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio]
        panel.prompt = "Attach Audio"
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func sourceAudioMetadata(for chapter: ChapterRecord) -> String? {
        var components: [String] = []
        if let sampleRate = chapter.sourceAudioSampleRate {
            components.append("\(Int(sampleRate.rounded())) Hz")
        }
        if let duration = chapter.sourceAudioDurationSeconds {
            components.append(Self.formattedDuration(duration))
        }
        return components.isEmpty ? nil : components.joined(separator: " • ")
    }

    private static func formattedDuration(_ duration: Double) -> String {
        let totalSeconds = max(Int(duration.rounded()), 0)
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0 {
            return String(format: "%d:%02d", minutes, seconds)
        }
        return "\(seconds)s"
    }
}
