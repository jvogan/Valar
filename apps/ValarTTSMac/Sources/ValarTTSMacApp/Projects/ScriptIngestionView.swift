import SwiftUI

struct ScriptIngestionView: View {
    @Bindable var workspaceState: ProjectWorkspaceState
    @Environment(\.dismiss) private var dismiss

    @State private var rawScript = ""
    @State private var chapterDelimiter = "---"
    @State private var isImporting = false
    @State private var importedChapterCount = 0
    @State private var totalChapterCount = 0
    @State private var importError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Import Script")
                .font(.title2.bold())

            Text("Paste your script below. Chapters will be split on the delimiter.")
                .font(.callout)
                .foregroundStyle(.secondary)

            HStack {
                Text("Delimiter")
                    .font(.callout.weight(.medium))
                TextField("---", text: $chapterDelimiter)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 120)
            }
            .disabled(isImporting)

            TextEditor(text: $rawScript)
                .font(.body.monospaced())
                .frame(minHeight: 260)
                .padding(8)
                .background(.surfaceRecessed)
                .clipShape(RoundedRectangle(cornerRadius: ValarRadius.field, style: .continuous))
                .disabled(isImporting)

            if isImporting {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: Double(importedChapterCount), total: Double(totalChapterCount))
                        .progressViewStyle(.linear)
                    Text("Imported \(importedChapterCount) of \(totalChapterCount) chapters")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            HStack {
                Text(isImporting ? "Importing chapters..." : "\(parsedChapterCount) chapters detected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .buttonStyle(.bordered)
                    .disabled(isImporting)
                Button(isImporting ? "Importing..." : "Parse & Import") {
                    importChapters()
                }
                .buttonStyle(.borderedProminent)
                .disabled(parsedChapterCount == 0 || isImporting)
            }
        }
        .alert("Import Error", isPresented: Binding(
            get: { importError != nil },
            set: { if !$0 { importError = nil } }
        )) {
            Button("OK") { importError = nil }
        } message: {
            Text(importError ?? "")
        }
        .padding(24)
        .frame(minWidth: 500, minHeight: 420)
    }

    private var parsedChapterCount: Int {
        parsedChunks.count
    }

    private var parsedChunks: [String] {
        let trimmedScript = rawScript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedScript.isEmpty else { return [] }

        let delimiter = chapterDelimiter.trimmingCharacters(in: .whitespaces)
        if delimiter.isEmpty {
            return [trimmedScript]
        }

        return trimmedScript.components(separatedBy: delimiter)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func importChapters() {
        let chunks = parsedChunks
        guard !chunks.isEmpty, !isImporting else { return }

        isImporting = true
        importedChapterCount = 0
        totalChapterCount = chunks.count

        Task { @MainActor in
            defer { isImporting = false }

            var failedChapters: [Int] = []
            for (index, chunk) in chunks.enumerated() {
                let firstLine = chunk.components(separatedBy: .newlines).first ?? "Chapter \(index + 1)"
                let title = firstLine.prefix(60).trimmingCharacters(in: .whitespaces)
                await workspaceState.addChapter()
                if var latest = workspaceState.selectedChapter ?? workspaceState.chapters.last {
                    latest.title = String(title)
                    latest.script = chunk
                    await workspaceState.updateChapter(latest)
                } else {
                    failedChapters.append(index + 1)
                }
                importedChapterCount = index + 1
            }

            if failedChapters.isEmpty {
                dismiss()
            } else {
                importError = "Failed to save chapter(s): \(failedChapters.map(String.init).joined(separator: ", "))"
            }
        }
    }
}
