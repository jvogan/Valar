import CryptoKit
import Foundation
import ValarAudio
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence

/// Snapshot of progress during a concurrent export operation.
struct ExportProgress: Sendable {
    let completedChapters: Int
    let totalChapters: Int
    let currentChapterTitle: String?

    /// Fraction complete in [0, 1].
    var fraction: Double {
        guard totalChapters > 0 else { return 0 }
        return min(1, Double(completedChapters) / Double(totalChapters))
    }

    /// Human-readable status suitable for display in the export UI.
    var statusLabel: String {
        guard totalChapters > 0 else { return "Preparing export…" }
        guard completedChapters < totalChapters else { return "Finalizing export…" }
        let ordinal = completedChapters + 1
        if let title = currentChapterTitle, !title.isEmpty {
            return "Rendering \(ordinal) of \(totalChapters) — \(title)"
        }
        return "Rendering chapter \(ordinal) of \(totalChapters)"
    }
}

enum ProjectExportFormat: String, CaseIterable, Sendable, Hashable {
    case wav = "WAV"
    case m4a = "M4A"

    var fileExtension: String {
        switch self {
        case .wav:
            return "wav"
        case .m4a:
            return "m4a"
        }
    }
}

enum ProjectExportMode: String, CaseIterable, Sendable, Hashable {
    case concatenated = "Single File"
    case chapters = "Per Chapter"
}

struct ProjectAudioExportResult: Sendable {
    let files: [URL]
    let exportedChapterCount: Int
}

protocol ProjectAudioExporting: Sendable {
    func exportProjectAudio(
        projectID: UUID,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions,
        format: ProjectExportFormat,
        mode: ProjectExportMode,
        destinationURL: URL,
        onProgress: @escaping @Sendable (ExportProgress) -> Void
    ) async throws -> ProjectAudioExportResult
}

actor ProjectAudioExportCoordinator: ProjectAudioExporting {
    typealias ChapterSynthesizer = @Sendable (ModelIdentifier, RenderSynthesisOptions, String) async throws -> AudioChunk

    private let projectStore: any ProjectStoring
    private let audioPipeline: AudioPipeline
    private let synthesizeChapter: ChapterSynthesizer

    init(
        projectStore: any ProjectStoring,
        audioPipeline: AudioPipeline,
        synthesizeChapter: @escaping ChapterSynthesizer
    ) {
        self.projectStore = projectStore
        self.audioPipeline = audioPipeline
        self.synthesizeChapter = synthesizeChapter
    }

    func exportProjectAudio(
        projectID: UUID,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions,
        format: ProjectExportFormat,
        mode: ProjectExportMode,
        destinationURL: URL,
        onProgress: @escaping @Sendable (ExportProgress) -> Void
    ) async throws -> ProjectAudioExportResult {
        guard let project = await projectStore.allProjects().first(where: { $0.id == projectID }) else {
            throw ProjectAudioExportError.projectNotFound(projectID)
        }

        let allChapters = await projectStore.chapters(for: projectID).sorted { $0.index < $1.index }
        let exportChapters = allChapters.filter { !$0.script.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !exportChapters.isEmpty else {
            throw ProjectAudioExportError.noRenderableChapters(project.title)
        }

        // Render one chapter at a time to bound peak memory to ~2 chapter buffers.
        // Progress is reported before each chapter so the UI shows which chapter is active.
        let totalChapters = exportChapters.count
        var renderedChapters: [RenderedChapter] = []
        renderedChapters.reserveCapacity(totalChapters)
        for (index, chapter) in exportChapters.enumerated() {
            onProgress(ExportProgress(
                completedChapters: index,
                totalChapters: totalChapters,
                currentChapterTitle: chapter.title
            ))
            try Task.checkCancellation()
            let rendered = try await renderedChapter(
                from: chapter,
                modelID: modelID,
                synthesisOptions: synthesisOptions
            )
            renderedChapters.append(rendered)
        }
        onProgress(ExportProgress(
            completedChapters: totalChapters,
            totalChapters: totalChapters,
            currentChapterTitle: nil
        ))

        guard renderedChapters.first?.buffer != nil else {
            throw ProjectAudioExportError.noRenderableChapters(project.title)
        }

        let exportedFiles: [URL]
        switch mode {
        case .concatenated:
            let markers = buildChapterMarkers(from: renderedChapters)
            let buffer = try await concatenatedBuffer(from: renderedChapters.map(\.buffer))
            let fileURL = destinationURL.appendingPathExtensionIfNeeded(format.fileExtension)
            let descriptor = AudioFormatDescriptor(
                sampleRate: buffer.format.sampleRate,
                channelCount: buffer.format.channelCount,
                sampleFormat: .float32,
                interleaved: false,
                container: format.fileExtension
            )
            _ = try await audioPipeline.export(
                buffer,
                as: descriptor,
                to: fileURL,
                chapterMarkers: format == .m4a ? markers : []
            )
            try await recordExport(for: projectID, fileURL: fileURL)
            exportedFiles = [fileURL]
        case .chapters:
            let directoryURL = destinationURL
            var urls: [URL] = []

            for (index, chapter) in renderedChapters.enumerated() {
                let fileURL = directoryURL.appendingPathComponent(
                    exportedFileName(for: chapter.chapter, ordinal: index + 1, format: format),
                    isDirectory: false
                )
                let descriptor = AudioFormatDescriptor(
                    sampleRate: chapter.buffer.format.sampleRate,
                    channelCount: chapter.buffer.format.channelCount,
                    sampleFormat: .float32,
                    interleaved: false,
                    container: format.fileExtension
                )
                let markers = format == .m4a ? [AudioChapterMarker(title: chapter.chapter.title, startTime: 0, duration: chapter.buffer.duration)] : []
                _ = try await audioPipeline.export(
                    chapter.buffer,
                    as: descriptor,
                    to: fileURL,
                    chapterMarkers: markers
                )
                try await recordExport(for: projectID, fileURL: fileURL)
                urls.append(fileURL)
            }

            exportedFiles = urls
        }
        return ProjectAudioExportResult(files: exportedFiles, exportedChapterCount: renderedChapters.count)
    }

    private func renderedChapter(
        from chapter: ChapterRecord,
        modelID: ModelIdentifier,
        synthesisOptions: RenderSynthesisOptions
    ) async throws -> RenderedChapter {
        let chunk = try await synthesizeChapter(modelID, synthesisOptions, chapter.script)
        let rawBuffer = audioBuffer(from: chunk)
        let normalizedBuffer = await audioPipeline.normalize(rawBuffer)

        return RenderedChapter(chapter: chapter, buffer: normalizedBuffer)
    }

    private func audioBuffer(from chunk: AudioChunk) -> AudioPCMBuffer {
        AudioPCMBuffer(mono: chunk.samples, sampleRate: chunk.sampleRate)
    }

    private func concatenatedBuffer(from buffers: [AudioPCMBuffer]) async throws -> AudioPCMBuffer {
        guard let combined = await audioPipeline.concatenate(buffers) else {
            throw ProjectAudioExportError.incompatibleBuffers
        }
        return combined
    }

    private func buildChapterMarkers(from chapters: [RenderedChapter]) -> [AudioChapterMarker] {
        var offset: TimeInterval = 0

        return chapters.map { chapter in
            defer { offset += chapter.buffer.duration }
            return AudioChapterMarker(
                title: chapter.chapter.title,
                startTime: offset,
                duration: chapter.buffer.duration
            )
        }
    }

    private func exportedFileName(
        for chapter: ChapterRecord,
        ordinal: Int,
        format: ProjectExportFormat
    ) -> String {
        let prefix = String(format: "%02d", ordinal)
        return "\(prefix)-\(sanitizedFileStem(chapter.title)).\(format.fileExtension)"
    }

    private func sanitizedFileStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let collapsed = value.unicodeScalars.map { allowed.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let stem = collapsed
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
            .lowercased()

        return stem.isEmpty ? "chapter" : stem
    }

    private func recordExport(for projectID: UUID, fileURL: URL) async throws {
        let checksum = try streamingSHA256(at: fileURL)
        await projectStore.addExport(
            ExportRecord(
                projectID: projectID,
                fileName: fileURL.lastPathComponent,
                checksum: checksum
            )
        )
    }

    private func streamingSHA256(at url: URL) throws -> String {
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { try? fileHandle.close() }
        var hasher = SHA256()
        let chunkSize = 64 * 1024
        while let chunk = try fileHandle.read(upToCount: chunkSize), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private struct RenderedChapter: Sendable {
    let chapter: ChapterRecord
    let buffer: AudioPCMBuffer
}

private enum ProjectAudioExportError: LocalizedError {
    case incompatibleBuffers
    case noRenderableChapters(String)
    case projectNotFound(UUID)

    var errorDescription: String? {
        switch self {
        case .incompatibleBuffers:
            return "The rendered chapter buffers could not be concatenated."
        case let .noRenderableChapters(title):
            return "Project '\(title)' has no chapters with script content to export."
        case let .projectNotFound(projectID):
            return "Project \(projectID.uuidString) was not found."
        }
    }
}

private extension URL {
    func appendingPathExtensionIfNeeded(_ pathExtension: String) -> URL {
        guard self.pathExtension.caseInsensitiveCompare(pathExtension) != .orderedSame else {
            return self
        }
        return appendingPathExtension(pathExtension)
    }
}

private extension Array {
    func asyncMap<T: Sendable>(_ transform: @escaping @Sendable (Element) async throws -> T) async throws -> [T] {
        var results: [T] = []
        results.reserveCapacity(count)

        for element in self {
            try Task.checkCancellation()
            results.append(try await transform(element))
        }

        return results
    }
}
