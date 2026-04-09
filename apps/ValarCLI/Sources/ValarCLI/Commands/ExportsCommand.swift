import ArgumentParser
import CryptoKit
import Foundation
import ValarCore
import ValarModelKit
import ValarPersistence

struct ExportsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "exports",
        abstract: "List and create chapter exports for the active project session.",
        subcommands: [List.self, Create.self]
    )

    mutating func run() throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension ExportsCommand {
    struct List: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "Show completed exports for the active project session."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let exports = await context.projectStore.exports(for: activeSession.projectID)
                if jsonRequested {
                    let orderedExports = exports.sorted(by: { $0.createdAt > $1.createdAt })
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("exports list"),
                        data: ExportListPayloadDTO(
                            message: orderedExports.isEmpty
                                ? "No exports found for '\(activeSession.title)'."
                                : "Loaded \(orderedExports.count) export(s) for '\(activeSession.title)'.",
                            projectTitle: activeSession.title,
                            exports: orderedExports.map { ExportDTO(from: $0) }
                        )
                    )
                    return
                }

                guard exports.isEmpty == false else {
                    print("No exports found for '\(activeSession.title)'.")
                    return
                }

                print("id\tcreatedAt\tfileName\tchecksum")
                for export in exports.sorted(by: { $0.createdAt > $1.createdAt }) {
                    print(
                        [
                            export.id.uuidString,
                            ProjectsCommand.iso8601String(from: export.createdAt),
                            Self.sanitize(export.fileName),
                            Self.sanitize(export.checksum ?? "-"),
                        ].joined(separator: "\t")
                    )
                }
            }
        }

        private static func sanitize(_ value: String) -> String {
            value
                .replacingOccurrences(of: "\t", with: " ")
                .replacingOccurrences(of: "\n", with: " ")
        }
    }

    struct Create: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "create",
            abstract: "Create or refresh an export for a chapter in the active project session."
        )

        @Option(name: .long, help: "Chapter UUID to export.")
        var chapter: String

        @Option(name: .long, help: "Override the speech model used when a new render is required.")
        var model: String?

        mutating func run() throws {
            let rawChapterID = chapter
            let explicitModelID = model
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                guard let chapterID = UUID(uuidString: rawChapterID.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw ValidationError("Chapter id must be a UUID.")
                }

                let context = try ProjectsCommand.ProjectCommandContext()
                var activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let chapters = await context.projectStore.chapters(for: activeSession.projectID)
                guard let chapter = chapters.first(where: { $0.id == chapterID }) else {
                    throw ValidationError("No chapter exists with id \(rawChapterID) in the active project session.")
                }

                guard let bundleLocation = await context.projectStore.bundleLocation(for: activeSession.projectID) else {
                    throw CLICommandError(message: "The active project bundle is unavailable for exports.")
                }

                try context.fileManager.createDirectory(
                    at: bundleLocation.exportsDirectory,
                    withIntermediateDirectories: true
                )

                let jobs = await RendersCommand.projectJobs(
                    in: context.runtime.renderQueue,
                    projectID: activeSession.projectID
                )
                let latestCompletedRender = jobs
                    .filter { $0.chapterID == chapter.id && $0.state == .completed }
                    .sorted(by: { $0.createdAt > $1.createdAt })
                    .first

                let outputFileName = latestCompletedRender?.outputFileName ?? RendersCommand.outputFileName(for: chapter)
                let outputURL = try RendersCommand.validatedOutputURL(
                    for: outputFileName,
                    in: bundleLocation.exportsDirectory
                )
                var usedModelID = latestCompletedRender?.modelID
                let synthesisOptions: RenderSynthesisOptions
                if let latestCompletedRender {
                    synthesisOptions = latestCompletedRender.synthesisOptions
                } else {
                    synthesisOptions = await RendersCommand.resolveSynthesisOptions(for: chapter, in: context)
                }

                if context.fileManager.fileExists(atPath: outputURL.path) == false {
                    let modelID: ModelIdentifier
                    if let latestCompletedRender {
                        modelID = latestCompletedRender.modelID
                    } else {
                        modelID = try await RendersCommand.resolveModelID(
                            explicitModelID: explicitModelID,
                            for: chapter,
                            in: context
                        )
                    }
                    usedModelID = modelID

                    let renderer = try CLITextToSpeechRenderer(
                        paths: context.paths,
                        fileManager: context.fileManager
                    )
                    let chunk = try await renderer.synthesize(
                        text: chapter.script,
                        modelID: modelID,
                        synthesisOptions: synthesisOptions
                    )
                    let wavData = try await renderer.exportWAV(from: chunk)

                    if context.fileManager.fileExists(atPath: outputURL.path) {
                        try context.fileManager.removeItem(at: outputURL)
                    }
                    try wavData.write(to: outputURL, options: .atomic)
                }

                let checksum = try Self.streamingSHA256(at: outputURL)
                let existingExports = await context.projectStore.exports(for: activeSession.projectID)
                let existingExport = existingExports.first(where: { $0.fileName == outputFileName })

                await context.projectStore.addExport(
                    ExportRecord(
                        id: existingExport?.id ?? UUID(),
                        projectID: activeSession.projectID,
                        fileName: outputFileName,
                        createdAt: .now,
                        checksum: checksum
                    )
                )
                activeSession = try await ProjectsCommand.persistActiveSession(activeSession, in: context)

                let message = "Created export \(outputFileName) for chapter '\(chapter.title)'."
                if jsonRequested {
                    let exportRecord = ExportRecord(
                        id: existingExport?.id ?? UUID(),
                        projectID: activeSession.projectID,
                        fileName: outputFileName,
                        createdAt: .now,
                        checksum: checksum
                    )
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("exports create"),
                        data: ExportCreatePayloadDTO(
                            message: message,
                            chapterID: chapter.id.uuidString,
                            modelID: usedModelID?.rawValue,
                            outputPath: outputURL.path,
                            export: ExportDTO(from: exportRecord)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }

        private static func streamingSHA256(at url: URL) throws -> String {
            let fileHandle = try FileHandle(forReadingFrom: url)
            defer { try? fileHandle.close() }

            var hasher = SHA256()
            let chunkSize = 64 * 1024

            while let chunk = try fileHandle.read(upToCount: chunkSize), chunk.isEmpty == false {
                hasher.update(data: chunk)
            }

            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }
    }
}
