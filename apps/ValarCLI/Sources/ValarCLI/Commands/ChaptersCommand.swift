@preconcurrency import ArgumentParser
import AVFoundation
import Foundation
import ValarCore
import ValarPersistence

struct ChaptersCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chapters",
        abstract: "List, add, update, and remove chapters in the active CLI project session.",
        subcommands: [List.self, Add.self, Update.self, AttachAudio.self, Remove.self]
    )

    mutating func run() throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension ChaptersCommand {
    struct List: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Show all chapters in the active CLI project session."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let chapters = await context.projectStore.chapters(for: activeSession.projectID)
                if jsonRequested {
                    let orderedChapters = chapters.sorted(by: { $0.index < $1.index })
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("chapters list"),
                        data: ChapterListPayloadDTO(
                            message: orderedChapters.isEmpty
                                ? "No chapters in '\(activeSession.title)'."
                                : "Loaded \(orderedChapters.count) chapter(s) from '\(activeSession.title)'.",
                            projectTitle: activeSession.title,
                            chapters: orderedChapters.map { ChapterDTO(from: $0) }
                        )
                    )
                    return
                }

                if chapters.isEmpty {
                    print("No chapters in '\(activeSession.title)'.")
                    return
                }

                for chapter in chapters.sorted(by: { $0.index < $1.index }) {
                    print(renderChapterSummaryLine(chapter))
                }
            }
        }
    }

    struct Add: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Add a chapter to the active CLI project session and persist it to the open bundle."
        )

        @Option(name: .long, help: "Chapter title.")
        var title: String

        @Option(name: .long, help: "Chapter text/script.")
        var text: String

        mutating func run() throws {
            let title = self.title
            let text = self.text
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let normalizedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
                guard normalizedTitle.isEmpty == false else {
                    throw ValidationError("Chapter title must not be empty.")
                }

                let session = try await ProjectsCommand.requireRuntimeSession(
                    for: activeSession.projectID,
                    in: context
                )
                let chapters = await session.chapters()
                let nextIndex = (chapters.map(\.index).max() ?? -1) + 1
                let chapter = ChapterRecord(
                    projectID: activeSession.projectID,
                    index: nextIndex,
                    title: normalizedTitle,
                    script: text
                )

                await session.addChapter(chapter)
                _ = try await ProjectsCommand.persistActiveSession(activeSession, in: context)

                let message = "Added chapter '\(chapter.title)' (\(chapter.id.uuidString))."
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("chapters add"),
                        data: ChapterMutationPayloadDTO(
                            message: message,
                            chapter: ChapterDTO(from: chapter)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Update: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Update a chapter in the active CLI project session and persist it to the open bundle."
        )

        @Argument(help: "Chapter id.")
        var id: String

        @Option(name: .long, help: "Updated chapter title.")
        var title: String?

        @Option(name: .long, help: "Updated chapter text/script.")
        var text: String?

        func validate() throws {
            guard title != nil || text != nil else {
                throw ValidationError("Provide at least one field to update with `--title` or `--text`.")
            }

            if let title, title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ValidationError("Chapter title must not be empty.")
            }
        }

        mutating func run() throws {
            let id = self.id
            let title = self.title?.trimmingCharacters(in: .whitespacesAndNewlines)
            let text = self.text
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let chapterID = try ChaptersCommand.parseChapterID(id)
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let session = try await ProjectsCommand.requireRuntimeSession(
                    for: activeSession.projectID,
                    in: context
                )
                var chapters = await session.chapters()
                guard let chapterIndex = chapters.firstIndex(where: { $0.id == chapterID }) else {
                    throw ValidationError("No chapter exists with id \(chapterID.uuidString).")
                }

                if let title {
                    chapters[chapterIndex].title = title
                }
                if let text {
                    chapters[chapterIndex].script = text
                }

                await session.updateChapter(chapters[chapterIndex])
                _ = try await ProjectsCommand.persistActiveSession(activeSession, in: context)

                let message = "Updated chapter '\(chapters[chapterIndex].title)' (\(chapterID.uuidString))."
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("chapters update"),
                        data: ChapterMutationPayloadDTO(
                            message: message,
                            chapter: ChapterDTO(from: chapters[chapterIndex])
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct AttachAudio: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "attach-audio",
            abstract: "Attach a source audio file to a chapter in the active CLI project session."
        )

        @Argument(help: "Chapter id.")
        var id: String

        @Option(name: .long, help: "Path to the source audio file.")
        var audio: String

        mutating func run() throws {
            let id = self.id
            let audio = self.audio
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let chapterID = try ChaptersCommand.parseChapterID(id)
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let chapters = await context.projectStore.chapters(for: activeSession.projectID)
                guard let existingChapter = chapters.first(where: { $0.id == chapterID }) else {
                    throw ValidationError("No chapter exists with id \(chapterID.uuidString).")
                }

                let sourceURL = try ChaptersCommand.resolvedAudioURL(
                    from: audio,
                    fileManager: context.fileManager
                )
                let stagedAudio = try ChaptersCommand.stageAudioAsset(
                    for: chapterID,
                    sourceURL: sourceURL,
                    previousAssetName: existingChapter.sourceAudioAssetName,
                    activeSession: activeSession,
                    fileManager: context.fileManager
                )

                await context.projectStore.attachAudio(
                    to: chapterID,
                    in: activeSession.projectID,
                    assetName: stagedAudio.assetName,
                    sampleRate: stagedAudio.sampleRate,
                    durationSeconds: stagedAudio.durationSeconds
                )

                guard let updatedChapter = await context.projectStore
                    .chapters(for: activeSession.projectID)
                    .first(where: { $0.id == chapterID }) else {
                    throw ValidationError("No chapter exists with id \(chapterID.uuidString).")
                }

                let message = "Attached source audio '\(sourceURL.lastPathComponent)' to chapter '\(updatedChapter.title)' (\(chapterID.uuidString))."
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("chapters attach-audio"),
                        data: ChapterMutationPayloadDTO(
                            message: message,
                            chapter: ChapterDTO(from: updatedChapter)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Remove: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Remove a chapter from the active CLI project session and persist it to the open bundle."
        )

        @Argument(help: "Chapter id.")
        var id: String

        mutating func run() throws {
            let id = self.id
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let chapterID = try ChaptersCommand.parseChapterID(id)
                let context = try ProjectsCommand.ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let session = try await ProjectsCommand.requireRuntimeSession(
                    for: activeSession.projectID,
                    in: context
                )
                let chapters = await session.chapters()
                guard let chapter = chapters.first(where: { $0.id == chapterID }) else {
                    throw ValidationError("No chapter exists with id \(chapterID.uuidString).")
                }

                await session.removeChapter(chapterID)
                _ = try await ProjectsCommand.persistActiveSession(activeSession, in: context)

                let message = "Removed chapter '\(chapter.title)' (\(chapterID.uuidString))."
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("chapters remove"),
                        data: ChapterMutationPayloadDTO(
                            message: message,
                            chapterID: chapterID.uuidString
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    static func parseChapterID(_ string: String) throws -> UUID {
        guard let uuid = UUID(uuidString: string) else {
            throw ValidationError("Invalid chapter ID: '\(string)'. Expected a UUID.")
        }
        return uuid
    }

    static func renderChapterSummaryLine(_ chapter: ChapterRecord) -> String {
        let ordinal = chapter.index + 1
        let textLength = chapter.script.count
        return "\(chapter.id.uuidString) | \(ordinal) | \(chapter.title) | \(textLength) chars | hasSourceAudio: \(chapter.hasSourceAudio)"
    }

    static func resolvedAudioURL(
        from path: String,
        fileManager: FileManager = .default
    ) throws -> URL {
        let resolvedURL = ProjectsCommand.resolvedURL(for: path, fileManager: fileManager)
        guard fileManager.fileExists(atPath: resolvedURL.path) else {
            throw ValidationError("Audio file not found at \(resolvedURL.path).")
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: resolvedURL.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw ValidationError("Audio path must point to a file, got \(resolvedURL.path).")
        }

        return resolvedURL
    }

    static func stagedAssetName(chapterID: UUID, sourceURL: URL) -> String {
        let pathExtension = sourceURL.pathExtension
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard pathExtension.isEmpty == false else {
            return chapterID.uuidString
        }
        return "\(chapterID.uuidString).\(pathExtension)"
    }

    static func stageAudioAsset(
        for chapterID: UUID,
        sourceURL: URL,
        previousAssetName: String?,
        activeSession: ProjectsCommand.ActiveProjectSession,
        fileManager: FileManager = .default
    ) throws -> (assetName: String, sampleRate: Double, durationSeconds: Double) {
        let audioFile = try AVAudioFile(forReading: sourceURL)
        let sampleRate = audioFile.processingFormat.sampleRate
        let durationSeconds = sampleRate > 0 ? Double(audioFile.length) / sampleRate : 0

        let bundleURL = ProjectsCommand.normalizedBundleURL(
            for: activeSession.bundlePath,
            fileManager: fileManager
        )
        let location = ValarProjectBundleLocation(
            projectID: activeSession.projectID,
            title: activeSession.title,
            bundleURL: bundleURL
        )
        try fileManager.createDirectory(at: location.bundleURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: location.assetsDirectory, withIntermediateDirectories: true)

        let assetName = stagedAssetName(chapterID: chapterID, sourceURL: sourceURL)
        try ValarAppPaths.validateRelativePath(assetName, label: "chapter source audio asset")

        let destinationURL = location.assetsDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(destinationURL, within: location.assetsDirectory, fileManager: fileManager)

        if let previousAssetName,
           previousAssetName != assetName {
            let previousAssetURL = location.assetsDirectory.appendingPathComponent(previousAssetName, isDirectory: false)
            try ValarAppPaths.validateContainment(previousAssetURL, within: location.assetsDirectory, fileManager: fileManager)
            if fileManager.fileExists(atPath: previousAssetURL.path) {
                try fileManager.removeItem(at: previousAssetURL)
            }
        }

        if sourceURL.standardizedFileURL != destinationURL.standardizedFileURL,
           fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        if sourceURL.standardizedFileURL != destinationURL.standardizedFileURL {
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
        }

        return (assetName, sampleRate, durationSeconds)
    }
}
