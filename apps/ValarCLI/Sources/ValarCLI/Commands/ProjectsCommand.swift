import ArgumentParser
import CryptoKit
import Foundation
import ValarCore
import ValarModelKit
import ValarPersistence

struct ProjectsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "projects",
        abstract: "Create, open, inspect, save, and close `.valarproject` bundles.",
        subcommands: [New.self, Import.self, Open.self, Save.self, Info.self, Close.self, Lint.self, ExportPack.self]
    )

    mutating func run() throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension ProjectsCommand {
    struct New: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Create a new `.valarproject` bundle and open it as the active CLI session."
        )

        @Option(name: .long, help: "Project title to store in the new bundle.")
        var name: String

        @Option(name: .long, help: "Bundle output path.")
        var path: String

        mutating func run() throws {
            let name = self.name
            let path = self.path
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                try ProjectsCommand.ensureNoActiveSession(in: context)

                let bundleURL = ProjectsCommand.normalizedBundleURL(for: path, fileManager: context.fileManager)
                let project = try await context.projectStore.create(title: name, notes: nil)
                await context.projectStore.updateBundleURL(bundleURL, for: project.id)
                let createdAt = Date.now
                let initialBundle = ProjectBundle(
                    manifest: ProjectBundleManifest(
                        version: 1,
                        createdAt: createdAt,
                        projectID: project.id,
                        title: project.title,
                        chapters: []
                    ),
                    snapshot: ProjectBundleSnapshot(
                        project: project,
                        chapters: [],
                        renderJobs: [],
                        exports: [],
                        speakers: []
                    )
                )
                _ = await context.runtime.createDocumentSession(for: initialBundle)
                let session = try await ProjectsCommand.requireRuntimeSession(
                    for: project.id,
                    in: context
                )
                let snapshot = try await session.snapshot(
                    preferredModelID: nil,
                    createdAt: createdAt,
                    version: 1
                )

                let location = ValarProjectBundleLocation(
                    projectID: project.id,
                    title: project.title,
                    bundleURL: bundleURL
                )
                let manifest = try ProjectBundleWriter(fileManager: context.fileManager).write(
                    snapshot.snapshot,
                    to: location,
                    createdAt: snapshot.manifest.createdAt
                )

                try context.sessionStore.save(
                    ActiveProjectSession(
                        version: manifest.version,
                        projectID: project.id,
                        title: project.title,
                        bundlePath: location.bundleURL.path,
                        createdAt: manifest.createdAt,
                        openedAt: .now
                    )
                )

                let message = "Created project '\(project.title)' at \(location.bundleURL.path)"
                if jsonRequested {
                    let sessionDTO = ProjectsCommand.projectSessionDTO(
                        ActiveProjectSession(
                            version: manifest.version,
                            projectID: project.id,
                            title: project.title,
                            bundlePath: location.bundleURL.path,
                            createdAt: manifest.createdAt,
                            openedAt: .now
                        )
                    )
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects new"),
                        data: ProjectSessionPayloadDTO(
                            message: message,
                            project: sessionDTO
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Open: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Load an existing `.valarproject` bundle into the active CLI session."
        )

        @Argument(help: "Bundle path.")
        var path: String

        mutating func run() throws {
            let path = self.path
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                try ProjectsCommand.ensureNoActiveSession(in: context)

                let bundleURL = ProjectsCommand.openableBundleURL(for: path, fileManager: context.fileManager)
                let bundle = try ProjectBundleReader(fileManager: context.fileManager).read(from: bundleURL)
                _ = await context.runtime.createDocumentSession(for: bundle)
                await context.projectStore.updateBundleURL(bundleURL, for: bundle.snapshot.project.id)

                try context.sessionStore.save(
                    ActiveProjectSession(
                        version: bundle.manifest.version,
                        projectID: bundle.snapshot.project.id,
                        title: bundle.snapshot.project.title,
                        bundlePath: bundleURL.path,
                        createdAt: bundle.manifest.createdAt,
                        openedAt: .now
                    )
                )

                let session = ActiveProjectSession(
                    version: bundle.manifest.version,
                    projectID: bundle.snapshot.project.id,
                    title: bundle.snapshot.project.title,
                    bundlePath: bundleURL.path,
                    createdAt: bundle.manifest.createdAt,
                    openedAt: .now
                )
                let message = "Opened project '\(bundle.snapshot.project.title)' from \(bundleURL.path)"
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects open"),
                        data: ProjectSessionPayloadDTO(
                            message: message,
                            project: ProjectsCommand.projectSessionDTO(session)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Import: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "import",
            abstract: "Import a TXT, Markdown, or simple dialogue script file into a new `.valarproject` bundle."
        )

        @Argument(help: "TXT, Markdown, or simple dialogue script file to import.")
        var input: String

        @Option(name: .long, help: "Project title. Defaults to the input filename.")
        var name: String?

        @Option(name: .long, help: "Bundle output path. Defaults to Valar's Projects directory.")
        var path: String?

        @Option(name: .long, help: "Split mode: markdown-headings, paragraphs, lines, dialogue, whole-document.")
        var splitMode: String = "markdown-headings"

        @Option(name: .long, help: "Default speaker label for imported segments.")
        var speaker: String?

        mutating func run() throws {
            let input = self.input
            let explicitName = self.name
            let explicitPath = self.path
            let splitMode = self.splitMode
            let speaker = self.speaker
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                try ProjectsCommand.ensureNoActiveSession(in: context)

                let inputURL = ProjectsCommand.resolvedFileURL(for: input, fileManager: context.fileManager)
                guard context.fileManager.fileExists(atPath: inputURL.path) else {
                    throw ValidationError("Input file does not exist: \(inputURL.path)")
                }
                let fileExtension = inputURL.pathExtension.lowercased()
                guard ["txt", "md", "markdown", "script"].contains(fileExtension) || fileExtension.isEmpty else {
                    throw ValidationError("Unsupported import extension '.\(fileExtension)'. Start with TXT, Markdown, or script files.")
                }

                let rawText = try String(contentsOf: inputURL, encoding: .utf8)
                let title = explicitName?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty
                    ?? inputURL.deletingPathExtension().lastPathComponent
                let bundleURL = try ProjectsCommand.importBundleURL(
                    explicitPath: explicitPath,
                    title: title,
                    context: context
                )
                let drafts = ProjectTextImporter.parse(
                    text: rawText,
                    fallbackTitle: title,
                    splitMode: splitMode,
                    defaultSpeakerLabel: speaker
                )
                guard drafts.isEmpty == false else {
                    throw ValidationError("No importable text segments were found in \(inputURL.path).")
                }

                let project = try await context.projectStore.create(
                    title: title,
                    notes: "Imported from \(inputURL.lastPathComponent) using split mode '\(splitMode)'."
                )
                await context.projectStore.updateBundleURL(bundleURL, for: project.id)
                let chapters = drafts.enumerated().map { index, draft in
                    ChapterRecord(
                        projectID: project.id,
                        index: index,
                        title: draft.title,
                        script: draft.text,
                        speakerLabel: draft.speakerLabel,
                        estimatedDurationSeconds: Double(draft.text.count) / 14.0
                    )
                }

                let createdAt = Date.now
                let bundle = ProjectBundle(
                    manifest: ProjectBundleManifest(
                        version: 1,
                        createdAt: createdAt,
                        projectID: project.id,
                        title: project.title,
                        chapters: chapters.map {
                            ProjectBundleManifest.ChapterSummary(
                                id: $0.id,
                                index: $0.index,
                                title: $0.title
                            )
                        }
                    ),
                    snapshot: ProjectBundleSnapshot(
                        project: project,
                        chapters: chapters,
                        renderJobs: [],
                        exports: [],
                        speakers: []
                    )
                )
                _ = await context.runtime.createDocumentSession(for: bundle)
                let session = try await ProjectsCommand.requireRuntimeSession(
                    for: project.id,
                    in: context
                )
                let snapshot = try await session.snapshot(
                    preferredModelID: nil,
                    createdAt: createdAt,
                    version: 1
                )
                let location = ValarProjectBundleLocation(
                    projectID: project.id,
                    title: project.title,
                    bundleURL: bundleURL
                )
                let manifest = try ProjectBundleWriter(fileManager: context.fileManager).write(
                    snapshot.snapshot,
                    to: location,
                    createdAt: snapshot.manifest.createdAt
                )
                let activeSession = ActiveProjectSession(
                    version: manifest.version,
                    projectID: project.id,
                    title: project.title,
                    bundlePath: location.bundleURL.path,
                    createdAt: manifest.createdAt,
                    openedAt: .now
                )
                try context.sessionStore.save(activeSession)

                let message = "Imported \(chapters.count) chapter(s) into '\(project.title)' at \(location.bundleURL.path)"
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects import"),
                        data: ProjectSessionPayloadDTO(
                            message: message,
                            project: ProjectsCommand.projectSessionDTO(activeSession)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Save: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Persist the active CLI session back to its `.valarproject` bundle."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let bundleURL = URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true)
                await context.projectStore.updateBundleURL(bundleURL, for: activeSession.projectID)
                let documentSession = try await ProjectsCommand.requireRuntimeSession(
                    for: activeSession.projectID,
                    in: context
                )
                let snapshot = try await documentSession.snapshot(
                    preferredModelID: nil,
                    createdAt: activeSession.createdAt,
                    version: activeSession.version
                )
                let location = ValarProjectBundleLocation(
                    projectID: activeSession.projectID,
                    title: snapshot.snapshot.project.title,
                    bundleURL: bundleURL
                )
                let manifest = try ProjectBundleWriter(fileManager: context.fileManager).write(
                    snapshot.snapshot,
                    to: location,
                    createdAt: activeSession.createdAt
                )

                try context.sessionStore.save(
                    ActiveProjectSession(
                        version: manifest.version,
                        projectID: activeSession.projectID,
                        title: snapshot.snapshot.project.title,
                        bundlePath: location.bundleURL.path,
                        createdAt: manifest.createdAt,
                        openedAt: activeSession.openedAt
                    )
                )

                let session = ActiveProjectSession(
                    version: manifest.version,
                    projectID: activeSession.projectID,
                    title: snapshot.snapshot.project.title,
                    bundlePath: location.bundleURL.path,
                    createdAt: manifest.createdAt,
                    openedAt: activeSession.openedAt
                )
                let message = "Saved project '\(snapshot.snapshot.project.title)' to \(location.bundleURL.path)"
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects save"),
                        data: ProjectSessionPayloadDTO(
                            message: message,
                            project: ProjectsCommand.projectSessionDTO(session)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Info: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print metadata for the active CLI project session."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

                let project = try await ProjectsCommand.requireProjectRecord(
                    for: activeSession.projectID,
                    in: context
                )
                let chapters = await context.projectStore.chapters(for: activeSession.projectID)
                let renderJobs = await context.projectStore.renderJobs(for: activeSession.projectID)
                let exports = await context.projectStore.exports(for: activeSession.projectID)
                let speakers = await context.projectStore.speakers(for: activeSession.projectID)

                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects info"),
                        data: ProjectInfoPayloadDTO(
                            message: "Loaded project metadata for '\(project.title)'.",
                            project: ProjectInfoDTO(
                                title: project.title,
                                projectID: project.id.uuidString,
                                bundlePath: activeSession.bundlePath,
                                createdAt: OutputFormat.iso8601String(from: activeSession.createdAt),
                                openedAt: OutputFormat.iso8601String(from: activeSession.openedAt),
                                chapters: chapters.count,
                                renderJobs: renderJobs.count,
                                exports: exports.count,
                                speakers: speakers.count
                            )
                        )
                    )
                    return
                }

                print("Title: \(project.title)")
                print("Project ID: \(project.id.uuidString)")
                print("Bundle: \(activeSession.bundlePath)")
                print("Created At: \(ProjectsCommand.iso8601String(from: activeSession.createdAt))")
                print("Opened At: \(ProjectsCommand.iso8601String(from: activeSession.openedAt))")
                print("Chapters: \(chapters.count)")
                print("Render Jobs: \(renderJobs.count)")
                print("Exports: \(exports.count)")
                print("Speakers: \(speakers.count)")
            }
        }
    }

    struct Close: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Tear down the active CLI project session."
        )

        mutating func run() throws {
            let jsonRequested = OutputContext.jsonRequested
            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                let activeSession = try ProjectsCommand.requireActiveSession(in: context)
                await context.runtime.closeDocumentSession(for: activeSession.projectID)
                try context.sessionStore.clear()

                let message = "Closed project '\(activeSession.title)'"
                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects close"),
                        data: ProjectSessionPayloadDTO(
                            message: message,
                            project: ProjectsCommand.projectSessionDTO(activeSession)
                        )
                    )
                } else {
                    print(message)
                }
            }
        }
    }

    struct Lint: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "lint",
            abstract: "Lint project script markup, cast setup, and selected model fit."
        )

        @Option(name: .long, help: "Optional `.valarproject` bundle path. Defaults to the active project session.")
        var path: String?

        @Option(name: .long, help: "Optional model identifier to check against expressive tags and cast consistency. Defaults to the bundle model when present.")
        var model: String?

        mutating func run() throws {
            let explicitPath = path
            let explicitModel = model
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                let loaded = try await ProjectsCommand.loadProjectSnapshot(
                    explicitPath: explicitPath,
                    context: context
                )
                let resolvedModel = try await ProjectsCommand.lookupSpeechModel(
                    explicitModel ?? loaded.snapshot.modelID,
                    in: context
                )
                let payload = ProjectScriptMarkup.lintProject(
                    project: loaded.snapshot.project,
                    chapters: loaded.snapshot.chapters,
                    speakers: loaded.snapshot.speakers,
                    model: resolvedModel
                )

                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects lint"),
                        data: payload
                    )
                    return
                }

                ProjectsCommand.printScriptLint(payload)
            }
        }
    }

    struct ExportPack: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "export-pack",
            abstract: "Write an agent-friendly project export pack manifest."
        )

        @Option(name: .long, help: "Optional `.valarproject` bundle path. Defaults to the active project session.")
        var path: String?

        @Option(name: .long, help: "Output directory for the pack manifest. Defaults to Exports/PublishPack in the bundle.")
        var outputDir: String?

        mutating func run() throws {
            let explicitPath = path
            let explicitOutputDir = outputDir
            let jsonRequested = OutputContext.jsonRequested

            try ProjectsCommand.runAsync {
                let context = try ProjectCommandContext()
                let loaded = try await ProjectsCommand.loadProjectSnapshot(
                    explicitPath: explicitPath,
                    context: context
                )
                let outputDirectory: URL
                if let explicitOutputDir = explicitOutputDir?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !explicitOutputDir.isEmpty {
                    outputDirectory = ProjectsCommand.resolvedURL(
                        for: explicitOutputDir,
                        fileManager: context.fileManager
                    )
                } else {
                    outputDirectory = loaded.bundleURL
                        .appendingPathComponent("Exports", isDirectory: true)
                        .appendingPathComponent("PublishPack", isDirectory: true)
                }

                try context.fileManager.createDirectory(
                    at: outputDirectory,
                    withIntermediateDirectories: true
                )

                let generatedAt = Date.now
                let manifest = try ProjectsCommand.exportPackManifest(
                    snapshot: loaded.snapshot,
                    bundleURL: loaded.bundleURL,
                    generatedAt: generatedAt,
                    fileManager: context.fileManager
                )
                let manifestURL = outputDirectory.appendingPathComponent("valar-export-pack.json", isDirectory: false)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(manifest).write(to: manifestURL, options: .atomic)

                let payload = ProjectExportPackPayloadDTO(
                    message: "Wrote project export pack manifest for '\(loaded.snapshot.project.title)'.",
                    manifestPath: manifestURL.path,
                    manifest: manifest
                )

                if jsonRequested {
                    try OutputFormat.writeSuccess(
                        command: OutputFormat.commandPath("projects export-pack"),
                        data: payload
                    )
                    return
                }

                print(payload.message)
                print("Manifest: \(manifestURL.path)")
                print("Chapters: \(manifest.chapterCount)")
                print("Artifacts: \(manifest.artifacts.count)")
            }
        }
    }

    struct ActiveProjectSession: Codable, Sendable {
        var version: Int
        var projectID: UUID
        var title: String
        var bundlePath: String
        var createdAt: Date
        var openedAt: Date
    }

    struct ProjectCommandContext {
        let fileManager: FileManager
        let paths: ValarAppPaths
        let runtime: ValarRuntime
        let projectStore: any ProjectStoring
        let sessionStore: SessionStore

        init(fileManager: FileManager = .default) throws {
            self.fileManager = fileManager
            self.paths = Self.resolvePaths(fileManager: fileManager)
            self.runtime = try ValarRuntime(paths: paths, fileManager: fileManager)
            self.projectStore = runtime.projectStore
            self.sessionStore = SessionStore(paths: paths, fileManager: fileManager)
        }

        private static func resolvePaths(fileManager: FileManager) -> ValarAppPaths {
            let environment = ProcessInfo.processInfo.environment
            if let override = environment["VALARTTS_CLI_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
               override.isEmpty == false {
                return ValarAppPaths(
                    baseURL: URL(fileURLWithPath: override, isDirectory: true).standardizedFileURL
                )
            }
            return ValarAppPaths(fileManager: fileManager)
        }
    }

    // Persist the active project because each `valartts` invocation is a fresh process.
    struct SessionStore {
        private let fileManager: FileManager
        private let stateURL: URL

        init(paths: ValarAppPaths, fileManager: FileManager = .default) {
            self.fileManager = fileManager
            self.stateURL = paths.applicationSupport
                .appendingPathComponent("CLI", isDirectory: true)
                .appendingPathComponent("active-project-session.json", isDirectory: false)
        }

        func load() throws -> ActiveProjectSession? {
            guard fileManager.fileExists(atPath: stateURL.path) else {
                return nil
            }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(
                ActiveProjectSession.self,
                from: Data(contentsOf: stateURL)
            )
        }

        func save(_ session: ActiveProjectSession) throws {
            let directoryURL = stateURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(session)
            try data.write(to: stateURL, options: .atomic)
        }

        func clear() throws {
            guard fileManager.fileExists(atPath: stateURL.path) else {
                return
            }
            try fileManager.removeItem(at: stateURL)
        }
    }

    static func ensureNoActiveSession(in context: ProjectCommandContext) throws {
        if let activeSession = try context.sessionStore.load() {
            throw ValidationError(
                "A project session is already open for '\(activeSession.title)'. Run `valartts projects save` or `valartts projects close` first."
            )
        }
    }

    static func requireActiveSession(in context: ProjectCommandContext) throws -> ActiveProjectSession {
        guard let activeSession = try context.sessionStore.load() else {
            throw ValidationError("No project session is currently open.")
        }
        return activeSession
    }

    static func requireProjectRecord(
        for projectID: UUID,
        in context: ProjectCommandContext
    ) async throws -> ProjectRecord {
        let project = await context.projectStore
            .allProjects()
            .first { $0.id == projectID }

        guard let project else {
            throw CLICommandError(message: "Project \(projectID.uuidString) is not available in the active CLI session.")
        }
        return project
    }

    static func hydrateRuntimeSession(
        for activeSession: ActiveProjectSession,
        in context: ProjectCommandContext
    ) async throws {
        if await context.runtime.documentSession(for: activeSession.projectID) != nil {
            await context.projectStore.updateBundleURL(
                URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true),
                for: activeSession.projectID
            )
            return
        }

        let bundleURL = URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true)
        let bundle = try ProjectBundleReader(fileManager: context.fileManager).read(from: bundleURL)
        _ = await context.runtime.createDocumentSession(for: bundle)
        await context.projectStore.updateBundleURL(bundleURL, for: bundle.snapshot.project.id)
    }

    static func normalizedBundleURL(for path: String, fileManager: FileManager) -> URL {
        var bundleURL = resolvedURL(for: path, fileManager: fileManager)
        if bundleURL.pathExtension != "valarproject" {
            bundleURL.appendPathExtension("valarproject")
        }
        return bundleURL
    }

    static func openableBundleURL(for path: String, fileManager: FileManager) -> URL {
        let rawURL = resolvedURL(for: path, fileManager: fileManager)
        if fileManager.fileExists(atPath: rawURL.path) {
            return rawURL
        }

        let normalizedURL = normalizedBundleURL(for: path, fileManager: fileManager)
        return normalizedURL
    }

    static func resolvedFileURL(for path: String, fileManager: FileManager) -> URL {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if (trimmedPath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: trimmedPath, isDirectory: false).standardizedFileURL
        }

        let currentDirectory = URL(
            fileURLWithPath: fileManager.currentDirectoryPath,
            isDirectory: true
        )
        return URL(
            fileURLWithPath: trimmedPath,
            relativeTo: currentDirectory
        ).standardizedFileURL
    }

    static func resolvedURL(for path: String, fileManager: FileManager) -> URL {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if (trimmedPath as NSString).isAbsolutePath {
            return URL(fileURLWithPath: trimmedPath, isDirectory: true).standardizedFileURL
        }

        let currentDirectory = URL(
            fileURLWithPath: fileManager.currentDirectoryPath,
            isDirectory: true
        )
        return URL(
            fileURLWithPath: trimmedPath,
            relativeTo: currentDirectory
        ).standardizedFileURL
    }

    static func importBundleURL(
        explicitPath: String?,
        title: String,
        context: ProjectCommandContext
    ) throws -> URL {
        if let explicitPath = explicitPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           explicitPath.isEmpty == false {
            return normalizedBundleURL(for: explicitPath, fileManager: context.fileManager)
        }

        try context.fileManager.createDirectory(
            at: context.paths.projectsDirectory,
            withIntermediateDirectories: true
        )
        return context.paths.projectsDirectory
            .appendingPathComponent(context.paths.sanitizeBundleName(title), isDirectory: true)
            .appendingPathExtension("valarproject")
            .standardizedFileURL
    }

    static func iso8601String(from date: Date) -> String {
        OutputFormat.iso8601String(from: date)
    }

    static func persistActiveSession(
        _ activeSession: ActiveProjectSession,
        in context: ProjectCommandContext
    ) async throws -> ActiveProjectSession {
        let bundleURL = URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true)
        await context.projectStore.updateBundleURL(bundleURL, for: activeSession.projectID)

        let session = try await requireRuntimeSession(
            for: activeSession.projectID,
            in: context
        )
        let snapshot = try await session.snapshot(
            preferredModelID: nil,
            createdAt: activeSession.createdAt,
            version: activeSession.version
        )
        let location = ValarProjectBundleLocation(
            projectID: activeSession.projectID,
            title: snapshot.snapshot.project.title,
            bundleURL: bundleURL
        )
        let manifest = try ProjectBundleWriter(fileManager: context.fileManager).write(
            snapshot.snapshot,
            to: location,
            createdAt: activeSession.createdAt
        )

        let updatedSession = ActiveProjectSession(
            version: manifest.version,
            projectID: activeSession.projectID,
            title: snapshot.snapshot.project.title,
            bundlePath: location.bundleURL.path,
            createdAt: manifest.createdAt,
            openedAt: activeSession.openedAt
        )
        try context.sessionStore.save(updatedSession)
        return updatedSession
    }

    static func requireRuntimeSession(
        for projectID: UUID,
        in context: ProjectCommandContext
    ) async throws -> any DocumentSession {
        guard let session = await context.runtime.documentSession(for: projectID) else {
            throw CLICommandError(message: "Project \(projectID.uuidString) is not available in the active CLI session.")
        }
        return session
    }

    static func runAsync(
        _ operation: @escaping @Sendable () async throws -> Void
    ) throws {
        let errorURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            defer { semaphore.signal() }

            do {
                try await operation()
            } catch {
                let capturedError: CapturedAsyncError
                if let exitCode = error as? ExitCode {
                    capturedError = CapturedAsyncError(
                        kind: .exitCode,
                        message: nil,
                        exitCode: exitCode.rawValue
                    )
                } else if let classifiedError = error as? CLIExitCodeProviding,
                          classifiedError.cliExitCode == .usageError {
                    capturedError = CapturedAsyncError(
                        kind: .validation,
                        message: OutputFormat.message(for: error),
                        exitCode: nil
                    )
                } else if let validationError = error as? ValidationError {
                    capturedError = CapturedAsyncError(
                        kind: .validation,
                        message: validationError.message,
                        exitCode: nil
                    )
                } else {
                    capturedError = CapturedAsyncError(
                        kind: .failure,
                        message: OutputFormat.message(for: error),
                        exitCode: nil
                    )
                }

                let encoder = JSONEncoder()
                if let data = try? encoder.encode(capturedError) {
                    try? data.write(to: errorURL, options: .atomic)
                }
            }
        }

        semaphore.wait()
        defer { try? FileManager.default.removeItem(at: errorURL) }

        guard FileManager.default.fileExists(atPath: errorURL.path) else {
            return
        }

        guard let data = try? Data(contentsOf: errorURL),
              let capturedError = try? JSONDecoder().decode(CapturedAsyncError.self, from: data) else {
            throw CLICommandError(message: "Command failed.")
        }

        switch capturedError.kind {
        case .validation:
            throw ValidationError(capturedError.message ?? "Command failed.")
        case .failure:
            throw CLICommandError(message: capturedError.message ?? "Command failed.")
        case .exitCode:
            throw ExitCode(capturedError.exitCode ?? CLIExitCode.failure.rawValue)
        }
    }

    static func projectSessionDTO(_ session: ActiveProjectSession) -> ProjectSessionDTO {
        ProjectSessionDTO(
            version: session.version,
            projectID: session.projectID.uuidString,
            title: session.title,
            bundlePath: session.bundlePath,
            createdAt: OutputFormat.iso8601String(from: session.createdAt),
            openedAt: OutputFormat.iso8601String(from: session.openedAt)
        )
    }

    struct LoadedProjectSnapshot {
        let snapshot: ProjectBundleSnapshot
        let bundleURL: URL
    }

    static func loadProjectSnapshot(
        explicitPath: String?,
        context: ProjectCommandContext
    ) async throws -> LoadedProjectSnapshot {
        if let explicitPath = explicitPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicitPath.isEmpty {
            let bundleURL = openableBundleURL(for: explicitPath, fileManager: context.fileManager)
            let bundle = try ProjectBundleReader(fileManager: context.fileManager).read(from: bundleURL)
            return LoadedProjectSnapshot(snapshot: bundle.snapshot, bundleURL: bundleURL)
        }

        let activeSession = try requireActiveSession(in: context)
        try await hydrateRuntimeSession(for: activeSession, in: context)
        let project = try await requireProjectRecord(for: activeSession.projectID, in: context)
        let bundleURL = URL(fileURLWithPath: activeSession.bundlePath, isDirectory: true)
        let manifestModelID = try? ProjectBundleReader(fileManager: context.fileManager)
            .read(from: bundleURL)
            .manifest
            .modelID
        return LoadedProjectSnapshot(
            snapshot: ProjectBundleSnapshot(
                project: project,
                modelID: manifestModelID ?? nil,
                chapters: await context.projectStore.chapters(for: activeSession.projectID),
                renderJobs: await context.projectStore.renderJobs(for: activeSession.projectID),
                exports: await context.projectStore.exports(for: activeSession.projectID),
                speakers: await context.projectStore.speakers(for: activeSession.projectID)
            ),
            bundleURL: bundleURL
        )
    }

    static func lookupSpeechModel(
        _ rawModelID: String?,
        in context: ProjectCommandContext
    ) async throws -> CatalogModel? {
        guard let rawModelID = rawModelID?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawModelID.isEmpty else {
            return nil
        }
        let models = try await context.runtime.modelCatalog.supportedModels()
        guard let model = models.first(where: { $0.id.rawValue == rawModelID }) else {
            throw ValidationError("Unknown model '\(rawModelID)'. Run 'valartts models list' to see supported identifiers.")
        }
        guard model.descriptor.capabilities.contains(.speechSynthesis) else {
            throw ValidationError("Model '\(rawModelID)' is not a speech synthesis model.")
        }
        return model
    }

    static func printScriptLint(_ payload: ProjectScriptLintPayloadDTO) {
        print(payload.message)
        if let projectTitle = payload.projectTitle {
            print("Project: \(projectTitle)")
        }
        if let modelID = payload.modelID {
            print("Model: \(modelID)")
        }
        print("Parsed script lines: \(payload.lines.count)")
        print("Issues: \(payload.issueCount) (\(payload.warningCount) warning, \(payload.errorCount) error)")

        if let voiceBible = payload.voiceBible, !voiceBible.profiles.isEmpty {
            print("")
            print("Voice bible:")
            for profile in voiceBible.profiles {
                let voice = profile.voiceModelID ?? "unassigned"
                let warnings = profile.warnings.isEmpty ? "" : " | \(profile.warnings.joined(separator: "; "))"
                print("  - \(profile.name): \(profile.segmentCount) segment(s), \(voice), \(profile.language)\(warnings)")
            }
        }

        guard !payload.issues.isEmpty else { return }
        print("")
        print("severity\tcode\tline\tspeaker\tmessage")
        for issue in payload.issues {
            print([
                issue.severity,
                issue.code,
                issue.lineNumber.map(String.init) ?? "-",
                issue.speakerLabel ?? "-",
                issue.message,
            ].map(tabSafe).joined(separator: "\t"))
        }
    }

    static func exportPackManifest(
        snapshot: ProjectBundleSnapshot,
        bundleURL: URL,
        generatedAt: Date,
        fileManager: FileManager
    ) throws -> ProjectExportPackManifestDTO {
        let chapters = snapshot.chapters.sorted { lhs, rhs in
            lhs.index == rhs.index ? lhs.title < rhs.title : lhs.index < rhs.index
        }
        let chapterDTOs = chapters.map { chapter in
            ProjectExportPackChapterDTO(
                id: chapter.id.uuidString,
                index: chapter.index,
                title: chapter.title,
                textLength: chapter.script.count,
                speakerLabel: chapter.speakerLabel,
                sourceHash: ProjectScriptMarkup.sourceHash(
                    title: chapter.title,
                    text: chapter.script,
                    speakerLabel: chapter.speakerLabel
                )
            )
        }
        let exportsDirectory = bundleURL.appendingPathComponent("Exports", isDirectory: true)
        let artifacts = snapshot.exports.sorted { $0.createdAt < $1.createdAt }.map { export in
            let fileName = safeExportFileName(export.fileName, fallbackID: export.id)
            let fileURL = exportsDirectory.appendingPathComponent(fileName, isDirectory: false)
            let exists = fileManager.fileExists(atPath: fileURL.path)
            let byteCount = exists ? fileSize(at: fileURL) : nil
            let checksum = export.checksum ?? (exists ? try? streamingSHA256(at: fileURL) : nil)
            return ProjectExportArtifactDTO(
                id: export.id.uuidString,
                kind: "chapter_audio",
                path: "Exports/\(fileName)",
                mimeType: mimeType(for: fileURL),
                checksum: checksum,
                byteCount: byteCount
            )
        }
        let voiceBible = ProjectScriptMarkup.voiceBible(
            projectID: snapshot.project.id,
            chapters: chapters,
            speakers: snapshot.speakers,
            generatedAt: generatedAt
        )
        let notes = artifacts.isEmpty
            ? ["No chapter audio exports were recorded yet. Run `valartts exports create --chapter <id>` first."]
            : ["Artifact paths are relative to the `.valarproject` bundle root.", "Checksums use SHA-256 when available."]

        return ProjectExportPackManifestDTO(
            generatedAt: OutputFormat.iso8601String(from: generatedAt),
            projectID: snapshot.project.id.uuidString,
            projectTitle: snapshot.project.title,
            modelID: snapshot.modelID,
            chapterCount: chapters.count,
            chapters: chapterDTOs,
            voiceBible: voiceBible,
            artifacts: artifacts,
            notes: notes
        )
    }

    static func fileSize(at url: URL) -> Int? {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
              let size = values.fileSize else {
            return nil
        }
        return size
    }

    static func safeExportFileName(_ rawValue: String, fallbackID: UUID) -> String {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let fileName = URL(fileURLWithPath: trimmed, isDirectory: false).lastPathComponent
        guard !fileName.isEmpty, fileName != "." else {
            return "\(fallbackID.uuidString).wav"
        }
        return fileName
    }

    static func mimeType(for url: URL) -> String? {
        switch url.pathExtension.lowercased() {
        case "wav": return "audio/wav"
        case "m4a": return "audio/mp4"
        case "mp3": return "audio/mpeg"
        case "json": return "application/json"
        case "srt": return "application/x-subrip"
        case "vtt": return "text/vtt"
        default: return nil
        }
    }

    static func streamingSHA256(at url: URL) throws -> String {
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { try? fileHandle.close() }

        var hasher = SHA256()
        let chunkSize = 64 * 1024

        while let chunk = try fileHandle.read(upToCount: chunkSize), chunk.isEmpty == false {
            hasher.update(data: chunk)
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func tabSafe(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\t", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
    }
}

private struct CapturedAsyncError: Codable {
    enum Kind: String, Codable {
        case validation
        case failure
        case exitCode
    }

    let kind: Kind
    let message: String?
    let exitCode: Int32?
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
