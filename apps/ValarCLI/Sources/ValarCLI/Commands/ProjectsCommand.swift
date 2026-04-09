import ArgumentParser
import Foundation
import ValarCore
import ValarPersistence

struct ProjectsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "projects",
        abstract: "Create, open, inspect, save, and close `.valarproject` bundles.",
        subcommands: [New.self, Open.self, Save.self, Info.self, Close.self]
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
