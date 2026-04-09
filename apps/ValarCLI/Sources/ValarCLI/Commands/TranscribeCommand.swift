@preconcurrency import ArgumentParser
import Foundation
import ValarAudio
import ValarCore
import ValarExport
import ValarModelKit
import ValarPersistence

struct TranscribeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe an audio file, or the attached source audio for a chapter in the active CLI project session."
    )

    static let defaultModelAlias = "Qwen3-ASR-0.6B"
    static let defaultModelIdentifier = "mlx-community/Qwen3-ASR-0.6B-8bit"

    @Argument(help: "Path to the audio file to transcribe. Omit to transcribe an attached chapter source audio asset.")
    var audioFile: String?

    @Option(name: .long, help: "Chapter id whose attached source audio should receive the transcription.")
    var chapter: String?

    @Option(name: .long, help: "Speech-to-text model identifier. Defaults to Qwen3-ASR-0.6B.")
    var model: String = Self.defaultModelAlias

    @Option(name: .long, help: "Destination path for the transcript.")
    var output: String?

    @Option(name: .long, help: "Output format: text (default), json, verbose_json, srt, or vtt. Inferred from --output file extension when omitted.")
    var format: TranscriptFormat = .text

    @Flag(name: .long, help: "Process audio in streaming chunks and print partial transcripts to stderr as each chunk completes.")
    var stream: Bool = false

    mutating func run() async throws {
        let source = try await resolveSource()
        let runtime = try ValarRuntime()
        let request = try await makeRequest(audioURL: source.audioURL, runtime: runtime)

        let transcript: RichTranscriptionResult
        if stream {
            transcript = try await runStreamingTranscription(runtime: runtime, request: request)
        } else {
            do {
                transcript = try await runtime.transcribe(request)
            } catch let error as ValarRuntime.TranscriptionError {
                throw mappedRuntimeError(error)
            } catch {
                throw TranscribeCommandError(message: "Transcription failed: \(OutputFormat.message(for: error))")
            }
        }

        if let chapterTarget = source.chapterTarget {
            let transcriptionJSON = try renderTranscriptionJSON(transcript)
            await chapterTarget.context.projectStore.setTranscription(
                for: chapterTarget.chapter.id,
                in: chapterTarget.activeSession.projectID,
                transcriptionJSON: transcriptionJSON,
                modelID: request.model.rawValue
            )
            _ = try await ProjectsCommand.persistActiveSession(
                chapterTarget.activeSession,
                in: chapterTarget.context
            )
        }

        if let output {
            let outputURL = resolvedPath(from: output)
            do {
                let resolvedFormat = format.resolving(fromFileExtension: outputURL.pathExtension)
                let content = try resolvedFormat.render(transcript)
                let directory = outputURL.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try content.write(to: outputURL, atomically: true, encoding: .utf8)
            } catch {
                throw TranscribeCommandError(
                    message: "Failed to write transcript to '\(outputURL.path)': \(OutputFormat.message(for: error))"
                )
            }
        }

        if OutputContext.jsonRequested {
            try writeJSON(transcript: transcript, source: source)
        } else if let chapterTarget = source.chapterTarget {
            let message = output == nil
                ? "Stored transcription for chapter '\(chapterTarget.chapter.title)' (\(chapterTarget.chapter.id.uuidString))."
                : "Stored transcription for chapter '\(chapterTarget.chapter.title)' (\(chapterTarget.chapter.id.uuidString)) and wrote text to \(resolvedPath(from: output!).path)."
            print(message)
        } else if output == nil {
            let resolvedFormat = format.resolving(fromFileExtension: nil)
            print(try resolvedFormat.render(transcript))
        }
    }

    // Processes audio in streaming chunks, printing partial transcripts to stderr as each
    // chunk completes, and returns the final merged RichTranscriptionResult.
    private func runStreamingTranscription(
        runtime: ValarRuntime,
        request: SpeechToTextRequest
    ) async throws -> RichTranscriptionResult {
        let eventStream = runtime.transcribeChunked(request)
        var completedResult: RichTranscriptionResult?
        let stderr = FileHandle.standardError

        do {
            for try await event in eventStream {
                switch event {
                case .partial(let segment):
                    stderr.write(Data(("[partial] \(segment.text)\n").utf8))
                case .finalSegment:
                    break
                case .completed(let result):
                    completedResult = result
                case .metrics:
                    break
                case .warning(let message):
                    stderr.write(Data(("Warning: \(message)\n").utf8))
                }
            }
        } catch let error as ValarRuntime.TranscriptionError {
            throw mappedRuntimeError(error)
        } catch {
            throw TranscribeCommandError(message: "Transcription failed: \(OutputFormat.message(for: error))")
        }

        guard let result = completedResult else {
            throw TranscribeCommandError(message: "Streaming transcription did not produce a final result.")
        }
        return result
    }

    private func makeRequest(audioURL inputURL: URL, runtime: ValarRuntime) async throws -> SpeechToTextRequest {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw TranscribeCommandError(
                message: "Audio file not found: \(inputURL.path)",
                cliExitCode: .usageError
            )
        }

        let audioData: Data
        do {
            audioData = try Data(contentsOf: inputURL)
        } catch {
            throw TranscribeCommandError(
                message: "Failed to read audio file '\(inputURL.path)': \(OutputFormat.message(for: error))"
            )
        }

        let decodedBuffer: AudioPCMBuffer
        do {
            let ext = inputURL.pathExtension.lowercased()
            if ext == "ogg" || ext == "oga" {
                decodedBuffer = try ChannelAudioImporter().decode(audioData)
            } else {
                decodedBuffer = try await AudioPipeline().decode(audioData, hint: inputURL.pathExtension)
            }
        } catch {
            throw TranscribeCommandError(
                message: "Invalid audio file '\(inputURL.path)': \(OutputFormat.message(for: error))"
            )
        }

        let monoBuffer = downmixToMono(decodedBuffer)
        guard !monoBuffer.channels.isEmpty, let samples = monoBuffer.channels.first, !samples.isEmpty else {
            throw TranscribeCommandError(message: "Audio file '\(inputURL.path)' does not contain any samples.")
        }

        let identifier = await resolvedModelIdentifier(from: model, runtime: runtime)
        return SpeechToTextRequest(
            model: ModelIdentifier(identifier),
            audio: AudioChunk(samples: samples, sampleRate: monoBuffer.format.sampleRate)
        )
    }

    private func resolveSource() async throws -> ResolvedTranscriptionSource {
        if let chapter {
            return try await resolveChapterSource(for: chapter)
        }

        if let audioFile {
            return ResolvedTranscriptionSource(
                audioURL: resolvedPath(from: audioFile),
                chapterTarget: nil
            )
        }

        let context = try ProjectsCommand.ProjectCommandContext()
        let activeSession = try ProjectsCommand.requireActiveSession(in: context)
        try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

        let chapters = await context.projectStore.chapters(for: activeSession.projectID)
        let chaptersWithAudio = chapters.filter { $0.sourceAudioAssetName != nil }

        guard let chapter = chaptersWithAudio.only else {
            throw TranscribeCommandError(
                message: chaptersWithAudio.isEmpty
                    ? "Provide an audio file or attach source audio to a chapter before running `valartts transcribe`."
                    : "Multiple chapters have attached source audio. Re-run with `--chapter <id>`."
            )
        }

        return ResolvedTranscriptionSource(
            audioURL: try chapterAudioURL(
                for: chapter,
                activeSession: activeSession,
                fileManager: context.fileManager
            ),
            chapterTarget: ChapterTranscriptionTarget(
                context: context,
                activeSession: activeSession,
                chapter: chapter
            )
        )
    }

    private func resolveChapterSource(for rawChapterID: String) async throws -> ResolvedTranscriptionSource {
        let chapterID = try ChaptersCommand.parseChapterID(rawChapterID)
        let context = try ProjectsCommand.ProjectCommandContext()
        let activeSession = try ProjectsCommand.requireActiveSession(in: context)
        try await ProjectsCommand.hydrateRuntimeSession(for: activeSession, in: context)

        let chapters = await context.projectStore.chapters(for: activeSession.projectID)
        guard let chapter = chapters.first(where: { $0.id == chapterID }) else {
            throw ValidationError("No chapter exists with id \(chapterID.uuidString).")
        }

        let audioURL = if let audioFile {
            resolvedPath(from: audioFile)
        } else {
            try chapterAudioURL(
                for: chapter,
                activeSession: activeSession,
                fileManager: context.fileManager
            )
        }

        return ResolvedTranscriptionSource(
            audioURL: audioURL,
            chapterTarget: ChapterTranscriptionTarget(
                context: context,
                activeSession: activeSession,
                chapter: chapter
            )
        )
    }

    private func chapterAudioURL(
        for chapter: ChapterRecord,
        activeSession: ProjectsCommand.ActiveProjectSession,
        fileManager: FileManager = .default
    ) throws -> URL {
        guard let assetName = chapter.sourceAudioAssetName else {
            throw ValidationError("Chapter '\(chapter.title)' does not have attached source audio.")
        }

        let bundleURL = ProjectsCommand.normalizedBundleURL(
            for: activeSession.bundlePath,
            fileManager: fileManager
        )
        let location = ValarProjectBundleLocation(
            projectID: activeSession.projectID,
            title: activeSession.title,
            bundleURL: bundleURL
        )
        let assetURL = location.assetsDirectory.appendingPathComponent(assetName, isDirectory: false)
        try ValarAppPaths.validateContainment(assetURL, within: location.assetsDirectory, fileManager: fileManager)

        guard fileManager.fileExists(atPath: assetURL.path) else {
            throw TranscribeCommandError(
                message: "Attached audio asset was not found at \(assetURL.path)."
            )
        }

        return assetURL
    }

    private func resolvedPath(from rawPath: String) -> URL {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return URL(fileURLWithPath: trimmedPath, relativeTo: currentDirectory).standardizedFileURL
    }

    private func resolvedModelIdentifier(from rawIdentifier: String, runtime: ValarRuntime) async -> String {
        let trimmedIdentifier = rawIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedIdentifier.isEmpty || trimmedIdentifier == Self.defaultModelAlias else {
            return trimmedIdentifier
        }

        // Prefer any already-installed ASR model over the hardcoded default.
        let supportedModels = (try? await runtime.modelCatalog.supportedModels()) ?? []
        if let installed = supportedModels.first(where: {
            $0.installState == .installed && $0.descriptor.capabilities.contains(.speechRecognition)
        }) {
            return installed.id.rawValue
        }

        // If the default model's files are on disk but not yet registered, auto-register them.
        let defaultID = ModelIdentifier(Self.defaultModelIdentifier)
        if let catalogModel = supportedModels.first(where: { $0.id == defaultID }),
           let packDir = try? runtime.paths.modelPackDirectory(
               familyID: catalogModel.descriptor.familyID.rawValue,
               modelID: defaultID.rawValue
           ),
           FileManager.default.fileExists(atPath: packDir.path),
           let manifest = try? await runtime.modelCatalog.installationManifest(for: defaultID) {
            let sourceKind: ModelPackSourceKind = catalogModel.providerURL == nil ? .localFile : .remoteURL
            let sourceLocation = catalogModel.providerURL?.absoluteString ?? defaultID.rawValue
            _ = try? await runtime.modelInstaller.install(
                manifest: manifest,
                sourceKind: sourceKind,
                sourceLocation: sourceLocation,
                notes: catalogModel.notes,
                mode: .metadataOnly
            )
        }

        return Self.defaultModelIdentifier
    }

    private func downmixToMono(_ buffer: AudioPCMBuffer) -> AudioPCMBuffer {
        guard buffer.channels.count > 1 else { return buffer }

        let frameCount = buffer.frameCount
        guard frameCount > 0 else {
            return AudioPCMBuffer(mono: [], sampleRate: buffer.format.sampleRate, container: buffer.format.container)
        }

        let channels = buffer.channels.filter { !$0.isEmpty }
        guard channels.isEmpty == false else { return buffer }

        var mixed = Array(repeating: Float.zero, count: frameCount)
        for channel in channels {
            for index in 0 ..< min(channel.count, frameCount) {
                mixed[index] += channel[index]
            }
        }

        let divisor = Float(channels.count)
        let normalized = mixed.map { $0 / divisor }
        return AudioPCMBuffer(
            mono: normalized,
            sampleRate: buffer.format.sampleRate,
            container: buffer.format.container
        )
    }

    private func mappedRuntimeError(_ error: ValarRuntime.TranscriptionError) -> TranscribeCommandError {
        switch error {
        case .modelNotFound(let identifier):
            return TranscribeCommandError(
                message: "Speech-to-text model '\(identifier.rawValue)' is not in the model catalog.\nRun: valartts models list to see supported identifiers.\nTo install the default ASR model, run: valartts models install \(Self.defaultModelIdentifier)",
                cliExitCode: .usageError
            )
        case .modelNotInstalled(let identifier):
            return TranscribeCommandError(
                message: "Speech-to-text model '\(identifier.rawValue)' is not installed.\nRun: valartts models install \(identifier.rawValue)",
                cliExitCode: .usageError
            )
        case .unsupportedModelFamily(let identifier, _):
            return TranscribeCommandError(
                message: "Model '\(identifier.rawValue)' does not support speech-to-text transcription — use an ASR model.\nTo install the default ASR model, run: valartts models install \(Self.defaultModelIdentifier)",
                cliExitCode: .usageError
            )
        case .noCompatibleBackend(let identifier):
            return TranscribeCommandError(
                message: "No compatible inference backend available for '\(identifier.rawValue)'. MLX inference requires Apple Silicon — check system readiness with: valartts doctor"
            )
        case .loadedModelTypeMismatch(let identifier):
            return TranscribeCommandError(
                message: "Model '\(identifier.rawValue)' loaded but is not a speech-to-text model — use an ASR model.\nTo install the default ASR model, run: valartts models install \(Self.defaultModelIdentifier)"
            )
        }
    }

    private func renderTranscriptionJSON(_ transcript: RichTranscriptionResult) throws -> String {
        let flat = TranscriptionResult(transcript)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(flat), as: UTF8.self)
    }

    private func writeJSON(
        transcript: RichTranscriptionResult,
        source: ResolvedTranscriptionSource
    ) throws {
        let payload = TranscriptionPayload(
            transcript: transcript.text,
            segments: transcript.segments.map(\.text),
            audioPath: source.audioURL.path,
            chapterID: source.chapterTarget?.chapter.id.uuidString,
            persistedToProject: source.chapterTarget != nil,
            outputPath: output.map { resolvedPath(from: $0).path }
        )
        try OutputFormat.writeSuccess(command: OutputFormat.commandPath("transcribe"), data: payload)
    }
}

private struct TranscriptionPayload: Codable, Sendable {
    let transcript: String
    let segments: [String]
    let audioPath: String
    let chapterID: String?
    let persistedToProject: Bool
    let outputPath: String?
}

private struct ResolvedTranscriptionSource {
    let audioURL: URL
    let chapterTarget: ChapterTranscriptionTarget?
}

private struct ChapterTranscriptionTarget {
    let context: ProjectsCommand.ProjectCommandContext
    let activeSession: ProjectsCommand.ActiveProjectSession
    let chapter: ChapterRecord
}

private extension Array {
    var only: Element? {
        count == 1 ? first : nil
    }
}

enum TranscriptFormat: String, ExpressibleByArgument, CaseIterable {
    case text
    case json
    case verbose_json
    case srt
    case vtt

    /// Returns the format inferred from a file extension, falling back to `self` when the
    /// extension is unrecognised or `fileExtension` is nil.
    func resolving(fromFileExtension fileExtension: String?) -> TranscriptFormat {
        guard self == .text, let ext = fileExtension else { return self }
        switch ext.lowercased() {
        case "srt": return .srt
        case "vtt": return .vtt
        default:    return self
        }
    }

    func render(_ result: RichTranscriptionResult) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        switch self {
        case .text:         return result.text
        case .json:         return String(decoding: try encoder.encode(TranscriptionResult(result)), as: UTF8.self)
        case .verbose_json: return String(decoding: try encoder.encode(result), as: UTF8.self)
        case .srt:          return TranscriptionFormatter.srt(from: result)
        case .vtt:          return TranscriptionFormatter.vtt(from: result)
        }
    }
}

private struct TranscribeCommandError: LocalizedError, CLIExitCodeProviding {
    let message: String
    let cliExitCode: CLIExitCode

    init(message: String, cliExitCode: CLIExitCode = .failure) {
        self.message = message
        self.cliExitCode = cliExitCode
    }

    var errorDescription: String? {
        message
    }
}
