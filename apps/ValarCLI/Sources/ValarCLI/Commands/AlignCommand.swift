@preconcurrency import ArgumentParser
import Foundation
import ValarAudio
import ValarCore
import ValarModelKit

struct AlignCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "align",
        abstract: "Align a transcript to an audio file and emit word-level timestamps."
    )

    static let defaultModelID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    @Argument(help: "Path to the audio file to align.")
    var audioFile: String

    @Option(name: .long, help: "Transcript text, or @path/to/file.txt to load transcript text from a file.")
    var transcript: String

    @Option(name: .long, help: "Aligner model identifier to load.")
    var model: String = defaultModelID

    @Option(name: .long, help: "Optional BCP-47 language hint, e.g. 'en'.")
    var language: String?

    @Option(name: .long, help: "Destination path for the alignment JSON output.")
    var output: String?

    mutating func run() async throws {
        let runtime = try ValarRuntime()
        let fileManager = FileManager.default
        let audioURL = try Self.resolvedInputURL(from: audioFile, fileManager: fileManager)
        let transcript = try Self.resolvedTranscript(from: transcript, fileManager: fileManager)
        let outputURL = try output.map { try Self.resolvedOutputURL(from: $0) }
        let result = try await Self.alignmentResult(
            audioURL: audioURL,
            transcript: transcript,
            modelID: ModelIdentifier(model),
            languageHint: language,
            runtime: runtime
        )
        let renderedAlignment = try Self.renderAlignmentJSON(result)

        if let outputURL {
            try fileManager.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try renderedAlignment.write(to: outputURL, atomically: true, encoding: .utf8)
        }

        if OutputContext.jsonRequested {
            let message = outputURL == nil
                ? "Aligned transcript for \(audioURL.lastPathComponent)."
                : "Aligned transcript for \(audioURL.lastPathComponent) and wrote JSON to \(outputURL!.path)."
            try OutputFormat.writeSuccess(
                command: OutputFormat.commandPath("align"),
                data: AlignmentPayloadDTO(
                    message: message,
                    modelID: model,
                    audioPath: audioURL.path,
                    outputPath: outputURL?.path,
                    transcript: result.transcript,
                    tokens: result.tokens.map(AlignmentTokenDTO.init(from:))
                )
            )
            return
        }

        if let outputURL {
            print("Wrote alignment JSON to \(outputURL.path)")
        } else {
            print(renderedAlignment)
        }
    }
}

extension AlignCommand {
    static func resolvedTranscript(
        from rawTranscript: String,
        fileManager: FileManager = .default
    ) throws -> String {
        let trimmedTranscript = rawTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedTranscript.isEmpty == false else {
            throw ValidationError("Transcript must not be empty.")
        }

        guard trimmedTranscript.hasPrefix("@") else {
            return trimmedTranscript
        }

        let filePath = String(trimmedTranscript.dropFirst()).trimmingCharacters(in: .whitespacesAndNewlines)
        guard filePath.isEmpty == false else {
            throw ValidationError("Transcript file path must not be empty.")
        }

        let transcriptURL = resolvedURL(from: filePath)
        guard fileManager.fileExists(atPath: transcriptURL.path) else {
            throw ValidationError("Transcript file was not found at \(transcriptURL.path).")
        }

        let fileContents = try String(contentsOf: transcriptURL, encoding: .utf8)
        let transcript = fileContents.trimmingCharacters(in: .whitespacesAndNewlines)
        guard transcript.isEmpty == false else {
            throw ValidationError("Transcript file \(transcriptURL.path) was empty.")
        }

        return transcript
    }

    static func renderAlignmentJSON(_ result: ForcedAlignmentResponse) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(result), as: UTF8.self)
    }

    private static func alignmentResult(
        audioURL: URL,
        transcript: String,
        modelID: ModelIdentifier,
        languageHint: String?,
        runtime: ValarRuntime
    ) async throws -> ForcedAlignmentResponse {
        let audioData = try Data(contentsOf: audioURL)
        let decoded = try await runtime.audioPipeline.decode(audioData, hint: audioURL.pathExtension)
        let monoSamples = mixToMono(decoded)
        guard monoSamples.isEmpty == false else {
            throw CLICommandError(message: "Audio file \(audioURL.path) did not contain decodable samples.")
        }

        let request = ForcedAlignmentRequest(
            model: modelID,
            audio: AudioChunk(samples: monoSamples, sampleRate: decoded.format.sampleRate),
            transcript: transcript,
            languageHint: languageHint
        )
        return try await runtime.align(request)
    }

    private static func resolvedInputURL(
        from rawPath: String,
        fileManager: FileManager = .default
    ) throws -> URL {
        let resolvedURL = resolvedURL(from: rawPath)
        guard fileManager.fileExists(atPath: resolvedURL.path) else {
            throw ValidationError("Audio file was not found at \(resolvedURL.path).")
        }
        return resolvedURL
    }

    private static func resolvedOutputURL(from rawPath: String) throws -> URL {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedPath.isEmpty == false else {
            throw ValidationError("Output path must not be empty.")
        }
        return resolvedURL(from: trimmedPath)
    }

    private static func resolvedURL(from rawPath: String) -> URL {
        let currentDirectory = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
        return URL(fileURLWithPath: rawPath, relativeTo: currentDirectory).standardizedFileURL
    }

    private static func mixToMono(_ buffer: AudioPCMBuffer) -> [Float] {
        guard buffer.channels.isEmpty == false else {
            return []
        }

        if buffer.channels.count == 1 {
            return buffer.channels[0]
        }

        let frameCount = buffer.frameCount
        guard frameCount > 0 else {
            return []
        }

        var mono = [Float](repeating: 0, count: frameCount)
        for frameIndex in 0 ..< frameCount {
            var sum: Float = 0
            var contributingChannels = 0

            for channel in buffer.channels where frameIndex < channel.count {
                sum += channel[frameIndex]
                contributingChannels += 1
            }

            if contributingChannels > 0 {
                mono[frameIndex] = sum / Float(contributingChannels)
            }
        }

        return mono
    }
}
