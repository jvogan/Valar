import ArgumentParser
import Foundation
import ValarModelKit
import XCTest
@testable import valartts

final class OutputFormatTests: XCTestCase {
    private struct ModelStub: Codable, Sendable, Equatable {
        let id: String
    }

    private struct ModelListStubPayload: Codable, Sendable, Equatable {
        let message: String
        let models: [ModelStub]
    }

    private struct UsageError: LocalizedError, CLIExitCodeProviding {
        let cliExitCode = CLIExitCode.usageError

        var errorDescription: String? {
            "bad input"
        }
    }

    private struct RuntimeError: LocalizedError, CLIExitCodeProviding {
        let cliExitCode = CLIExitCode.failure

        var errorDescription: String? {
            "runtime failure"
        }
    }

    func testJSONFlagParsesAcrossCommandTree() throws {
        let validUUID = "00000000-0000-0000-0000-000000000001"
        let argumentSets = [
            ["--json"],
            ["models", "--json"],
            ["models", "list", "--json"],
            ["models", "install", "demo-model", "--json"],
            ["models", "remove", "demo-model", "--json"],
            ["models", "status", "--json"],
            ["voices", "--json"],
            ["voices", "list", "--json"],
            ["voices", "audition", validUUID, "--json"],
            ["speak", "hello", "--model", "demo-model", "--voice", validUUID, "--output", "/tmp/out.wav", "--json"],
            ["transcribe", "--json"],
            ["transcribe", "/tmp/input.wav", "--json"],
            ["transcribe", "--chapter", validUUID, "--json"],
            ["transcribe", "/tmp/input.wav", "--model", "demo-model", "--output", "/tmp/out.txt", "--json"],
            ["transcribe", "/tmp/input.wav", "--chapter", validUUID, "--json"],
            ["projects", "--json"],
            ["projects", "new", "--name", "Demo", "--path", "/tmp/demo", "--json"],
            ["projects", "open", "/tmp/demo.valarproject", "--json"],
            ["projects", "save", "--json"],
            ["projects", "info", "--json"],
            ["projects", "close", "--json"],
            ["chapters", "--json"],
            ["chapters", "list", "--json"],
            ["chapters", "add", "--title", "Intro", "--text", "Hello", "--json"],
            ["chapters", "update", validUUID, "--title", "Updated", "--json"],
            ["chapters", "attach-audio", validUUID, "--audio", "/tmp/recording.wav", "--json"],
            ["chapters", "remove", validUUID, "--json"],
            ["renders", "--json"],
            ["renders", "queue", "--json"],
            ["renders", "start", "--json"],
            ["renders", "status", "--json"],
            ["renders", "cancel", validUUID, "--json"],
            ["exports", "--json"],
            ["exports", "list", "--json"],
            ["exports", "create", "--chapter", validUUID, "--json"],
        ]

        for arguments in argumentSets {
            let parsed = OutputFormat.consumeJSONFlag(from: arguments)
            XCTAssertTrue(parsed.jsonRequested, "Expected json flag to be detected: \(arguments)")
            XCTAssertNoThrow(try ValarCLI.parseAsRoot(parsed.arguments), "Failed to parse: \(arguments)")
        }
    }

    func testJSONFlagStopsAtDoubleDash() {
        let parsed = OutputFormat.consumeJSONFlag(from: ["speak", "--json", "--", "--json"])

        XCTAssertEqual(parsed.arguments, ["speak", "--", "--json"])
        XCTAssertTrue(parsed.jsonRequested)
    }

    func testRootInvocationRequestsHelp() async throws {
        var command = try XCTUnwrap(try ValarCLI.parseAsRoot([]) as? ValarCLI)

        do {
            try await command.run()
            XCTFail("Expected root invocation to request help.")
        } catch {
            XCTAssertEqual(OutputFormat.exitCode(for: error), .success)
        }
    }

    func testAsyncContainerInvocationRequestsHelp() async throws {
        var command = try XCTUnwrap(try ValarCLI.parseAsRoot(["models"]) as? ModelsCommand)

        do {
            try await command.run()
            XCTFail("Expected container invocation to request help.")
        } catch {
            XCTAssertEqual(OutputFormat.exitCode(for: error), .success)
        }
    }

    func testSyncContainerInvocationRequestsHelp() throws {
        var command = try XCTUnwrap(try ValarCLI.parseAsRoot(["voices"]) as? VoicesCommand)

        XCTAssertThrowsError(try command.run()) { error in
            XCTAssertEqual(OutputFormat.exitCode(for: error), .success)
        }
    }

    func testHelpSubcommandResolvesTargetCommand() {
        let rootHelp = ValarCLI.resolveCommand(in: ["help"])
        XCTAssertEqual(rootHelp.path, "valartts")
        XCTAssertEqual(rootHelp.type._commandName, "valartts")

        let modelsHelp = ValarCLI.resolveCommand(in: ["help", "models"])
        XCTAssertEqual(modelsHelp.path, "valartts models")
        XCTAssertEqual(modelsHelp.type._commandName, "models")

        let nestedHelp = ValarCLI.resolveCommand(in: ["help", "renders", "status"])
        XCTAssertEqual(nestedHelp.path, "valartts renders status")
        XCTAssertEqual(nestedHelp.type._commandName, "status")

        let transcribe = ValarCLI.resolveCommand(in: ["transcribe", "/tmp/input.wav"])
        XCTAssertEqual(transcribe.path, "valartts transcribe")
        XCTAssertEqual(transcribe.type._commandName, "transcribe")
    }

    func testTranscribeCommandParsesDefaultModelAlias() throws {
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot(["transcribe", "/tmp/input.wav"]) as? TranscribeCommand
        )

        XCTAssertEqual(command.audioFile, "/tmp/input.wav")
        XCTAssertNil(command.chapter)
        XCTAssertEqual(command.model, TranscribeCommand.defaultModelAlias)
        XCTAssertNil(command.output)
    }

    func testTranscribeCommandParsesChapterOnlyMode() throws {
        let validUUID = "00000000-0000-0000-0000-000000000001"
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot([
                "transcribe",
                "--chapter", validUUID,
            ]) as? TranscribeCommand
        )

        XCTAssertNil(command.audioFile)
        XCTAssertEqual(command.chapter, validUUID)
        XCTAssertEqual(command.model, TranscribeCommand.defaultModelAlias)
        XCTAssertNil(command.output)
    }

    func testTranscribeCommandParsesExplicitOptions() throws {
        let validUUID = "00000000-0000-0000-0000-000000000001"
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot([
                "transcribe",
                "/tmp/input.wav",
                "--chapter", validUUID,
                "--model", "custom-model",
                "--output", "/tmp/output.txt",
            ]) as? TranscribeCommand
        )

        XCTAssertEqual(command.audioFile, "/tmp/input.wav")
        XCTAssertEqual(command.chapter, validUUID)
        XCTAssertEqual(command.model, "custom-model")
        XCTAssertEqual(command.output, "/tmp/output.txt")
    }

    func testSpeakCommandParsesDefaultModelAndOptionalVoice() throws {
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot([
                "speak",
                "hello",
                "--output", "/tmp/out.wav",
            ]) as? SpeakCommand
        )

        XCTAssertEqual(command.inputText, "hello")
        XCTAssertEqual(command.model, SpeakCommand.defaultModelAlias)
        XCTAssertNil(command.voice)
        XCTAssertNil(command.language)
        XCTAssertNil(command.referenceAudio)
        XCTAssertNil(command.referenceTranscript)
        XCTAssertEqual(command.output, "/tmp/out.wav")
    }

    func testRenderSuccessProducesDocumentedSchema() throws {
        let rendered = try OutputFormat.renderSuccess(
            command: OutputFormat.commandPath("models list"),
            data: ModelListStubPayload(
                message: "Loaded 1 supported model(s).",
                models: [ModelStub(id: "demo-model")]
            )
        )

        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["command"] as? String, "valartts models list")

        let data = try XCTUnwrap(payload["data"] as? [String: Any])
        XCTAssertEqual(data["message"] as? String, "Loaded 1 supported model(s).")

        let models = try XCTUnwrap(data["models"] as? [[String: Any]])
        XCTAssertEqual(models.first?["id"] as? String, "demo-model")
    }

    func testSuccessPayloadUsesMessageForCleanExitMessage() throws {
        let rendered = try OutputFormat.renderSuccess(
            command: OutputFormat.commandPath("models install"),
            data: OutputFormat.successPayload(
                for: CleanExit.message("Already installed."),
                command: ModelsCommand.Install.self
            )
        )

        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["command"] as? String, "valartts models install")

        let data = try XCTUnwrap(payload["data"] as? [String: Any])
        XCTAssertEqual(data["message"] as? String, "Already installed.")
        XCTAssertNil(data["help"])
    }

    func testSuccessPayloadUsesRenderedHelpForHelpRequest() throws {
        let rendered = try OutputFormat.renderSuccess(
            command: OutputFormat.commandPath("models"),
            data: OutputFormat.successPayload(
                for: CleanExit.helpRequest(ModelsCommand.self),
                command: ModelsCommand.self
            )
        )

        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["ok"] as? Bool, true)
        XCTAssertEqual(payload["command"] as? String, "valartts models")

        let data = try XCTUnwrap(payload["data"] as? [String: Any])
        let help = try XCTUnwrap(data["help"] as? String)
        XCTAssertTrue(help.contains("USAGE:") && help.contains("models"), "Expected help text with USAGE and models, got: \(help.prefix(100))")
        XCTAssertNil(data["message"])
    }

    func testSuccessPayloadUsesHelpFieldForBuiltInHelpSubcommand() throws {
        var command = try ValarCLI.parseAsRoot(["help", "models"])

        XCTAssertThrowsError(try command.run()) { error in
            let rendered = try? OutputFormat.renderSuccess(
                command: OutputFormat.commandPath("models"),
                data: OutputFormat.successPayload(for: error, command: ModelsCommand.self)
            )

            let payload = try? XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data((rendered ?? "").utf8)) as? [String: Any]
            )
            let data = try? XCTUnwrap(payload?["data"] as? [String: Any])
            let help = data?["help"] as? String

            XCTAssertNotNil(help)
            XCTAssertTrue(help?.contains("USAGE: valartts models <subcommand>") == true)
            XCTAssertNil(data?["message"])
        }
    }

    func testSuccessPayloadUsesHelpFieldForDumpHelpRequest() {
        XCTAssertThrowsError(try ValarCLI.parseAsRoot(["models", "--experimental-dump-help"])) { error in
            let rendered = try? OutputFormat.renderSuccess(
                command: OutputFormat.commandPath("models"),
                data: OutputFormat.successPayload(for: error, command: ModelsCommand.self)
            )

            let payload = try? XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data((rendered ?? "").utf8)) as? [String: Any]
            )
            let data = try? XCTUnwrap(payload?["data"] as? [String: Any])

            // Dump-help output may appear in either 'help' or 'message' depending
            // on whether ArgumentParser's exit matches CleanExit at parse time.
            let content = (data?["help"] as? String) ?? (data?["message"] as? String)
            XCTAssertNotNil(content, "Expected dump-help content in 'help' or 'message' field")
            XCTAssertTrue(content?.contains("command") == true, "Expected JSON command schema in output")
        }
    }

    func testRenderErrorProducesStructuredUsagePayload() throws {
        let rendered = try OutputFormat.renderError(
            command: OutputFormat.commandPath("models install"),
            error: ValidationError("Unknown model."),
            commandType: ModelsCommand.Install.self
        )

        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(payload["command"] as? String, "valartts models install")

        let error = try XCTUnwrap(payload["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? Int, 2)
        XCTAssertEqual(error["kind"] as? String, "usage")
        XCTAssertEqual(error["message"] as? String, "Unknown model.")
        XCTAssertNotNil(error["help"] as? String)
    }

    func testRenderErrorProducesStructuredRuntimePayload() throws {
        let rendered = try OutputFormat.renderError(
            command: OutputFormat.commandPath("voices audition"),
            error: CLICommandError(message: "Failed to play preview audio."),
            commandType: VoicesCommand.Audition.self
        )

        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(payload["ok"] as? Bool, false)
        XCTAssertEqual(payload["command"] as? String, "valartts voices audition")

        let error = try XCTUnwrap(payload["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? Int, 1)
        XCTAssertEqual(error["kind"] as? String, "error")
        XCTAssertEqual(error["message"] as? String, "Failed to play preview audio.")
        XCTAssertNil(error["help"])
    }

    func testRunAsyncPreservesValidationFailures() {
        XCTAssertThrowsError(
            try ProjectsCommand.runAsync {
                throw ValidationError("bad input")
            }
        ) { error in
            XCTAssertEqual((error as? ValidationError)?.message, "bad input")
        }
    }

    func testRunAsyncPreservesCustomUsageFailures() {
        XCTAssertThrowsError(
            try ProjectsCommand.runAsync {
                throw UsageError()
            }
        ) { error in
            XCTAssertEqual((error as? ValidationError)?.message, "bad input")
        }
    }

    func testRunAsyncPreservesRuntimeFailures() {
        XCTAssertThrowsError(
            try ProjectsCommand.runAsync {
                throw CLICommandError(message: "runtime failure")
            }
        ) { error in
            XCTAssertEqual((error as? CLICommandError)?.message, "runtime failure")
        }
    }

    func testRunAsyncPreservesExitCodes() {
        XCTAssertThrowsError(
            try ProjectsCommand.runAsync {
                throw ExitCode.failure
            }
        ) { error in
            XCTAssertEqual((error as? ExitCode)?.rawValue, ExitCode.failure.rawValue)
        }
    }

    func testPlaybackFailureUsesRuntimeExitCode() {
        XCTAssertEqual(
            OutputFormat.exitCode(for: CLICommandError(message: "Failed to play preview audio.")),
            .failure
        )
    }

    func testCustomUsageErrorUsesUsageExitCode() {
        XCTAssertEqual(OutputFormat.exitCode(for: UsageError()), .usageError)
    }

    func testCustomRuntimeErrorUsesFailureExitCode() {
        XCTAssertEqual(OutputFormat.exitCode(for: RuntimeError()), .failure)
    }

    // MARK: - TranscriptFormat

    private func makeRichResult(
        text: String = "Hello world.",
        language: String? = "en",
        durationSeconds: Double? = 3.0,
        segments: [TranscriptionSegment] = []
    ) -> RichTranscriptionResult {
        RichTranscriptionResult(
            text: text,
            language: language,
            durationSeconds: durationSeconds,
            segments: segments,
            backendMetadata: BackendMetadata(modelId: "test-model", backendKind: .mock)
        )
    }

    func testTranscribeCommandParsesFormatFlag() throws {
        let cases: [(String, TranscriptFormat)] = [
            ("text", .text),
            ("json", .json),
            ("verbose_json", .verbose_json),
            ("srt", .srt),
            ("vtt", .vtt),
        ]
        for (rawValue, expected) in cases {
            let command = try XCTUnwrap(
                try ValarCLI.parseAsRoot(["transcribe", "/tmp/input.wav", "--format", rawValue]) as? TranscribeCommand
            )
            XCTAssertEqual(command.format, expected, "Failed for format: \(rawValue)")
        }
    }

    func testTranscribeCommandDefaultFormatIsText() throws {
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot(["transcribe", "/tmp/input.wav"]) as? TranscribeCommand
        )
        XCTAssertEqual(command.format, .text)
    }

    func testTranscriptFormatRenderText() throws {
        let result = makeRichResult(text: "Hello world.")
        XCTAssertEqual(try TranscriptFormat.text.render(result), "Hello world.")
    }

    func testTranscriptFormatRenderJSON() throws {
        let result = makeRichResult(text: "Hello world.", segments: [
            TranscriptionSegment(text: "Hello world.", startTime: 0, endTime: 3, confidence: nil, isFinal: true),
        ])
        let rendered = try TranscriptFormat.json.render(result)
        let parsed = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(parsed["text"] as? String, "Hello world.")
        let segments = try XCTUnwrap(parsed["segments"] as? [String])
        XCTAssertEqual(segments, ["Hello world."])
    }

    func testTranscriptFormatRenderVerboseJSON() throws {
        let result = makeRichResult(text: "Hello world.", language: "en", durationSeconds: 3.0)
        let rendered = try TranscriptFormat.verbose_json.render(result)
        let parsed = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any]
        )
        XCTAssertEqual(parsed["text"] as? String, "Hello world.")
        XCTAssertEqual(parsed["language"] as? String, "en")
        XCTAssertEqual(parsed["durationSeconds"] as? Double, 3.0)
    }

    func testTranscriptFormatRenderSRT() throws {
        let segment = TranscriptionSegment(text: "Hello.", startTime: 0, endTime: 2, confidence: nil, isFinal: true)
        let result = makeRichResult(text: "Hello.", segments: [segment])
        let rendered = try TranscriptFormat.srt.render(result)
        XCTAssertTrue(rendered.contains("1\n"), "Expected SRT sequence number")
        XCTAssertTrue(rendered.contains("-->"), "Expected SRT timestamp arrow")
        XCTAssertTrue(rendered.contains("Hello."), "Expected segment text")
    }

    func testTranscriptFormatRenderVTT() throws {
        let segment = TranscriptionSegment(text: "Hello.", startTime: 0, endTime: 2, confidence: nil, isFinal: true)
        let result = makeRichResult(text: "Hello.", segments: [segment])
        let rendered = try TranscriptFormat.vtt.render(result)
        XCTAssertTrue(rendered.hasPrefix("WEBVTT"), "Expected WebVTT header")
        XCTAssertTrue(rendered.contains("-->"), "Expected VTT timestamp arrow")
        XCTAssertTrue(rendered.contains("Hello."), "Expected segment text")
    }

    func testTranscriptFormatResolvesFromExtension() {
        XCTAssertEqual(TranscriptFormat.text.resolving(fromFileExtension: "srt"), .srt)
        XCTAssertEqual(TranscriptFormat.text.resolving(fromFileExtension: "vtt"), .vtt)
        XCTAssertEqual(TranscriptFormat.text.resolving(fromFileExtension: "txt"), .text)
        XCTAssertEqual(TranscriptFormat.text.resolving(fromFileExtension: nil), .text)
        // Non-text formats are never overridden by extension
        XCTAssertEqual(TranscriptFormat.srt.resolving(fromFileExtension: "vtt"), .srt)
        XCTAssertEqual(TranscriptFormat.json.resolving(fromFileExtension: "srt"), .json)
        XCTAssertEqual(TranscriptFormat.verbose_json.resolving(fromFileExtension: "vtt"), .verbose_json)
    }

    func testTranscribeFormatAndJSONFlagAreIndependentAxes() throws {
        let parsed = OutputFormat.consumeJSONFlag(from: ["transcribe", "/tmp/in.wav", "--format", "verbose_json", "--json"])
        XCTAssertTrue(parsed.jsonRequested)
        let command = try XCTUnwrap(try ValarCLI.parseAsRoot(parsed.arguments) as? TranscribeCommand)
        XCTAssertEqual(command.format, .verbose_json)
    }
}
