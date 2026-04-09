import ArgumentParser
import Foundation
import ValarCore

struct ParsedCommandArguments: Equatable, Sendable {
    let arguments: [String]
    let jsonRequested: Bool
}

enum OutputContext {
    @TaskLocal
    static var jsonRequested = false
}

enum CLIExitCode: Int32 {
    case success = 0
    case failure = 1
    case usageError = 2
}

protocol CLIExitCodeProviding: Error {
    var cliExitCode: CLIExitCode { get }
}

struct CLICommandError: LocalizedError, Sendable {
    let message: String

    var errorDescription: String? {
        message
    }
}

/// Structured output for `valartts --json`.
///
/// Success schema:
/// {
///   "ok": true,
///   "command": "<command path>",
///   "data": {
///     "message": "<human-readable summary>",
///     ... command-specific fields ...
///   }
/// }
///
/// Error schema:
/// {
///   "ok": false,
///   "command": "<command path>",
///   "error": {
///     "code": 1 | 2,
///     "kind": "error" | "usage",
///     "message": "<error summary>",
///     "help": "<optional usage/help text>"
///   }
/// }
///
/// Help requests are treated as successful output:
/// {
///   "ok": true,
///   "command": "<command path>",
///   "data": {
///     "help": "<full help text>"
///   }
/// }
enum OutputFormat {
    static func commandPath(_ components: String...) -> String {
        let suffix = components
            .flatMap { $0.split(separator: " ") }
            .map(String.init)
            .filter { $0.isEmpty == false }

        let root = ValarCLI.configuration.commandName ?? ValarCLI._commandName
        guard suffix.isEmpty == false else {
            return root
        }

        return ([root] + suffix).joined(separator: " ")
    }

    static func consumeJSONFlag(from arguments: [String]) -> ParsedCommandArguments {
        var sanitized: [String] = []
        var jsonRequested = false
        var passthrough = false

        for argument in arguments {
            if passthrough {
                sanitized.append(argument)
                continue
            }

            if argument == "--" {
                passthrough = true
                sanitized.append(argument)
                continue
            }

            if argument == "--json" {
                jsonRequested = true
                continue
            }

            sanitized.append(argument)
        }

        return ParsedCommandArguments(arguments: sanitized, jsonRequested: jsonRequested)
    }

    static func exitCode(for error: Error) -> CLIExitCode {
        if let classifiedError = error as? CLIExitCodeProviding {
            return classifiedError.cliExitCode
        }

        let rawValue = ValarCLI.exitCode(for: error).rawValue

        switch rawValue {
        case 0:
            return .success
        case ExitCode.validationFailure.rawValue:
            return .usageError
        default:
            return .failure
        }
    }

    static func message(for error: Error) -> String {
        if let localizedError = error as? LocalizedError,
           let description = localizedError.errorDescription?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           description.isEmpty == false {
            return description
        }

        let description = String(describing: error).trimmingCharacters(in: .whitespacesAndNewlines)
        if description.isEmpty == false {
            return description
        }

        return error.localizedDescription
    }

    static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    static func helpPayload(for command: ParsableCommand.Type) -> ValarCommandSuccessPayloadDTO {
        ValarCommandSuccessPayloadDTO(help: ValarCLI.helpMessage(for: command))
    }

    static func successPayload(
        for error: Error,
        command: ParsableCommand.Type
    ) -> ValarCommandSuccessPayloadDTO {
        let renderedMessage = command.message(for: error)
        let helpMessage = ValarCLI.helpMessage(for: command)
        let dumpHelpMessage = command._dumpHelp()

        if renderedMessage == helpMessage || renderedMessage == dumpHelpMessage || isHelpExit(error) {
            return ValarCommandSuccessPayloadDTO(help: renderedMessage)
        }

        return ValarCommandSuccessPayloadDTO(message: renderedMessage)
    }

    static func errorPayload(
        for error: Error,
        command: ParsableCommand.Type
    ) -> ValarCommandErrorDTO {
        let exitCode = exitCode(for: error)
        let helpText = exitCode == .usageError ? ValarCLI.usageString(for: command) : nil
        return ValarCommandErrorDTO(
            code: Int(exitCode.rawValue),
            kind: exitCode == .usageError ? "usage" : "error",
            message: message(for: error),
            help: helpText
        )
    }

    static func renderSuccess<Payload: Codable & Sendable>(
        command: String,
        data: Payload
    ) throws -> String {
        try render(
            ValarCommandEnvelope(
                ok: true,
                command: command,
                data: data,
                error: nil
            )
        )
    }

    static func renderHelp(
        command: String,
        commandType: ParsableCommand.Type
    ) throws -> String {
        try renderSuccess(command: command, data: helpPayload(for: commandType))
    }

    static func renderError(
        command: String,
        error: Error,
        commandType: ParsableCommand.Type
    ) throws -> String {
        try render(
            ValarCommandEnvelope<ValarCommandSuccessPayloadDTO>(
                ok: false,
                command: command,
                data: nil,
                error: errorPayload(for: error, command: commandType)
            )
        )
    }

    static func writeSuccess<Payload: Codable & Sendable>(
        command: String,
        data: Payload
    ) throws {
        print(try renderSuccess(command: command, data: data))
    }

    static func writeHelp(
        command: String,
        commandType: ParsableCommand.Type
    ) throws {
        print(try renderHelp(command: command, commandType: commandType))
    }

    static func writeError(
        command: String,
        error: Error,
        commandType: ParsableCommand.Type
    ) throws {
        print(try renderError(command: command, error: error, commandType: commandType))
    }

    private static func render<Payload: Codable & Sendable>(
        _ envelope: ValarCommandEnvelope<Payload>
    ) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(envelope), as: UTF8.self)
    }
    private static func isHelpExit(_ error: Error) -> Bool {
        guard let cleanExit = error as? CleanExit else {
            return false
        }
        return cleanExit.description == "--help" || cleanExit.description == "--experimental-dump-help"
    }
}
