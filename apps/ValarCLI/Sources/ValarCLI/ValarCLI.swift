import ArgumentParser
import Darwin
import Foundation
import ValarCore

@main
struct ValarCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "valartts",
        abstract: "Local speech tooling for macOS, powered by Apple Silicon.",
        subcommands: [
            DoctorCommand.self,
            CapabilitiesCommand.self,
            ModelsCommand.self,
            VoicesCommand.self,
            SpeakCommand.self,
            TranscribeCommand.self,
            AlignCommand.self,
            ProjectsCommand.self,
            ChaptersCommand.self,
            RendersCommand.self,
            ExportsCommand.self,
        ]
    )

    mutating func run() async throws {
        throw CleanExit.helpRequest(Self.self)
    }

    static func main() async {
        // Enable low-latency Metal synchronization for MLX inference.
        // Opt-out: set VALAR_METAL_FAST_SYNCH=0 in the environment.
        if ProcessInfo.processInfo.environment["VALAR_METAL_FAST_SYNCH"] != "0" {
            setenv("MLX_METAL_FAST_SYNCH", "1", 1)
        }
        let invocation = OutputFormat.consumeJSONFlag(from: Array(CommandLine.arguments.dropFirst()))
        let resolvedCommand = resolveCommand(in: invocation.arguments)

        await OutputContext.$jsonRequested.withValue(invocation.jsonRequested) {
            do {
                var command = try parseAsRoot(invocation.arguments)
                do {
                    if var asyncCommand = command as? AsyncParsableCommand {
                        try await asyncCommand.run()
                    } else {
                        try command.run()
                    }
                } catch {
                    let outputCommand = OutputFormat.exitCode(for: error) == .success
                        ? resolvedCommand
                        : ResolvedCommand(type: type(of: command), path: resolvedCommand.path)
                    terminate(
                        for: error,
                        jsonRequested: invocation.jsonRequested,
                        commandType: outputCommand.type,
                        commandName: outputCommand.path
                    )
                }
            } catch {
                terminate(
                    for: error,
                    jsonRequested: invocation.jsonRequested,
                    commandType: resolvedCommand.type,
                    commandName: resolvedCommand.path
                )
            }
        }
    }

    private static func terminate(
        for error: Error,
        jsonRequested: Bool,
        commandType: ParsableCommand.Type,
        commandName: String
    ) -> Never {
        let exitCode = OutputFormat.exitCode(for: error)

        if jsonRequested {
            do {
                if exitCode == .success {
                    try OutputFormat.writeSuccess(
                        command: commandName,
                        data: OutputFormat.successPayload(for: error, command: commandType)
                    )
                } else {
                    try OutputFormat.writeError(
                        command: commandName,
                        error: error,
                        commandType: commandType
                    )
                }
            } catch {
                let fallback = OutputFormat.message(for: error)
                let fallbackEnvelope = ValarCommandEnvelope<ValarCommandSuccessPayloadDTO>(
                    ok: false,
                    command: commandName,
                    data: nil,
                    error: ValarCommandErrorDTO(
                        code: Int(CLIExitCode.failure.rawValue),
                        kind: "error",
                        message: fallback
                    )
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                if let data = try? encoder.encode(fallbackEnvelope),
                   let text = String(data: data, encoding: .utf8) {
                    print(text)
                }
                Darwin.exit(CLIExitCode.failure.rawValue)
            }
        } else {
            if exitCode == .success {
                print(commandType.message(for: error))
            } else {
                let message = OutputFormat.message(for: error)
                let errorText = exitCode == .usageError
                    ? "Error: \(message)\n\n\(usageString(for: commandType))"
                    : "Error: \(message)"
                FileHandle.standardError.write(Data((errorText + "\n").utf8))
            }
        }

        Darwin.exit(exitCode.rawValue)
    }

    static func resolveCommand(in arguments: [String]) -> ResolvedCommand {
        var current = ResolvedCommand(type: Self.self, path: configuration.commandName ?? Self._commandName)
        var remainingArguments = arguments[...]

        if remainingArguments.first == "help" {
            remainingArguments = remainingArguments.dropFirst()
        }

        for argument in remainingArguments {
            if argument == "--" || argument.hasPrefix("-") {
                break
            }

            guard let subcommand = current.type.configuration.subcommands.first(where: { subcommand in
                subcommand._commandName == argument || subcommand.configuration.aliases.contains(argument)
            }) else {
                break
            }

            current = ResolvedCommand(
                type: subcommand,
                path: "\(current.path) \(subcommand._commandName)"
            )
        }

        return current
    }
}

struct ResolvedCommand {
    let type: ParsableCommand.Type
    let path: String
}
