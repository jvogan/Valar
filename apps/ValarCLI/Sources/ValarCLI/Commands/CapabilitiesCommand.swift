import ArgumentParser
import Foundation
import Metal
import ValarCore
import ValarModelKit

struct CapabilitiesCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "capabilities",
        abstract: "Show what this machine can do right now."
    )

    mutating func run() async throws {
        let runtime = try ValarRuntime()
        let metallibAvailable = Self.checkMetallibAvailable()
        let daemonBaseURL = Self.daemonBaseURL()

        // Ping daemon health first, then readiness.
        var daemonReachable = false
        var daemonReady = false
        var daemonReadyDTO: DaemonReadyDTO?
        if let healthURL = daemonBaseURL?.appendingPathComponent("v1/health") {
            if let (_, response) = try? await URLSession.shared.data(from: healthURL),
               let http = response as? HTTPURLResponse,
               http.statusCode == 200 {
                daemonReachable = true
            }
        }
        if daemonReachable, let url = daemonBaseURL?.appendingPathComponent("v1/ready") {
            if let (data, response) = try? await URLSession.shared.data(from: url),
               let http = response as? HTTPURLResponse {
                if let dto = try? JSONDecoder().decode(DaemonReadyDTO.self, from: data) {
                    daemonReadyDTO = dto
                    daemonReady = dto.ready
                } else {
                    daemonReady = http.statusCode == 200
                }
            }
        }

        let snapshot = try await runtime.capabilitySnapshot(
            daemonReachable: daemonReachable,
            daemonReady: daemonReady,
            metallibAvailable: metallibAvailable
        )

        if OutputContext.jsonRequested {
            try OutputFormat.writeSuccess(
                command: OutputFormat.commandPath("capabilities"),
                data: snapshot
            )
            return
        }

        printHumanSummary(snapshot, daemonReadyDTO: daemonReadyDTO)
    }

    // MARK: - Metallib detection

    private static func checkMetallibAvailable() -> Bool {
        // Check adjacent to the running binary
        let binaryURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let binaryDir = binaryURL.deletingLastPathComponent()
        let candidates = [
            binaryDir.appendingPathComponent("default.metallib"),
            binaryDir.appendingPathComponent("mlx.metallib"),
        ]
        for url in candidates {
            if FileManager.default.fileExists(atPath: url.path) {
                return true
            }
        }
        return false
    }

    private static func daemonBaseURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        let trimmedHost = environment["VALARTTSD_BIND_HOST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedPort = environment["VALARTTSD_BIND_PORT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let host = (trimmedHost?.isEmpty == false ? trimmedHost : nil) ?? "127.0.0.1"
        let port = (trimmedPort?.isEmpty == false ? trimmedPort : nil) ?? "8787"
        return URL(string: "http://\(host):\(port)")
    }

    // MARK: - Human-readable output

    private func printHumanSummary(
        _ s: CapabilitySnapshotDTO,
        daemonReadyDTO: DaemonReadyDTO?
    ) {
        func status(_ ok: Bool) -> String { ok ? "READY" : "NOT READY" }

        print("Valar Capabilities")
        print("==================")
        print()
        print("Speech synthesis:    \(status(s.canSpeakNow))")
        print("Transcription:       \(status(s.canTranscribeNow))")
        print("Forced alignment:    \(status(s.canAlignNow))")
        print("Voice cloning:       \(status(s.canCloneVoiceNow))")
        print()

        if !s.installedTTSModels.isEmpty {
            print("TTS models:          \(s.installedTTSModels.joined(separator: ", "))")
        }
        if !s.installedASRModels.isEmpty {
            print("ASR models:          \(s.installedASRModels.joined(separator: ", "))")
        }
        if !s.cachedButNotRegistered.isEmpty {
            print("Cached (not registered): \(s.cachedButNotRegistered.joined(separator: ", "))")
            print("Next step:         valartts models install <id>")
        }
        print("Installed voices:    \(s.installedVoiceCount)")
        print("Metal library:       \(s.metallibAvailable ? "found" : "missing")")
        print("Daemon reachable:    \(s.daemonReachable ? "yes" : "no")")
        print("Daemon ready:        \(s.daemonReady ? "yes" : "no")")
        if let dto = daemonReadyDTO {
            if let mode = dto.readinessMode?.rawValue {
                print("Ready mode:          \(mode)")
            }
            if let assetsReady = dto.inferenceAssetsReady {
                print("Inference assets:    \(assetsReady ? "ready" : "missing")")
            }
            if let reason = dto.reason {
                print("Daemon reason:       \(reason)")
            }
        }

        if !s.missingPrerequisites.isEmpty {
            print()
            print("NEEDS:")
            for item in s.missingPrerequisites {
                print("  • \(item)")
            }
        } else {
            print()
            print("All prerequisites met.")
        }
    }
}
