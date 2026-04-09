import Foundation
import Hummingbird
import HummingbirdCore
import ValarCore
import ValarMLX
import ValarModelKit
import ValarPersistence
import Darwin

private let daemonBindHostEnvVar = "VALARTTSD_BIND_HOST"
private let daemonBindPortEnvVar = "VALARTTSD_BIND_PORT"

private func daemonBindHost(environment: [String: String]) throws -> String {
    let host = environment[daemonBindHostEnvVar]?.trimmingCharacters(in: .whitespacesAndNewlines)
    let resolved = (host?.isEmpty == false ? host! : "127.0.0.1")
    guard isLoopbackHost(resolved) else {
        throw RuntimeError("The public daemon only supports loopback binds. Refusing to bind to \(resolved).")
    }
    return resolved
}

private func daemonBindPort(environment: [String: String]) throws -> Int {
    guard let raw = environment[daemonBindPortEnvVar]?.trimmingCharacters(in: .whitespacesAndNewlines),
          !raw.isEmpty else {
        return 8787
    }
    guard let port = Int(raw), (1 ... 65535).contains(port) else {
        throw RuntimeError("Invalid \(daemonBindPortEnvVar) value '\(raw)'. Expected an integer between 1 and 65535.")
    }
    return port
}

private func isLoopbackHost(_ host: String) -> Bool {
    switch host.lowercased() {
    case "127.0.0.1", "::1", "localhost":
        return true
    default:
        return false
    }
}

private struct RuntimeError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}

private func daemonPIDFileURL(paths: ValarAppPaths) -> URL {
    paths.daemonPIDFileURL
}

private func printStartupMaintenance(
    _ report: RuntimeStartupMaintenanceReport
) {
    if !report.modelPackState.removedStaleModelIDs.isEmpty {
        let removed = report.modelPackState.removedStaleModelIDs.map(\.rawValue).joined(separator: ", ")
        print("Removed stale installed model records: \(removed)")
        print("Review current model state with: valartts models status")
    }
    if !report.modelPackState.orphanedModelPackPaths.isEmpty {
        print("Found orphaned ModelPacks not registered in the install ledger:")
        for path in report.modelPackState.orphanedModelPackPaths.sorted() {
            print("  - \(ValarPathRedaction.redact(path))")
        }
        print("Preview cleanup with: valartts models cleanup --dry-run")
        print("Apply cleanup with:   valartts models cleanup --apply")
    }
    if !report.voiceLibrary.upgradedReusableQwenClonePromptVoiceIDs.isEmpty {
        let upgraded = report.voiceLibrary.upgradedReusableQwenClonePromptVoiceIDs
            .map(\.uuidString)
            .joined(separator: ", ")
        print("Upgraded saved Qwen stable narrator voices to reusable clone prompts: \(upgraded)")
    }
}

private func reconcileExistingPIDFile(
    at pidFileURL: URL,
    fileManager: FileManager = .default
) throws {
    guard fileManager.fileExists(atPath: pidFileURL.path) else {
        return
    }
    let rawPID = try String(contentsOf: pidFileURL, encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard let pid = Int32(rawPID), pid > 0 else {
        try? fileManager.removeItem(at: pidFileURL)
        return
    }

    if pid == ProcessInfo.processInfo.processIdentifier {
        return
    }

    if kill(pid, 0) == 0 {
        throw RuntimeError("Another valarttsd is already running (PID: \(pid)). Stop it first: pkill valarttsd")
    }

    try? fileManager.removeItem(at: pidFileURL)
}

// Enable low-latency Metal synchronization for MLX inference.
// Opt-out: set VALAR_METAL_FAST_SYNCH=0 in the environment.
if ProcessInfo.processInfo.environment["VALAR_METAL_FAST_SYNCH"] != "0" {
    setenv("MLX_METAL_FAST_SYNCH", "1", 1)
}

// Singleton guard: refuse to start if another daemon is already running.
do {
    let existing = Process()
    existing.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
    existing.arguments = ["-x", "valarttsd"]
    let pipe = Pipe()
    existing.standardOutput = pipe
    try existing.run()
    existing.waitUntilExit()
    let pids = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .split(separator: "\n")
        .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        .filter { $0 != ProcessInfo.processInfo.processIdentifier } ?? []
    if !pids.isEmpty {
        print("ERROR: Another valarttsd is already running (PIDs: \(pids)). Stop it first: pkill valarttsd")
        exit(1)
    }
}

let startedAt = Date.now
let environment = ProcessInfo.processInfo.environment
let runtimeConfiguration = RuntimeConfiguration.configured(from: environment, defaultWarmPolicy: .eager)
let runtime = try ValarRuntime(runtimeConfiguration: runtimeConfiguration)
let pidFileURL = daemonPIDFileURL(paths: runtime.paths)
try reconcileExistingPIDFile(at: pidFileURL)
try "\(ProcessInfo.processInfo.processIdentifier)\n".write(to: pidFileURL, atomically: true, encoding: .utf8)
defer { try? FileManager.default.removeItem(at: pidFileURL) }
let startupMaintenance = await runtime.ensureStartupMaintenance()
printStartupMaintenance(startupMaintenance)
if runtimeConfiguration.warmPolicy == .eager {
    await runtime.prewarmInstalledModels()
}

// Pre-compile Metal kernels with a short warmup inference.
if runtimeConfiguration.warmPolicy == .eager {
do {
    let warmupModels = await runtime.warmStartCatalogModels()
    if let catalogModel = warmupModels.first(where: {
        $0.descriptor.capabilities.contains(.speechSynthesis)
    }) {
        let descriptor = catalogModel.descriptor
        let policy = BackendSelectionPolicy()
        let backendRuntime = BackendSelectionPolicy.Runtime(
            availableBackends: [runtime.inferenceBackend.backendKind]
        )
        let configuration = try policy.runtimeConfiguration(
            for: descriptor,
            runtime: backendRuntime
        )
        let warmupRequest = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "warmup",
            sampleRate: descriptor.defaultSampleRate ?? 24_000,
            responseFormat: "pcm_f32le"
        )
        _ = try await runtime.withReservedTextToSpeechWorkflowSession(
            descriptor: descriptor,
            configuration: configuration
        ) { reserved in
            try await reserved.workflow.synthesize(
                request: warmupRequest,
                in: reserved.session
            )
        }
        print("Metal kernel warmup complete")
    }
} catch {
    print("Warmup skipped: \(error)")
}
}

let builtRouter = ValarDaemonRouter.build(runtime: runtime, startedAt: startedAt)
defer {
    for task in builtRouter.backgroundTasks {
        task.cancel()
    }
}
let bindHost = try daemonBindHost(environment: environment)
let bindPort = try daemonBindPort(environment: environment)
let configuration = ApplicationConfiguration(
    address: .hostname(bindHost, port: bindPort)
)
let server = HTTPServerBuilder.http1(
    configuration: .init(
        additionalChannelHandlers: [ClientInputCloseHandler()]
    )
)
let app = Application(
    router: builtRouter.router,
    server: server,
    configuration: configuration
)

print("valarttsd listening on http://\(bindHost):\(bindPort)")
try await app.runService()
