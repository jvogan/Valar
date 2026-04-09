import Foundation

func mlxRuntimeReadyForCurrentProcess(fileManager: FileManager = .default) -> Bool {
    let binaryURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    let binaryDir = binaryURL.deletingLastPathComponent()
    let candidates = [
        binaryDir.appendingPathComponent("default.metallib"),
        binaryDir.appendingPathComponent("mlx.metallib"),
    ]
    return candidates.contains { fileManager.fileExists(atPath: $0.path) }
}
