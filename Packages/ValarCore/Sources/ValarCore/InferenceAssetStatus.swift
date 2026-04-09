import Foundation

public struct LocalInferenceAssetsStatus: Codable, Sendable, Equatable {
    public let metallibAvailable: Bool
    public let candidatePaths: [String]
    public let binaryName: String

    public init(
        metallibAvailable: Bool,
        candidatePaths: [String],
        binaryName: String
    ) {
        self.metallibAvailable = metallibAvailable
        self.candidatePaths = candidatePaths
        self.binaryName = binaryName
    }

    public var failureReason: String {
        let candidateList = candidatePaths.joined(separator: ", ")
        let binaryDir = URL(fileURLWithPath: candidatePaths.first ?? CommandLine.arguments[0])
            .deletingLastPathComponent()
            .path
        return "Local MLX inference assets are missing: expected default.metallib or mlx.metallib next to \(binaryName). Checked \(candidateList). Run: bash scripts/build_metallib.sh \(binaryDir)"
    }

    public static func currentProcess(
        arguments: [String] = CommandLine.arguments,
        fileManager: FileManager = .default
    ) -> LocalInferenceAssetsStatus {
        let binaryURL = URL(fileURLWithPath: arguments.first ?? CommandLine.arguments[0]).standardizedFileURL
        let binaryDir = binaryURL.deletingLastPathComponent()
        let candidates = [
            binaryDir.appendingPathComponent("default.metallib").path,
            binaryDir.appendingPathComponent("mlx.metallib").path,
        ]
        return LocalInferenceAssetsStatus(
            metallibAvailable: candidates.contains { fileManager.fileExists(atPath: $0) },
            candidatePaths: candidates,
            binaryName: binaryURL.lastPathComponent
        )
    }
}
