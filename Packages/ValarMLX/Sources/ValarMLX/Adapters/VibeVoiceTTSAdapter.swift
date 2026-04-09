import Foundation
import ValarModelKit

public struct VibeVoiceTTSAdapter: ModelAdapter, Sendable {
    public let familyID: ModelFamilyID = .vibevoiceRealtimeTTS
    public let supportedCapabilities: Set<CapabilityID> = [
        .speechSynthesis, .presetVoices, .streaming,
    ]
    public let supportedBackends: [BackendKind] = [.mlx]

    public init() {}

    public func validate(manifest: ModelPackManifest) throws {
        guard manifest.familyID == .vibevoiceRealtimeTTS else {
            throw AdapterError.familyMismatch(expected: .vibevoiceRealtimeTTS, got: manifest.familyID)
        }
        guard manifest.artifacts.contains(where: { $0.role == .config && $0.relativePath == "config.json" }) else {
            throw AdapterError.missingArtifacts("VibeVoice requires config.json")
        }
        guard manifest.artifacts.contains(where: { $0.role == .weights && $0.relativePath == "model.safetensors" }) else {
            throw AdapterError.missingArtifacts("VibeVoice requires model.safetensors")
        }
        guard manifest.artifacts.contains(where: { $0.relativePath == "preprocessor_config.json" }) else {
            throw AdapterError.missingArtifacts("VibeVoice requires preprocessor_config.json")
        }
        guard let tokenizer = manifest.tokenizer,
              tokenizer.kind == "huggingface",
              let tokenizerPath = tokenizer.configPath,
              manifest.artifacts.contains(where: { $0.role == .tokenizer && $0.relativePath == tokenizerPath }) else {
            throw AdapterError.missingArtifacts("VibeVoice requires a Hugging Face tokenizer artifact (tokenizer.json).")
        }
        guard manifest.artifacts.contains(where: { $0.relativePath == "tokenizer_config.json" }) else {
            throw AdapterError.missingArtifacts("VibeVoice requires tokenizer_config.json for tokenizer loading.")
        }
        let voiceArtifacts = manifest.artifacts.filter { $0.role == .voiceAsset }
        guard !voiceArtifacts.isEmpty else {
            throw AdapterError.missingArtifacts("VibeVoice requires at least one voice cache artifact in voices/")
        }
        if let presetVoices = manifest.presetVoices {
            let availableVoicePaths = Set(voiceArtifacts.map(\.relativePath))
            let missingPresetArtifacts = presetVoices.filter {
                !availableVoicePaths.contains("voices/\($0.name).safetensors")
            }
            guard missingPresetArtifacts.isEmpty else {
                let missingNames = missingPresetArtifacts.map(\.name).joined(separator: ", ")
                throw AdapterError.missingArtifacts(
                    "VibeVoice manifest is missing voice cache artifacts for preset voices: \(missingNames)."
                )
            }
        }
    }

    public func makeDescriptor(from manifest: ModelPackManifest) throws -> ModelDescriptor {
        ModelDescriptor(manifest: manifest)
    }
}
