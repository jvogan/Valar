import Foundation

public enum SupportedModelCatalog {
    private static let publicQwenIDs: Set<ModelIdentifier> = [
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "mlx-community/Qwen3-ASR-0.6B-8bit",
        "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    ]

    private static let publicSopranoIDs: Set<ModelIdentifier> = [
        "mlx-community/Soprano-1.1-80M-bf16",
    ]

    private static let publicVibeVoiceIDs: Set<ModelIdentifier> = [
        VibeVoiceCatalog.mlx4BitModelIdentifier,
    ]

    private static let publicVoxtralIDs: Set<ModelIdentifier> = [
        VoxtralCatalog.mlx4BitModelIdentifier,
    ]

    public static var allSupportedEntries: [SupportedModelCatalogEntry] {
        let qwen = QwenCatalog.supportedEntries.filter { publicQwenIDs.contains($0.id) }
        let soprano = SopranoCatalog.supportedEntries.filter { publicSopranoIDs.contains($0.id) }
        let vibeVoice = VibeVoiceCatalog.supportedEntries.filter { publicVibeVoiceIDs.contains($0.id) }
        let voxtral = VoxtralCatalog.supportedEntries.filter { publicVoxtralIDs.contains($0.id) }
        return qwen + soprano + vibeVoice + voxtral
    }

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        allSupportedEntries
    }

    public static func curatedEntries(
        includeNonCommercial: Bool = includesNonCommercialModels()
    ) -> [SupportedModelCatalogEntry] {
        guard includeNonCommercial == false else { return allSupportedEntries }
        return allSupportedEntries.filter { !$0.manifest.licenses.contains(where: \.isNonCommercial) }
    }

    public static func entry(for identifier: ModelIdentifier) -> SupportedModelCatalogEntry? {
        allSupportedEntries.first(where: { $0.id == identifier })
    }

    public static func includesNonCommercialModels(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let rawValue = environment["VALARTTS_ENABLE_NONCOMMERCIAL_MODELS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            rawValue.isEmpty == false else {
            return false
        }

        switch rawValue {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }
}
