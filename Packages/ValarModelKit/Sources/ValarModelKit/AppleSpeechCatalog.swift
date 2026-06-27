import Foundation

public enum AppleSpeechCatalog {
    public static let ttsModelIdentifier: ModelIdentifier = "apple/system-tts"
    public static let asrModelIdentifier: ModelIdentifier = "apple/system-asr"

    public static var supportedEntries: [SupportedModelCatalogEntry] {
        [
            makeTTSManifest(),
            makeASRManifest(),
        ]
    }

    private static func makeTTSManifest() -> SupportedModelCatalogEntry {
        let manifest = ModelPackManifest(
            id: ttsModelIdentifier,
            familyID: .appleSpeechSynthesis,
            displayName: "Apple System TTS",
            domain: .tts,
            capabilities: [.speechSynthesis, .presetVoices],
            supportedBackends: [
                BackendRequirement(
                    backendKind: .apple,
                    requiresLocalExecution: true
                ),
            ],
            artifacts: [],
            audio: AudioConstraint(
                defaultSampleRate: 24_000,
                minimumSampleRate: 8_000,
                maximumSampleRate: 48_000
            ),
            promptSchema: PromptSchema(
                kind: "apple.tts",
                requiredFields: ["text"],
                optionalFields: ["voice", "language"]
            ),
            notes: "Uses macOS AVSpeechSynthesizer voices. No Valar model pack is downloaded."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            isRecommended: false,
            distributionTier: .bundledFirstRun,
            tags: ["apple", "system", "tts", "local"]
        )
    }

    private static func makeASRManifest() -> SupportedModelCatalogEntry {
        let manifest = ModelPackManifest(
            id: asrModelIdentifier,
            familyID: .appleSpeechRecognition,
            displayName: "Apple System ASR",
            domain: .stt,
            capabilities: [.speechRecognition],
            supportedBackends: [
                BackendRequirement(
                    backendKind: .apple,
                    requiresLocalExecution: true
                ),
            ],
            artifacts: [],
            audio: AudioConstraint(
                defaultSampleRate: 16_000,
                minimumSampleRate: 8_000,
                maximumSampleRate: 48_000
            ),
            promptSchema: PromptSchema(
                kind: "apple.asr",
                requiredFields: ["audio"],
                optionalFields: ["language"]
            ),
            supportTier: .supported,
            notes: "Uses macOS Speech with on-device recognition required. No Valar model pack is downloaded."
        )

        return SupportedModelCatalogEntry(
            manifest: manifest,
            isRecommended: false,
            distributionTier: .bundledFirstRun,
            tags: ["apple", "system", "asr", "local"]
        )
    }
}
