import Foundation
import ValarModelKit
import ValarPersistence

public extension VoiceLibraryRecord {
    private struct QwenClonePromptPayloadProbe: Decodable {
        let version: Int
        let refSpeakerEmbedding: Data?
        let refCode: Data?
        let numCodeGroups: Int?
        let frameCount: Int?
        let xVectorOnlyMode: Bool
        let iclMode: Bool
    }

    var preferredRuntimeModelIdentifier: ModelIdentifier? {
        if resolvedVoiceKind == .legacyPrompt {
            return ValarRuntime.defaultVoiceDesignModelID
        }

        if let runtimeModelIdentifier = normalizedModelIdentifier(runtimeModelID) {
            return runtimeModelIdentifier
        }

        if let sourceModelIdentifier = normalizedModelIdentifier(modelID) {
            if resolvedVoiceKind == .clonePrompt {
                return sourceModelIdentifier
            }
            return sourceModelIdentifier
        }

        if resolvedVoiceKind == .clonePrompt {
            return ValarRuntime.defaultVoiceCloneRuntimeModelID
        }

        return nil
    }

    var preferredRuntimeModelID: String? {
        preferredRuntimeModelIdentifier?.rawValue
    }

    var resolvedVoiceKind: VoiceKind? {
        if let stored = voiceKind.flatMap(VoiceKind.init(rawValue:)) {
            if stored == .preset, isModelDeclaredPreset == false, backendVoiceID != nil {
                return .namedSpeaker
            }
            return stored
        }

        if isModelDeclaredPreset {
            return .preset
        }

        if let effective = effectiveVoiceKind.flatMap(VoiceKind.init(rawValue:)) {
            if effective == .preset, backendVoiceID != nil, isModelDeclaredPreset == false {
                return .namedSpeaker
            }
            return effective
        }

        return nil
    }

    var isLegacyExpressiveVoice: Bool {
        resolvedVoiceKind == .legacyPrompt
    }

    var hasReusableQwenClonePrompt: Bool {
        guard conditioningFormat == Self.qwenClonePromptConditioningFormat,
              let payload = speakerEmbedding,
              !payload.isEmpty,
              let decoded = try? JSONDecoder().decode(QwenClonePromptPayloadProbe.self, from: payload)
        else {
            return false
        }

        if decoded.iclMode {
            return decoded.refCode != nil
                && decoded.numCodeGroups != nil
                && decoded.frameCount != nil
        }
        return decoded.refSpeakerEmbedding != nil
    }

    var preferredVoiceBehavior: SpeechSynthesisVoiceBehavior {
        switch resolvedVoiceKind {
        case .clonePrompt, .embeddingOnly, .tadaReference:
            return .stableNarrator
        case .legacyPrompt, .namedSpeaker:
            return .expressive
        case .preset, .none:
            return .auto
        }
    }

    func makeVoiceProfile(localeIdentifier: String? = nil) -> VoiceProfile {
        let kind = resolvedVoiceKind
        let runtimeModelIdentifier = preferredRuntimeModelIdentifier
        let sourceModelIdentifier = if kind == .legacyPrompt {
            ValarRuntime.defaultVoiceDesignModelID
        } else if let declaredModelIdentifier = normalizedModelIdentifier(modelID) {
            declaredModelIdentifier
        } else if let runtimeModelIdentifier {
            runtimeModelIdentifier
        } else {
            ValarRuntime.defaultVoiceCreateModelID
        }
        let selectorLabel: String
        if kind == .legacyPrompt {
            selectorLabel = voicePrompt ?? label
        } else {
            selectorLabel = label
        }

        return VoiceProfile(
            id: id,
            label: selectorLabel,
            backendVoiceID: backendVoiceID,
            sourceModel: sourceModelIdentifier,
            localeIdentifier: localeIdentifier,
            runtimeModel: runtimeModelIdentifier,
            referenceAudioAssetName: referenceAudioAssetName,
            referenceTranscript: referenceTranscript,
            speakerEmbedding: speakerEmbedding,
            conditioningFormat: conditioningFormat,
            voiceKind: kind,
            isLegacyExpressive: kind == .legacyPrompt
        )
    }

    private func normalizedModelIdentifier(_ rawValue: String?) -> ModelIdentifier? {
        guard let trimmed = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return ModelIdentifier(trimmed)
    }
}
