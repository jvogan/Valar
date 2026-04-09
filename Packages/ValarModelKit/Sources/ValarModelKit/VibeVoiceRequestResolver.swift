import Foundation

public enum VibeVoiceSelectionMode: String, Codable, Sendable {
    case explicit
    case autoDefault = "auto_default"
    case random
}

public struct VibeVoiceResolvedRequest: Equatable, Codable, Sendable {
    public let effectiveVoice: String
    public let effectiveLanguage: String
    public let selectionMode: VibeVoiceSelectionMode

    public init(
        effectiveVoice: String,
        effectiveLanguage: String,
        selectionMode: VibeVoiceSelectionMode
    ) {
        self.effectiveVoice = effectiveVoice
        self.effectiveLanguage = effectiveLanguage
        self.selectionMode = selectionMode
    }
}

public enum VibeVoiceRequestResolutionError: LocalizedError, Equatable, Sendable {
    case invalidVoice(String)
    case unsupportedLanguage(String)
    case explicitVoiceLanguageMismatch(
        voice: String,
        requestedLanguage: String,
        presetLanguage: String,
        suggestedVoice: String?
    )

    public var errorDescription: String? {
        switch self {
        case .invalidVoice(let voice):
            return "Voice '\(voice)' is not a valid VibeVoice preset, display name, or 'random'."
        case .unsupportedLanguage(let language):
            let supported = VibeVoiceCatalog.supportedLanguageCodes.joined(separator: ", ")
            return "Language '\(language)' is not supported by VibeVoice. Supported values: \(supported)."
        case let .explicitVoiceLanguageMismatch(voice, requestedLanguage, presetLanguage, suggestedVoice):
            let suggestion: String
            if let suggestedVoice {
                suggestion = " Try '\(suggestedVoice)' for language '\(requestedLanguage)'."
            } else {
                suggestion = ""
            }
            return "Voice '\(voice)' is a \(presetLanguage) preset and cannot be used with language '\(requestedLanguage)'.\(suggestion)"
        }
    }
}

public enum VibeVoiceRequestResolver {
    public typealias RandomPicker = @Sendable ([PresetVoiceSpec]) -> PresetVoiceSpec?

    public static func resolve(
        voice: String?,
        language: String?,
        randomPicker: RandomPicker? = nil
    ) throws -> VibeVoiceResolvedRequest {
        let trimmedVoice = voice?.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveVoice = trimmedVoice?.isEmpty == false ? trimmedVoice : nil
        let effectiveLanguage = try normalizedLanguage(from: language)
        let picker = randomPicker ?? { presets in
            presets.randomElement()
        }

        if let effectiveVoice {
            if effectiveVoice.lowercased() == "random" {
                let pool = try randomPool(for: effectiveLanguage)
                guard let chosenPreset = picker(pool) ?? pool.first,
                      let chosenLanguage = VibeVoiceCatalog.primaryLanguage(for: chosenPreset) else {
                    throw VibeVoiceRequestResolutionError.invalidVoice(effectiveVoice)
                }
                return VibeVoiceResolvedRequest(
                    effectiveVoice: chosenPreset.name,
                    effectiveLanguage: effectiveLanguage ?? chosenLanguage,
                    selectionMode: .random
                )
            }

            guard let preset = VibeVoiceCatalog.preset(matching: effectiveVoice),
                  let presetLanguage = VibeVoiceCatalog.primaryLanguage(for: preset) else {
                throw VibeVoiceRequestResolutionError.invalidVoice(effectiveVoice)
            }

            if let effectiveLanguage, effectiveLanguage != presetLanguage {
                let suggestion = VibeVoiceCatalog.defaultPreset(forLanguage: effectiveLanguage)?.name
                    ?? VibeVoiceCatalog.presets(forLanguage: effectiveLanguage).first?.name
                throw VibeVoiceRequestResolutionError.explicitVoiceLanguageMismatch(
                    voice: preset.name,
                    requestedLanguage: effectiveLanguage,
                    presetLanguage: presetLanguage,
                    suggestedVoice: suggestion
                )
            }

            return VibeVoiceResolvedRequest(
                effectiveVoice: preset.name,
                effectiveLanguage: effectiveLanguage ?? presetLanguage,
                selectionMode: .explicit
            )
        }

        let resolvedLanguage = effectiveLanguage ?? "en"
        guard let preset = VibeVoiceCatalog.defaultPreset(forLanguage: resolvedLanguage) else {
            throw VibeVoiceRequestResolutionError.unsupportedLanguage(resolvedLanguage)
        }

        return VibeVoiceResolvedRequest(
            effectiveVoice: preset.name,
            effectiveLanguage: resolvedLanguage,
            selectionMode: .autoDefault
        )
    }

    private static func normalizedLanguage(from language: String?) throws -> String? {
        guard let trimmedLanguage = language?.trimmingCharacters(in: .whitespacesAndNewlines),
              trimmedLanguage.isEmpty == false else {
            return nil
        }
        guard let normalizedLanguage = VibeVoiceCatalog.normalizedLanguageCode(trimmedLanguage) else {
            throw VibeVoiceRequestResolutionError.unsupportedLanguage(trimmedLanguage)
        }
        return normalizedLanguage
    }

    private static func randomPool(for language: String?) throws -> [PresetVoiceSpec] {
        if let language {
            let presets = VibeVoiceCatalog.presets(forLanguage: language)
            guard presets.isEmpty == false else {
                throw VibeVoiceRequestResolutionError.unsupportedLanguage(language)
            }
            return presets
        }
        return VibeVoiceCatalog.presetVoices
    }
}
