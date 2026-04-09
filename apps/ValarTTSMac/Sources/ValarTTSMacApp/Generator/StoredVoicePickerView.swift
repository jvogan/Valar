import SwiftUI
import ValarModelKit
import ValarPersistence

struct StoredVoicePickerView: View {
    let voices: [VoiceLibraryRecord]
    let selectedModelID: ModelIdentifier?
    let selectedFamilyID: ModelFamilyID?
    @Binding var selectedVoiceID: UUID?

    init(
        voices: [VoiceLibraryRecord],
        selectedModelID: ModelIdentifier?,
        selectedFamilyID: ModelFamilyID? = nil,
        selectedVoiceID: Binding<UUID?>
    ) {
        self.voices = voices
        self.selectedModelID = selectedModelID
        self.selectedFamilyID = selectedFamilyID
        self._selectedVoiceID = selectedVoiceID
    }

    var body: some View {
        let partition = Self.partitionVoices(voices, selectedModelID: selectedModelID)
        let useLanguageGrouping = selectedFamilyID == .vibevoiceRealtimeTTS
        let visiblePresetVoices = useLanguageGrouping
            ? partition.preset.filter { Self.releaseVisibleVibeVoicePreset($0) }
            : partition.preset
        Picker("Voice", selection: $selectedVoiceID) {
            Text("Default voice").tag(nil as UUID?)
            if !visiblePresetVoices.isEmpty {
                if useLanguageGrouping {
                    ForEach(Self.languageGrouped(visiblePresetVoices), id: \.language) { group in
                        Section(group.displayLabel) {
                            ForEach(group.voices) { voice in
                                Text(voice.label).tag(voice.id as UUID?)
                            }
                        }
                    }
                } else {
                    Section("Preset Voices") {
                        ForEach(partition.preset) { voice in
                            Text(voice.label).tag(voice.id as UUID?)
                        }
                    }
                }
            }
            if !partition.saved.isEmpty {
                Section("Saved Voices") {
                    ForEach(partition.saved) { voice in
                        Text(voice.label).tag(voice.id as UUID?)
                    }
                }
            }
        }
        .frame(width: 180)
        .help("Optional preset or saved voice")
    }

    // MARK: - Language Grouping

    private struct LanguageGroup {
        let language: String
        let displayLabel: String
        let voices: [VoiceLibraryRecord]
    }

    private static func languageGrouped(_ voices: [VoiceLibraryRecord]) -> [LanguageGroup] {
        var groups: [String: [VoiceLibraryRecord]] = [:]
        for voice in voices {
            let code = inferredLanguageCode(for: voice)
            groups[code, default: []].append(voice)
        }
        return groups
            .sorted { $0.key < $1.key }
            .map { LanguageGroup(language: $0.key, displayLabel: languageDisplayName($0.key), voices: $0.value) }
    }

    private static func inferredLanguageCode(for voice: VoiceLibraryRecord) -> String {
        guard let backendID = voice.backendVoiceID, !backendID.isEmpty else {
            return "en"
        }
        let normalized = backendID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let prefix = normalized.split(separator: "-", maxSplits: 1).first.map(String.init) ?? normalized
        switch prefix {
        case "en", "a", "b":
            return "en"
        case "de":
            return "de"
        case "fr":
            return "fr"
        case "sp", "es":
            return "es"
        case "it":
            return "it"
        case "jp", "ja":
            return "ja"
        case "kr", "ko":
            return "ko"
        case "nl":
            return "nl"
        case "pl":
            return "pl"
        case "pt":
            return "pt"
        case "in", "hi":
            return "hi"
        default:
            return "en"
        }
    }

    private static func languageDisplayName(_ code: String) -> String {
        switch code {
        case "en": return "English"
        case "fr": return "French"
        case "de": return "German"
        case "es": return "Spanish"
        case "it": return "Italian"
        case "ja": return "Japanese"
        case "ko": return "Korean"
        case "nl": return "Dutch"
        case "pl": return "Polish"
        case "pt": return "Portuguese"
        case "hi": return "Hindi"
        default: return Locale.current.localizedString(forLanguageCode: code) ?? code.uppercased()
        }
    }

    private static func releaseVisibleVibeVoicePreset(_ voice: VoiceLibraryRecord) -> Bool {
        guard let identifier = voice.backendVoiceID ?? nonEmptyLabel(for: voice) else {
            return true
        }
        return VibeVoiceCatalog.isReleaseVisiblePreset(identifier)
    }

    private static func nonEmptyLabel(for voice: VoiceLibraryRecord) -> String? {
        let trimmed = voice.label.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // MARK: - Partitioning

    static func partitionVoices(
        _ voices: [VoiceLibraryRecord],
        selectedModelID: ModelIdentifier?
    ) -> (preset: [VoiceLibraryRecord], saved: [VoiceLibraryRecord]) {
        let presetVoices = voices.filter { voice in
            guard voice.isModelDeclaredPreset else { return false }
            guard let selectedModelID else { return true }
            return voiceMatchesSelectedModel(voice, selectedModelID: selectedModelID)
        }
        let savedVoices = voices.filter { voice in
            guard !voice.isModelDeclaredPreset else { return false }
            guard let selectedModelID else { return true }
            return voiceMatchesSelectedModel(voice, selectedModelID: selectedModelID)
        }
        return (preset: presetVoices, saved: savedVoices)
    }

    private static func voiceMatchesSelectedModel(
        _ voice: VoiceLibraryRecord,
        selectedModelID: ModelIdentifier
    ) -> Bool {
        let selected = normalizedModelID(selectedModelID.rawValue)
        return normalizedModelID(voice.runtimeModelID) == selected
            || normalizedModelID(voice.modelID) == selected
    }

    private static func normalizedModelID(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else {
            return nil
        }
        return trimmed
    }
}
