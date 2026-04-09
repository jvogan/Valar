import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

@Observable
@MainActor
final class VoiceLibraryState {
    var voices: [VoiceLibraryRecord] = []
    var availableCloneModels: [RuntimeModelPickerOption] = []
    var searchText = ""
    var selectedVoice: VoiceLibraryRecord?
    var cloneWizardOpen = false
    var designSheetOpen = false
    var pendingCloneAudioURL: URL?
    var isLoading = true
    var errorMessage: String?

    var filteredVoices: [VoiceLibraryRecord] {
        voices.filter(matchesSearch)
    }

    var filteredPresetVoices: [VoiceLibraryRecord] {
        filteredVoices.filter(\.isModelDeclaredPreset)
    }

    var filteredSavedVoices: [VoiceLibraryRecord] {
        filteredVoices.filter { !$0.isModelDeclaredPreset }
    }

    private let services: ValarServiceHub

    init(services: ValarServiceHub) {
        self.services = services
    }

    func apply(snapshot: ValarDashboardSnapshot) {
        voices = snapshot.voices
        availableCloneModels = snapshot.installedCatalogModels
            .filter { $0.descriptor.voiceSupport.supportsReferenceAudio }
            .map { model in
                let voiceSupport = model.descriptor.voiceSupport
                return RuntimeModelPickerOption(
                    id: model.id,
                    displayName: model.descriptor.displayName,
                    familyID: model.familyID,
                    voiceFeatures: voiceSupport.features,
                    isRecommended: model.isRecommended
                )
            }
            .sorted { lhs, rhs in
                if lhs.familyID == .tadaTTS, rhs.familyID != .tadaTTS {
                    return true
                }
                if lhs.familyID != .tadaTTS, rhs.familyID == .tadaTTS {
                    return false
                }
                if lhs.isRecommended != rhs.isRecommended {
                    return lhs.isRecommended && !rhs.isRecommended
                }
                return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName) == .orderedAscending
            }
        if let selectedVoice {
            self.selectedVoice = snapshot.voices.first(where: { $0.id == selectedVoice.id })
        }
    }

    func loadVoices() async {
        voices = await services.runtime.storedVoices()
        isLoading = false
    }

    func createVoice(label: String, modelID: ModelIdentifier) async {
        errorMessage = nil

        do {
            _ = try await services.createVoice(label: label, modelID: modelID)
            await loadVoices()
        } catch {
            errorMessage = voiceCreationErrorMessage(
                action: "create the voice",
                label: label,
                error: error
            )
        }
    }

    func cloneVoice(_ draft: VoiceCloneDraft) async throws -> VoiceLibraryRecord {
        let result = try await services.cloneVoice(draft)
        await loadVoices()
        return result
    }

    func createDesignedVoice(label: String, prompt: String) async {
        errorMessage = nil

        do {
            _ = try await services.createDesignedVoice(label: label, prompt: prompt)
            await loadVoices()
        } catch {
            errorMessage = voiceCreationErrorMessage(
                action: "save the designed voice",
                label: label,
                error: error
            )
        }
    }

    func deleteVoice(_ id: UUID) async {
        errorMessage = nil
        let deletedVoiceLabel = voices.first(where: { $0.id == id })?.label ?? ""

        do {
            try await services.deleteVoice(id)
        } catch {
            errorMessage = voiceCreationErrorMessage(
                action: "delete the voice",
                label: deletedVoiceLabel,
                error: error
            )
            return
        }

        if selectedVoice?.id == id { selectedVoice = nil }
        await loadVoices()
    }

    func dismissError() {
        errorMessage = nil
    }

    private func voiceCreationErrorMessage(action: String, label: String, error: any Error) -> String {
        let trimmedLabel = label.trimmingCharacters(in: .whitespacesAndNewlines)
        let voiceDescription = trimmedLabel.isEmpty ? "this voice" : "\"\(trimmedLabel)\""
        let reason = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)

        if reason.isEmpty {
            return "Couldn't \(action) for \(voiceDescription)."
        }

        return "Couldn't \(action) for \(voiceDescription): \(reason)"
    }

    private func matchesSearch(_ voice: VoiceLibraryRecord) -> Bool {
        guard !searchText.isEmpty else { return true }
        let query = searchText.lowercased()
        return voice.label.lowercased().contains(query)
            || voice.modelID.lowercased().contains(query)
            || (voice.backendVoiceID?.lowercased().contains(query) ?? false)
            || (voice.voicePrompt?.lowercased().contains(query) ?? false)
    }
}
