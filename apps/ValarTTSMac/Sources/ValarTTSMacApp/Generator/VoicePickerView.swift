import SwiftUI
import ValarModelKit

struct VoicePickerView: View {
    @Environment(AppState.self) private var appState
    @Binding var selectedModelID: ModelIdentifier?

    var body: some View {
        let models = appState.generatorState.availableModels
        let modelIDs = models.map(\.id)

        Picker("Model", selection: $selectedModelID) {
            Text(models.isEmpty ? "No models installed" : "Select model...")
                .tag(nil as ModelIdentifier?)
                .disabled(true)
            if !models.isEmpty {
                Divider()
                ForEach(models) { model in
                    if model.distributionTier == .compatibilityPreview {
                        Label(model.displayName, systemImage: "checkmark.shield")
                            .tag(model.id as ModelIdentifier?)
                    } else if model.supportTier == .preview {
                        Label(model.displayName, systemImage: "eye.trianglebadge.exclamationmark")
                            .tag(model.id as ModelIdentifier?)
                    } else if model.supportTier == .experimental {
                        Label(model.displayName, systemImage: "flask.fill")
                            .tag(model.id as ModelIdentifier?)
                    } else {
                        Text(model.displayName).tag(model.id as ModelIdentifier?)
                    }
                }
            }
        }
        .frame(width: 200)
        .disabled(models.isEmpty)
        .onAppear {
            clampSelectedModelID(to: modelIDs)
        }
        .onChange(of: modelIDs) { _, updatedModelIDs in
            clampSelectedModelID(to: updatedModelIDs)
        }
    }

    private func clampSelectedModelID(to modelIDs: [ModelIdentifier]) {
        if let selectedModelID, !modelIDs.contains(selectedModelID) {
            self.selectedModelID = nil
        }
    }
}
