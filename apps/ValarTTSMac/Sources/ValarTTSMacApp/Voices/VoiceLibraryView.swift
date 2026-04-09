import SwiftUI
import ValarCore
import ValarModelKit
import ValarPersistence

struct VoiceLibraryView: View {
    @Environment(AppState.self) private var appState
    @State private var isAudioDropTargeted = false

    private let columns = [GridItem(.adaptive(minimum: 200, maximum: 280), spacing: 16)]

    var body: some View {
        content(appState.voiceLibraryState)
            .navigationTitle("Voices")
            .task {
                await appState.voiceLibraryState.loadVoices()
            }
    }

    private func content(_ state: VoiceLibraryState) -> some View {
        @Bindable var s = state
        return ScrollView(.vertical) {
            if state.isLoading {
                LazyVGrid(columns: columns, spacing: 16) {
                    ForEach(0..<6, id: \.self) { _ in
                        SkeletonVoiceCard()
                    }
                }
                .padding(20)
                .transition(.opacity)
            } else if state.voices.isEmpty {
                ContentUnavailableView(
                    "No voices yet",
                    systemImage: "person.wave.2",
                    description: Text("Clone a voice from a reference audio clip, or design one from a text description.\nDrop audio files here to start cloning.")
                )
                .frame(maxWidth: .infinity, minHeight: 300)
                .transition(.opacity)
            } else if state.filteredVoices.isEmpty {
                ContentUnavailableView.search(text: state.searchText)
                    .frame(maxWidth: .infinity, minHeight: 300)
            } else {
                VStack(alignment: .leading, spacing: 24) {
                    voiceSection(
                        title: "Preset Voices",
                        subtitle: "Model-declared preset voices are immutable.",
                        voices: state.filteredPresetVoices,
                        state: state
                    )
                    voiceSection(
                        title: "Saved Voices",
                        subtitle: "User-created and cloned voices saved in the local library.",
                        voices: state.filteredSavedVoices,
                        state: state
                    )
                }
                .padding(20)
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: state.isLoading)
        .overlay {
            if isAudioDropTargeted {
                ZStack {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.accentColor.opacity(0.05))
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            Color.accentColor,
                            style: StrokeStyle(lineWidth: 2.5, dash: [10, 7])
                        )
                    VStack(spacing: 8) {
                        Image(systemName: "mic.badge.plus")
                            .font(.largeTitle)
                            .foregroundStyle(Color.accentColor)
                            .symbolEffect(.pulse, options: .repeating)
                        Text("Drop audio to clone a voice")
                            .font(.headline)
                            .foregroundStyle(.primary)
                        Text(VoiceCloneFileValidator.allowedExtensionsDisplayText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(20)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                .padding(8)
                .transition(.opacity.combined(with: .scale(scale: 0.97)))
                .allowsHitTesting(false)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: isAudioDropTargeted)
        .dropDestination(for: URL.self) { items, _ in
            guard let droppedFile = items.first else { return false }

            do {
                try VoiceCloneFileValidator.validateFileSelection(droppedFile)
            } catch {
                state.errorMessage = error.localizedDescription
                return false
            }

            state.pendingCloneAudioURL = droppedFile
            state.errorMessage = nil
            state.cloneWizardOpen = true
            return true
        } isTargeted: { targeted in
            isAudioDropTargeted = targeted
        }
        .searchable(text: $s.searchText, prompt: "Search voices")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    state.cloneWizardOpen = true
                } label: {
                    Label("Clone Voice", systemImage: "mic.badge.plus")
                }
                Button {
                    state.designSheetOpen = true
                } label: {
                    Label("Design Voice", systemImage: "paintbrush")
                }
            }
        }
        .inspector(isPresented: Binding(
            get: { state.selectedVoice != nil },
            set: { if !$0 { state.selectedVoice = nil } }
        )) {
            if let voice = state.selectedVoice {
                VoiceDetailSheet(voice: voice, onDelete: {
                    Task {
                        await state.deleteVoice(voice.id)
                        await appState.refreshSnapshot()
                    }
                }, onUseForGeneration: {
                    appState.generatorState.selectVoice(voice.id)
                    appState.selectedSection = .generate
                })
                .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
            }
        }
        .sheet(isPresented: $s.cloneWizardOpen) {
            VoiceCloneWizardView(
                prefilledAudioURL: state.pendingCloneAudioURL,
                availableModels: state.availableCloneModels
            ) { @MainActor draft in
                _ = try await state.cloneVoice(draft)
                state.cloneWizardOpen = false
                state.pendingCloneAudioURL = nil
                await appState.refreshSnapshot()
            }
            .onDisappear {
                state.pendingCloneAudioURL = nil
            }
        }
        .sheet(isPresented: $s.designSheetOpen) {
            VoiceDesignView { label, prompt in
                Task {
                    await state.createDesignedVoice(label: label, prompt: prompt)
                    guard state.errorMessage == nil else { return }
                    state.designSheetOpen = false
                    await appState.refreshSnapshot()
                }
            }
        }
        .alert(
            "Voice Creation Failed",
            isPresented: Binding(
                get: { state.errorMessage != nil },
                set: { isPresented in
                    if !isPresented {
                        state.dismissError()
                    }
                }
            )
        ) {
            Button("OK", role: .cancel) {
                state.dismissError()
            }
        } message: {
            Text(state.errorMessage ?? "")
        }
    }

    @ViewBuilder
    private func voiceSection(
        title: String,
        subtitle: String,
        voices: [VoiceLibraryRecord],
        state: VoiceLibraryState
    ) -> some View {
        if !voices.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                LazyVGrid(columns: columns, spacing: 16) {
                    ForEach(voices) { voice in
                        Button {
                            state.selectedVoice = voice
                        } label: {
                            VoiceCardView(voice: voice, isSelected: state.selectedVoice?.id == voice.id)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}
