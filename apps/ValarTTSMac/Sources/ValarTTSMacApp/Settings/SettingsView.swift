import SwiftUI

private enum SettingsTab: Hashable {
    case runtime
    case storage
    case audio
    case advanced
}

struct SettingsView: View {
    @Bindable var state: SettingsState
    @State private var selectedTab: SettingsTab = .runtime
    @State private var showingResetConfirmation = false

    private let sampleRates = [16_000, 22_050, 24_000, 44_100, 48_000]
    private let bufferSizes = [256, 512, 1_024, 2_048, 4_096]

    private var runtimeModelsPath: String {
        PathRedaction.redact(state.defaults.snapshot.modelsDirectoryPath)
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            runtimeTab
                .tabItem { Label("Runtime", systemImage: "bolt.circle") }
                .tag(SettingsTab.runtime)

            storageTab
                .tabItem { Label("Storage", systemImage: "externaldrive") }
                .tag(SettingsTab.storage)

            audioTab
                .tabItem { Label("Audio", systemImage: "speaker.wave.2") }
                .tag(SettingsTab.audio)

            advancedTab
                .tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }
                .tag(SettingsTab.advanced)
        }
        .frame(minWidth: 640, minHeight: 440)
        .padding(20)
        .confirmationDialog(
            "Reset Settings?",
            isPresented: $showingResetConfirmation,
            titleVisibility: .visible
        ) {
            Button("Reset", role: .destructive) {
                state.resetToDefaults()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("All settings will be reset to defaults. Your installed models, voices, and project data are not affected. This cannot be undone.")
        }
    }

    private var runtimeTab: some View {
        settingsPane {
            SurfaceCard(title: "Runtime", symbol: "bolt.circle") {
                VStack(alignment: .leading, spacing: 14) {
                    Picker("MLX compute units", selection: $state.runtimeComputeUnits) {
                        ForEach(RuntimeComputeUnits.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }

                    Stepper(value: $state.runtimeThreadCount, in: 1 ... 32) {
                        LabeledContent("Thread count", value: "\(state.runtimeThreadCount)")
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Memory budget")
                            Spacer()
                            Text(String(format: "%.0f GiB", state.runtimeMemoryBudgetGiB))
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $state.runtimeMemoryBudgetGiB, in: 1 ... 64, step: 1)
                    }

                    Text("Changes are staged in preferences and persist across launches.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var storageTab: some View {
        settingsPane {
            SurfaceCard(title: "Storage", symbol: "externaldrive") {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Shared models directory")
                        HStack {
                            TextField("Models Directory", text: .constant(runtimeModelsPath))
                                .textFieldStyle(.roundedBorder)
                                .disabled(true)
                        }
                        Text("The Mac app currently uses the shared Valar model-pack location shown above. Custom model directories are not yet supported in the app.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    LabeledContent("Cache size", value: state.cacheSizeDescription)

                    HStack {
                        Button("Clear Cache", role: .destructive) {
                            state.clearCache()
                        }
                        Text(PathRedaction.redact(state.defaults.cacheDirectoryURL.path))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .textSelection(.enabled)
                    }
                    Text("Clear Cache only removes app cache files under the path shown above. Installed model packs and shared Hugging Face caches are not deleted here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("To repair or refresh a model outside the app, use the CLI: `valartts models install <id> --refresh-cache --allow-download` or `valartts models purge-cache <id>`.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var audioTab: some View {
        settingsPane {
            SurfaceCard(title: "Audio", symbol: "speaker.wave.2") {
                VStack(alignment: .leading, spacing: 14) {
                    Picker("Default sample rate", selection: $state.defaultSampleRate) {
                        ForEach(sampleRates, id: \.self) { sampleRate in
                            Text("\(sampleRate.formatted()) Hz").tag(sampleRate)
                        }
                    }

                    Picker("Output device", selection: $state.outputDeviceID) {
                        ForEach(state.outputDevices) { device in
                            Text(device.name).tag(device.id)
                        }
                    }

                    Picker("Buffer size", selection: $state.bufferSize) {
                        ForEach(bufferSizes, id: \.self) { size in
                            Text("\(size.formatted()) frames").tag(size)
                        }
                    }
                }
            }
        }
    }

    private var advancedTab: some View {
        settingsPane(showRestartNote: true) {
            SurfaceCard(title: "Advanced", symbol: "slider.horizontal.3") {
                VStack(alignment: .leading, spacing: 14) {
                    Toggle("Enable debug logging", isOn: $state.debugLoggingEnabled)
                    Toggle("Opt out of telemetry", isOn: $state.telemetryOptOutEnabled)
                    Toggle("Show non-commercial experimental models (restart required)", isOn: $state.showNonCommercialModels)

                    Text("Voxtral and other non-commercial models stay hidden by default. Turn this on only if you intend to use those models under their published license terms.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Divider()

                    Button("Reset to Defaults", role: .destructive) {
                        showingResetConfirmation = true
                    }
                }
            }
        }
    }

    private func settingsPane<Content: View>(
        showRestartNote: Bool = false,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            content()

            Text(state.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)

            if showRestartNote {
                Text("Some changes require restarting the app")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
    }
}
