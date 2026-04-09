import SwiftUI
import ValarCore
import ValarModelKit

struct ModelDetailView: View {
    let model: CatalogModel
    let downloadProgress: Double?
    let diskUsageBytes: Int
    let onInstall: () -> Void
    let onUninstall: () -> Void
    let onCancelDownload: () -> Void
    @State private var isShowingUninstallConfirmation = false

    private var isDownloading: Bool {
        downloadProgress != nil
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                header
                    .padding(.bottom, 20)

                detailsSection
                    .padding(.bottom, 16)

                if model.familyID == .vibevoiceRealtimeTTS {
                    vibeVoiceInfoCallout
                        .padding(.bottom, 16)
                }

                capabilitiesSection
                    .padding(.bottom, 16)

                if !model.supportedBackends.isEmpty {
                    backendsSection
                        .padding(.bottom, 16)
                }

                if let progress = downloadProgress {
                    downloadSection(progress: progress)
                        .padding(.bottom, 16)
                }

                Spacer(minLength: 20)

                actions
            }
            .padding()
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(domainColor.opacity(0.12))
                        .frame(width: 48, height: 48)
                    Image(systemName: domainIcon)
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(domainColor)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(model.descriptor.displayName)
                        .font(.title3.bold())
                    Text(model.id.rawValue)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }

            HStack(spacing: 8) {
                StatusBadge(state: model.installState)
                if let badge = supportBadge {
                    badge
                }
                if model.isRecommended {
                    recommendedBadge
                }
            }
        }
    }

    private var supportBadge: AnyView? {
        if model.distributionTier == .compatibilityPreview {
            return AnyView(
                Label {
                    Text("Compatibility")
                        .font(.caption2.weight(.semibold))
                } icon: {
                    Image(systemName: "checkmark.shield.fill")
                        .font(.system(size: 8))
                }
                .foregroundStyle(.teal)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.teal.opacity(0.12))
                .clipShape(Capsule())
            )
        }
        switch model.supportTier {
        case .preview:
            return AnyView(
                Label {
                    Text("Preview")
                        .font(.caption2.weight(.semibold))
                } icon: {
                    Image(systemName: "eye.trianglebadge.exclamationmark.fill")
                        .font(.system(size: 8))
                }
                .foregroundStyle(.orange)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.orange.opacity(0.12))
                .clipShape(Capsule())
            )
        case .experimental:
            return AnyView(experimentalBadge)
        case .supported:
            return nil
        }
    }

    private var experimentalBadge: some View {
        Label {
            Text("Experimental")
                .font(.caption2.weight(.semibold))
        } icon: {
            Image(systemName: "flask.fill")
                .font(.system(size: 8))
        }
        .foregroundStyle(.mint)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.mint.opacity(0.12))
        .clipShape(Capsule())
    }

    private var recommendedBadge: some View {
        Label {
            Text("Recommended")
                .font(.caption2.weight(.semibold))
        } icon: {
            Image(systemName: "star.fill")
                .font(.system(size: 8))
        }
        .foregroundStyle(.orange)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.orange.opacity(0.1))
        .clipShape(Capsule())
    }

    // MARK: - Details

    private var detailsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionLabel("Details")

            VStack(spacing: 0) {
                detailRow(label: "Family", value: familyDisplayName(model.familyID.rawValue))
                detailRow(label: "Domain", value: model.descriptor.domain.rawValue.capitalized)
                detailRow(label: "Provider", value: model.providerName)
                if let url = model.providerURL {
                    detailRow(label: "Source", value: url.host() ?? url.absoluteString, selectable: true)
                }
                if let licenseName = model.licenseName {
                    detailRow(label: "License", value: licenseName)
                }
                if let licenseURL = model.licenseURL {
                    detailRow(label: "License URL", value: licenseURL.absoluteString, selectable: true)
                }
                detailRow(label: "Support", value: model.supportTier.rawValue.capitalized)
                detailRow(label: "Release", value: model.releaseEligible ? "Eligible" : "Internal only")
                detailRow(label: "Distribution", value: distributionTierLabel)
                detailRow(label: "State", value: installStateText)
                detailRow(label: "Artifacts", value: "\(model.artifactCount)")
                detailRow(label: "Disk", value: diskUsageText, monospaced: model.installState == .installed)
                if let sr = model.descriptor.defaultSampleRate {
                    detailRow(label: "Sample Rate", value: "\(Int(sr)) Hz", monospaced: true)
                }
                if !model.qualityTierByLanguage.isEmpty {
                    detailRow(label: "Languages", value: languageTierSummary, muted: true)
                }
                if let notes = model.notes {
                    detailRow(label: "Notes", value: notes, muted: true)
                }
            }
            .background(.quaternary.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private func detailRow(
        label: String,
        value: String,
        selectable: Bool = false,
        monospaced: Bool = false,
        muted: Bool = false
    ) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .leading)

            Group {
                if selectable {
                    Text(value)
                        .textSelection(.enabled)
                } else {
                    Text(value)
                }
            }
            .font(monospaced ? .caption.monospacedDigit() : .caption)
            .foregroundStyle(muted ? .secondary : .primary)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
    }

    // MARK: - VibeVoice Info

    private var vibeVoiceInfoCallout: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Model Info")
            VStack(alignment: .leading, spacing: 6) {
                infoRow(icon: "person.fill", text: "Preset-voice only")
                infoRow(icon: "checkmark.circle.fill", text: "English is release-supported")
                infoRow(icon: "exclamationmark.triangle.fill", text: "Multilingual voices remain preview and short-form focused")
                infoRow(icon: "mic.slash.fill", text: "No voice cloning, voice design, or reference-audio conditioning")
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.mint.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(.mint.opacity(0.15), lineWidth: 1)
            )
        }
    }

    private var distributionTierLabel: String {
        switch model.distributionTier {
        case .bundledFirstRun:
            return "Bundled first-run"
        case .optionalInstall:
            return "Optional install"
        case .compatibilityPreview:
            return "Compatibility preview"
        }
    }

    private var languageTierSummary: String {
        model.qualityTierByLanguage
            .sorted { $0.key < $1.key }
            .map { "\($0.key): \($0.value.rawValue)" }
            .joined(separator: ", ")
    }

    private func infoRow(icon: String, text: String) -> some View {
        Label {
            Text(text)
                .font(.caption)
                .foregroundStyle(.primary)
        } icon: {
            Image(systemName: icon)
                .font(.system(size: 10))
                .foregroundStyle(.mint)
                .frame(width: 16)
        }
    }

    // MARK: - Capabilities

    private var capabilitiesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Capabilities")
            FlowLayout(spacing: 6) {
                ForEach(model.descriptor.capabilities.sorted(by: { $0.rawValue < $1.rawValue }), id: \.rawValue) { cap in
                    capabilityTag(capabilityDisplayName(cap.rawValue))
                }
            }
        }
    }

    private func familyDisplayName(_ rawValue: String) -> String {
        switch rawValue {
        case "qwen3_tts": return "Qwen3 TTS"
        case "tada_tts": return "TADA"
        case "voxtral_tts": return "Voxtral"
        case "orpheus_tts": return "Orpheus"
        case "marvis_tts": return "Marvis"
        case "chatterbox_tts": return "Chatterbox"
        case "soprano": return "Soprano"
        case "pocket_tts": return "Pocket TTS"
        case "qwen3_asr": return "Qwen3 ASR"
        case "qwen3_forced_aligner": return "Qwen3 Aligner"
        case "whisper": return "Whisper"
        case "lfm_audio": return "LFM Audio"
        case "vibevoice_realtime_tts": return "VibeVoice"
        default: return rawValue
        }
    }

    private func capabilityDisplayName(_ rawValue: String) -> String {
        switch rawValue {
        case "speech.synthesis": return "Speech Synthesis"
        case "speech.recognition": return "Speech Recognition"
        case "speech.enhancement": return "Speech Enhancement"
        case "speech.streaming": return "Streaming"
        case "speech.forced_alignment": return "Forced Alignment"
        case "speech.to_speech": return "Speech to Speech"
        case "voice.cloning": return "Voice Cloning"
        case "voice.design": return "Voice Design"
        case "voice.preset_voices": return "Preset Voices"
        case "language.multilingual": return "Multilingual"
        case "translation.text": return "Translation"
        case "text.tokenization": return "Tokenization"
        case "render.long_form": return "Long-Form Rendering"
        case "audio.conditioning": return "Audio Conditioning"
        default: return rawValue
        }
    }

    private func capabilityTag(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(domainColor)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(domainColor.opacity(0.1))
            .clipShape(Capsule())
    }

    // MARK: - Backends

    private var backendsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Backends")
            HStack(spacing: 6) {
                ForEach(model.supportedBackends, id: \.rawValue) { backend in
                    Label {
                        Text(backend.rawValue.uppercased())
                            .font(.caption2.weight(.bold))
                    } icon: {
                        Image(systemName: backendIcon(for: backend))
                            .font(.system(size: 9))
                    }
                    .foregroundStyle(.primary.opacity(0.7))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(.quaternary.opacity(0.4))
                    .clipShape(Capsule())
                }
            }
        }
    }

    private func backendIcon(for backend: BackendKind) -> String {
        switch backend {
        case .mlx: return "apple.terminal"
        case .coreml: return "cpu"
        default: return "square.stack.3d.up"
        }
    }

    // MARK: - Download

    private func downloadSection(progress: Double) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Downloading")
            ModelDownloadProgressView(
                modelName: model.descriptor.displayName,
                progress: progress
            ) {
                onCancelDownload()
            }
        }
    }

    // MARK: - Actions

    private var actions: some View {
        Group {
            if model.installState == .installed {
                Button(role: .destructive) {
                    isShowingUninstallConfirmation = true
                } label: {
                    Label("Uninstall", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .accessibilityLabel("Uninstall \(model.descriptor.displayName)")
                .buttonStyle(.bordered)
                .controlSize(.large)
                .confirmationDialog(
                    "Uninstall model \"\(model.descriptor.displayName)\"?",
                    isPresented: $isShowingUninstallConfirmation,
                    titleVisibility: .visible
                ) {
                    Button("Uninstall", role: .destructive) {
                        onUninstall()
                    }
                    .accessibilityLabel("Uninstall \(model.descriptor.displayName)")
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("This removes the installed Valar model pack from this Mac. Shared Hugging Face cache entries, if any, remain available for reuse.")
                }
            } else {
                Button {
                    onInstall()
                } label: {
                    Label(primaryActionTitle, systemImage: primaryActionSymbol)
                        .frame(maxWidth: .infinity)
                }
                .disabled(isDownloading)
                .accessibilityLabel(primaryActionAccessibilityLabel)
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
    }

    // MARK: - Shared helpers

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private var domainIcon: String {
        switch model.descriptor.domain {
        case .tts: return "waveform"
        case .stt: return "mic.fill"
        case .sts: return "arrow.triangle.swap"
        case .codec: return "doc.zipper"
        case .utility: return "gearshape.fill"
        }
    }

    private var domainColor: Color {
        switch model.descriptor.domain {
        case .tts: return .blue
        case .stt: return .purple
        case .sts: return .teal
        case .codec: return .indigo
        case .utility: return .gray
        }
    }

    private var installStateText: String {
        switch model.installState {
        case .installed:
            return "Installed locally"
        case .cached:
            return "Cached in shared storage; installing may still create a local ModelPack"
        case .supported:
            return "Available in catalog only"
        }
    }

    private var diskUsageText: String {
        switch model.installState {
        case .installed:
            return ByteCountFormatter.string(fromByteCount: Int64(diskUsageBytes), countStyle: .file)
        case .cached:
            return "Shared Hugging Face cache (local install may still use disk)"
        case .supported:
            return "Not downloaded yet"
        }
    }

    private var primaryActionTitle: String {
        if isDownloading {
            return "Downloading…"
        }
        switch model.installState {
        case .installed:
            return "Installed"
        case .cached:
            return "Register Cached Model"
        case .supported:
            return "Install Model"
        }
    }

    private var primaryActionSymbol: String {
        if isDownloading {
            return "arrow.down.circle.fill"
        }
        switch model.installState {
        case .installed:
            return "checkmark.circle.fill"
        case .cached:
            return "externaldrive.fill"
        case .supported:
            return "arrow.down.circle.fill"
        }
    }

    private var primaryActionAccessibilityLabel: String {
        if isDownloading {
            return "Downloading…"
        }
        switch model.installState {
        case .installed:
            return "\(model.descriptor.displayName) installed"
        case .cached:
            return "Register cached model \(model.descriptor.displayName)"
        case .supported:
            return "Install \(model.descriptor.displayName)"
        }
    }
}

struct FlowLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalSize: CGSize = .zero

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
            totalSize.width = max(totalSize.width, x - spacing)
            totalSize.height = max(totalSize.height, y + rowHeight)
        }
        return (totalSize, positions)
    }
}
