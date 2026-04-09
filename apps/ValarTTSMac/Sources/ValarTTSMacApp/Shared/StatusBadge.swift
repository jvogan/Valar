import SwiftUI
import ValarCore

struct StatusBadge: View {
    let state: CatalogInstallState

    private var icon: String {
        switch state {
        case .installed: return "checkmark.circle.fill"
        case .cached: return "externaldrive.fill"
        case .supported: return "arrow.down.circle"
        }
    }

    private var label: String {
        switch state {
        case .installed: return "Installed"
        case .cached: return "Cached"
        case .supported: return "Available"
        }
    }

    private var tint: Color {
        switch state {
        case .installed: return .green
        case .cached: return .orange
        case .supported: return .secondary
        }
    }

    var body: some View {
        Label {
            Text(label)
                .font(.caption2.weight(.semibold))
        } icon: {
            Image(systemName: icon)
                .font(.caption2)
        }
        .foregroundStyle(state == .supported ? .secondary : tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(state == .supported ? Color.accentBadge : tint.opacity(0.12))
        .clipShape(Capsule())
    }
}
