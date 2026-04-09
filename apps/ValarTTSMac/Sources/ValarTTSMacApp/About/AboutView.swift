import AppKit
import SwiftUI

enum AboutInfo {
    static var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "2.0.0"
    }

    static var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
    }

    static var versionLabel: String {
        "Version \(version) (\(build))"
    }

    static var copyableVersionString: String {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osString = "macOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        return "Valar \(version) (\(build)) · \(osString) · Apple Silicon"
    }

    static let copyright = "© 2024 Prince Canuma, 2026 Valar contributors"

    static let githubURL = URL(string: "https://github.com/valartts/valartts")!
    static let docsURL = URL(string: "https://github.com/valartts/valartts/tree/main/docs")!
    static let licenseURL = URL(string: "https://github.com/valartts/valartts/blob/main/LICENSE")!
}

struct AboutView: View {
    @State private var iconFloat: Bool = false
    @State private var glowPulse: Bool = false
    @State private var copied = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer().frame(height: 36)

            brandMark
            Spacer().frame(height: 20)
            titleBlock
            Spacer().frame(height: 16)
            versionRow
            Spacer().frame(height: 24)
            divider
            Spacer().frame(height: 16)
            descriptionBlock
            Spacer().frame(height: 20)
            linksRow
            Spacer().frame(height: 24)
            divider
            Spacer().frame(height: 14)
            credits
            Spacer().frame(height: 24)
        }
        .frame(width: 360)
        .onAppear {
            withAnimation(.easeInOut(duration: 2.8).repeatForever(autoreverses: true)) {
                iconFloat = true
            }
            withAnimation(.easeInOut(duration: 3.4).repeatForever(autoreverses: true)) {
                glowPulse = true
            }
        }
    }

    private var brandMark: some View {
        BrandMarkView(size: 96)
            .shadow(
                color: BrandColors.amber.opacity(glowPulse ? 0.35 : 0.12),
                radius: glowPulse ? 20 : 10
            )
            .offset(y: iconFloat ? -3 : 3)
    }

    private var titleBlock: some View {
        VStack(spacing: 4) {
            Text("Valar")
                .font(.system(size: 26, weight: .bold, design: .rounded))
            Text("Native Text-to-Speech for macOS")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var versionRow: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(AboutInfo.copyableVersionString, forType: .string)
            withAnimation(.easeInOut(duration: 0.2)) { copied = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) {
                withAnimation(.easeInOut(duration: 0.2)) { copied = false }
            }
        } label: {
            HStack(spacing: 6) {
                Text(AboutInfo.versionLabel)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .font(.caption)
                    .foregroundStyle(copied ? Color.green : Color.gray)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.18))
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .help("Click to copy version info for bug reports")
    }

    private var divider: some View {
        Rectangle()
            .fill(.quaternary)
            .frame(height: 1)
            .padding(.horizontal, 40)
    }

    private var descriptionBlock: some View {
        VStack(spacing: 6) {
            Text("Local-first speech synthesis powered by")
                .foregroundStyle(.secondary)
            Text("Qwen models on Apple Silicon via MLX.")
                .foregroundStyle(.secondary)
        }
        .font(.callout)
        .multilineTextAlignment(.center)
        .padding(.horizontal, 32)
    }

    private var linksRow: some View {
        HStack(spacing: 12) {
            linkButton("GitHub", symbol: "chevron.left.forwardslash.chevron.right", url: AboutInfo.githubURL)
            linkButton("Docs", symbol: "book", url: AboutInfo.docsURL)
            linkButton("License", symbol: "doc.text", url: AboutInfo.licenseURL)
        }
    }

    private func linkButton(_ title: String, symbol: String, url: URL) -> some View {
        Button {
            NSWorkspace.shared.open(url)
        } label: {
            Label(title, systemImage: symbol)
                .font(.caption.weight(.medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.18))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            if hovering {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
    }

    private var credits: some View {
        VStack(spacing: 2) {
            Text(AboutInfo.copyright)
            Text("Built with Swift & MLX")
        }
        .font(.caption2)
        .foregroundStyle(.tertiary)
    }
}

@MainActor
enum AboutWindowController {
    private static var windowController: NSWindowController?

    static func show() {
        if let existing = windowController, let window = existing.window, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let hosting = NSHostingController(rootView: AboutView())
        let window = NSWindow(contentViewController: hosting)
        window.title = "About Valar"
        window.styleMask = [.closable, .titled, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isReleasedWhenClosed = false
        window.setContentSize(NSSize(width: 360, height: 480))
        window.center()

        let controller = NSWindowController(window: window)
        windowController = controller
        controller.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
