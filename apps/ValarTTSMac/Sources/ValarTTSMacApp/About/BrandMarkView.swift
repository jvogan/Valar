import SwiftUI

enum BrandColors {
    static let amber = Color(red: 0.85, green: 0.57, blue: 0.23)
    static let navy = Color(red: 0.18, green: 0.29, blue: 0.43)
    static let navyDeep = Color(red: 0.05, green: 0.07, blue: 0.10)
}

struct BrandMarkView: View {
    var size: CGFloat = 120
    var strokeColor: Color = .primary
    var accentColor: Color = BrandColors.amber

    var body: some View {
        Canvas { context, canvasSize in
            let s = canvasSize.width / 28.0

            // Left page
            let leftRect = RoundedRectangle(cornerRadius: 2.8 * s, style: .continuous)
                .path(in: CGRect(x: 3.5 * s, y: 5 * s, width: 9.5 * s, height: 18 * s))
            context.stroke(
                leftRect,
                with: .color(strokeColor.opacity(0.9)),
                lineWidth: 1.8 * s
            )

            // Right page
            let rightRect = RoundedRectangle(cornerRadius: 2.8 * s, style: .continuous)
                .path(in: CGRect(x: 10.2 * s, y: 5 * s, width: 9.5 * s, height: 18 * s))
            context.stroke(
                rightRect,
                with: .color(strokeColor.opacity(0.55)),
                lineWidth: 1.8 * s
            )

            // Waveform
            var wave = Path()
            wave.move(to: CGPoint(x: 7.2 * s, y: 16.2 * s))
            wave.addCurve(
                to: CGPoint(x: 10.9 * s, y: 13.7 * s),
                control1: CGPoint(x: 8.4 * s, y: 14.5 * s),
                control2: CGPoint(x: 9.6 * s, y: 13.7 * s)
            )
            wave.addCurve(
                to: CGPoint(x: 13.7 * s, y: 15.4 * s),
                control1: CGPoint(x: 12.0 * s, y: 13.7 * s),
                control2: CGPoint(x: 12.8 * s, y: 14.4 * s)
            )
            wave.addCurve(
                to: CGPoint(x: 16.3 * s, y: 17.1 * s),
                control1: CGPoint(x: 14.6 * s, y: 16.4 * s),
                control2: CGPoint(x: 15.3 * s, y: 17.1 * s)
            )
            wave.addCurve(
                to: CGPoint(x: 20.3 * s, y: 14.1 * s),
                control1: CGPoint(x: 17.6 * s, y: 17.1 * s),
                control2: CGPoint(x: 18.8 * s, y: 16.2 * s)
            )
            context.stroke(
                wave,
                with: .color(strokeColor),
                style: StrokeStyle(lineWidth: 2.1 * s, lineCap: .round, lineJoin: .round)
            )

            // Accent dot
            let dotRect = CGRect(
                x: (20.9 - 2.2) * s,
                y: (9.1 - 2.2) * s,
                width: 4.4 * s,
                height: 4.4 * s
            )
            context.fill(Path(ellipseIn: dotRect), with: .color(accentColor))
        }
        .frame(width: size, height: size)
        .accessibilityLabel("Valar logo")
    }
}

struct AppIconView: View {
    let size: CGFloat

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [BrandColors.navy, BrandColors.navyDeep],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Color.white.opacity(0.08), Color.clear],
                        startPoint: .top,
                        endPoint: .center
                    )
                )

            BrandMarkView(
                size: size * 0.52,
                strokeColor: .white,
                accentColor: BrandColors.amber
            )
        }
        .frame(width: size, height: size)
    }
}

@MainActor
enum AppIconRenderer {
    @available(macOS 14.0, *)
    static func setAppIcon() {
        let view = AppIconView(size: 1024)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1.0
        guard let cgImage = renderer.cgImage else { return }
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: 1024, height: 1024))
        NSApplication.shared.applicationIconImage = nsImage
    }
}
