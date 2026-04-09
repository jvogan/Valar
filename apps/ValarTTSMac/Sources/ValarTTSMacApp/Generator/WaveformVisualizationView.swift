import SwiftUI

struct WaveformVisualizationView: View {
    let samples: [Float]
    let progress: Double
    let accentColor: Color

    private let barWidth: CGFloat = 2.5
    private let barSpacing: CGFloat = 1.5
    private let minimumBarFraction: CGFloat = 0.05

    private var clampedProgress: Double {
        min(max(progress, 0), 1)
    }

    private var accessibilityProgressValue: String {
        "\(Int((clampedProgress * 100).rounded()))% complete"
    }

    init(samples: [Float], progress: Double, accentColor: Color = .accentColor) {
        self.samples = samples
        self.progress = progress
        self.accentColor = accentColor
    }

    var body: some View {
        Canvas { context, size in
            draw(in: &context, size: size)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Audio waveform")
        .accessibilityValue(accessibilityProgressValue)
    }

    private func draw(in context: inout GraphicsContext, size: CGSize) {
        let step = barWidth + barSpacing
        let barCount = max(1, Int(size.width / step))
        let resampled = resampledAndNormalized(to: barCount)
        let midY = size.height / 2
        let maxHalf = size.height / 2 - 1
        let playedEnd = clampedProgress * Double(barCount)

        for i in 0..<barCount {
            let x = CGFloat(i) * step + barSpacing / 2
            let amplitude = CGFloat(resampled[i])
            let clamped = max(minimumBarFraction, amplitude)
            let halfH = clamped * maxHalf
            let rect = CGRect(
                x: x,
                y: midY - halfH,
                width: barWidth,
                height: halfH * 2
            )
            let path = Path(roundedRect: rect, cornerRadius: barWidth / 2)
            let played = Double(i) < playedEnd

            if played {
                context.fill(path, with: .color(accentColor))
            } else {
                context.fill(path, with: .color(Color.secondary.opacity(0.18)))
            }
        }

        // Playback cursor
        guard clampedProgress > 0.003, clampedProgress < 0.997 else { return }
        let cursorX = CGFloat(clampedProgress) * size.width

        // Vertical line
        let line = CGRect(x: cursorX - 0.5, y: 1, width: 1, height: size.height - 2)
        context.fill(Path(roundedRect: line, cornerRadius: 0.5),
                     with: .color(Color.primary.opacity(0.3)))

        // Small head dot
        let dotSize: CGFloat = 5
        let dot = CGRect(
            x: cursorX - dotSize / 2,
            y: 0,
            width: dotSize,
            height: dotSize
        )
        context.fill(Path(ellipseIn: dot), with: .color(Color.primary.opacity(0.45)))
    }

    private func resampledAndNormalized(to count: Int) -> [Float] {
        guard !samples.isEmpty, count > 0 else {
            return [Float](repeating: 0, count: max(count, 0))
        }

        let peak = samples.max() ?? 1
        let scale: Float = peak > 0.001 ? 1.0 / peak : 1.0

        if samples.count == count {
            return samples.map { min($0 * scale, 1) }
        }

        return (0..<count).map { i in
            let lo = i * samples.count / count
            let hi = min((i + 1) * samples.count / count, samples.count)
            guard lo < hi else { return 0 }
            return min((samples[lo..<hi].max() ?? 0) * scale, 1)
        }
    }
}
