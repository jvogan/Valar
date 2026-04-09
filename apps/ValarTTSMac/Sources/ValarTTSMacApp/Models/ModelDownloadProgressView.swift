import SwiftUI

struct ModelDownloadProgressView: View {
    let modelName: String
    let progress: Double
    let onCancel: () -> Void

    @State private var shimmerPhase: CGFloat = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(modelName)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)

                Spacer()

                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 36, alignment: .trailing)
                    .contentTransition(.numericText())

                Button {
                    onCancel()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .controlSize(.small)
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track
                    Capsule()
                        .fill(.quaternary)

                    // Fill
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [.accentColor, .accentColor.opacity(0.7)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: max(0, geo.size.width * progress))

                    // Shimmer overlay on fill
                    Capsule()
                        .fill(
                            LinearGradient(
                                stops: [
                                    .init(color: .white.opacity(0), location: shimmerPhase - 0.15),
                                    .init(color: .white.opacity(0.25), location: shimmerPhase),
                                    .init(color: .white.opacity(0), location: shimmerPhase + 0.15),
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: max(0, geo.size.width * progress))
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())
            .animation(.easeInOut(duration: 0.35), value: progress)
        }
        .onAppear {
            withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                shimmerPhase = 1.3
            }
        }
    }
}
