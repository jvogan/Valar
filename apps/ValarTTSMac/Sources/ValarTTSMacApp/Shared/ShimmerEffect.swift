import SwiftUI

struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = -1.2

    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    stops: [
                        .init(color: .clear, location: phase - 0.3),
                        .init(color: .white.opacity(0.18), location: phase),
                        .init(color: .clear, location: phase + 0.3),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .clipped()
            .accessibilityHidden(true)
            .onAppear {
                withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) {
                    phase = 2.2
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// MARK: - Skeleton Shapes

struct SkeletonLine: View {
    var width: CGFloat? = nil
    var height: CGFloat = 12

    var body: some View {
        RoundedRectangle(cornerRadius: height / 2, style: .continuous)
            .fill(.quaternary)
            .frame(width: width, height: height)
            .shimmer()
    }
}

struct SkeletonModelCard: View {
    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(.quaternary)
                .frame(width: 28, height: 28)

            VStack(alignment: .leading, spacing: 6) {
                SkeletonLine(width: 140, height: 14)
                HStack(spacing: 6) {
                    SkeletonLine(width: 48, height: 10)
                    SkeletonLine(width: 100, height: 10)
                }
                SkeletonLine(width: 60, height: 9)
            }

            Spacer()

            SkeletonLine(width: 64, height: 22)
        }
        .padding(.vertical, 4)
    }
}

struct SkeletonVoiceCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(.quaternary)
                    .frame(width: 32, height: 32)
                Spacer()
                Circle()
                    .fill(.quaternary)
                    .frame(width: 24, height: 24)
            }

            SkeletonLine(width: 100, height: 16)
            SkeletonLine(width: 140, height: 10)

            HStack {
                SkeletonLine(width: 70, height: 9)
                Spacer()
                SkeletonLine(width: 50, height: 9)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.18))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct SkeletonFormField: View {
    var labelWidth: CGFloat = 100

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SkeletonLine(width: labelWidth, height: 14)
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(.quaternary)
                .frame(height: 28)
                .shimmer()
        }
    }
}
