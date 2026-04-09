import SwiftUI

struct SurfaceCard<Content: View>: View {
    let title: String
    let symbol: String
    let content: Content

    init(title: String, symbol: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.symbol = symbol
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: symbol)
                    .foregroundStyle(Color.accentColor)
                Text(title)
                    .font(.headline)
                Spacer()
            }
            content
        }
        .padding(ValarSpacing.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.surfacePrimary, in: RoundedRectangle(cornerRadius: ValarRadius.card, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: ValarRadius.card, style: .continuous)
                .strokeBorder(Color.surfaceOutline, lineWidth: 1)
        )
    }
}
