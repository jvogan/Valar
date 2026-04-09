import SwiftUI

struct InteractiveCardBackground: ViewModifier {
    let isSelected: Bool
    let isHovered: Bool
    let cornerRadius: CGFloat

    init(
        isSelected: Bool = false,
        isHovered: Bool = false,
        cornerRadius: CGFloat = ValarRadius.card
    ) {
        self.isSelected = isSelected
        self.isHovered = isHovered
        self.cornerRadius = cornerRadius
    }

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        let strokeColor = isSelected ? Color.accentTintMedium : (isHovered ? Color.surfaceOutlineStrong : Color.surfaceOutline)

        content
            .background {
                shape.fill(.surfacePrimary)

                if isSelected {
                    shape.fill(Color.accentTintLight)
                } else if isHovered {
                    shape.fill(.surfaceElevated)
                }
            }
            .overlay {
                shape.strokeBorder(strokeColor, lineWidth: isSelected ? 1.5 : 1)
            }
            .shadow(color: isHovered ? Color.black.opacity(0.08) : .clear, radius: 10, y: 4)
    }
}

extension View {
    func interactiveCardBackground(
        isSelected: Bool = false,
        isHovered: Bool = false,
        cornerRadius: CGFloat = ValarRadius.card
    ) -> some View {
        modifier(InteractiveCardBackground(
            isSelected: isSelected,
            isHovered: isHovered,
            cornerRadius: cornerRadius
        ))
    }
}
