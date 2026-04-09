import SwiftUI

// MARK: - App Accent

/// Warm amber accent inspired by the ValarTTS web palette.
/// In light mode it leans toward a rich amber; in dark mode it lifts
/// slightly to stay vibrant against dark backgrounds.
enum ValarAccent {
    static let color = Color("ValarAccent", bundle: nil)

    /// Fallback when the asset catalog is not available.
    static let light = Color(hue: 0.07, saturation: 0.78, brightness: 0.88)
    static let dark  = Color(hue: 0.07, saturation: 0.65, brightness: 0.92)
}

// MARK: - Semantic Surface Tokens

/// Adaptive surface fills that stay visible in both light and dark modes.
/// In light mode, surfaces use a neutral gray tint; in dark mode they use
/// a lighter overlay so cards remain distinguishable from the window background.
extension ShapeStyle where Self == Color {

    /// Primary card surface (SurfaceCard, VoiceCard, RenderJobRow).
    static var surfacePrimary: Color {
        Color(light: Color.black.opacity(0.04), dark: Color.white.opacity(0.06))
    }

    /// Elevated card surface for selected or hovered states.
    static var surfaceElevated: Color {
        Color(light: Color.black.opacity(0.06), dark: Color.white.opacity(0.09))
    }

    /// Recessed surface for text editors, code fields, input areas.
    static var surfaceRecessed: Color {
        Color(light: Color.black.opacity(0.03), dark: Color.white.opacity(0.04))
    }

    /// Subtle badge/chip background (capabilities, backends, counts).
    static var surfaceBadge: Color {
        Color(light: Color.black.opacity(0.05), dark: Color.white.opacity(0.08))
    }
}

// MARK: - Accent Tint Tokens

extension Color {
    static var surfaceOutline: Color {
        Color(light: Color.black.opacity(0.08), dark: Color.white.opacity(0.12))
    }
    static var surfaceOutlineStrong: Color {
        Color(light: Color.black.opacity(0.14), dark: Color.white.opacity(0.18))
    }

    /// Accent-tinted background for selected items, active badges.
    /// Adapts opacity so the tint reads clearly in both modes.
    static var accentTintLight: Color {
        Color.accentColor.opacity(0.10)
    }

    /// Stronger accent tint for selection borders, focused states.
    static var accentTintMedium: Color {
        Color.accentColor.opacity(0.18)
    }

    /// Accent tint for badges/chips alongside accent-colored text.
    static var accentBadge: Color {
        Color.accentColor.opacity(0.12)
    }
}

// MARK: - Status Colors

/// Semantic status colors with adaptive brightness for dark mode legibility.
/// SwiftUI's built-in named colors (.green, .orange, etc.) are already
/// adaptive, but these wrappers let us tune opacity for badge backgrounds.
enum StatusColor {
    static let success    = Color.green
    static let warning    = Color.orange
    static let error      = Color.red
    static let info       = Color.blue
    static let neutral    = Color.secondary

    /// Badge background for a given status color — tuned for dark mode.
    static func badge(_ color: Color) -> Color {
        color.opacity(0.16)
    }
}

// MARK: - Corner Radii

enum ValarSpacing {
    static let cardPadding: CGFloat = 16
}

enum ValarRadius {
    static let card: CGFloat = 14
    static let badge: CGFloat = 6
    static let field: CGFloat = 10
    static let sheet: CGFloat = 20
}

// MARK: - Adaptive Color Initializer

extension Color {
    /// Creates a color that adapts between light and dark appearance.
    init(light: Color, dark: Color) {
        self.init(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            return isDark ? NSColor(dark) : NSColor(light)
        })
    }
}
