import SwiftUI

enum AppSection: String, CaseIterable, Identifiable, Hashable {
    case generate
    case project = "projects"
    case voices
    case models
    case diagnostics

    var id: String { rawValue }

    var title: String {
        switch self {
        case .generate: return "Generate"
        case .project: return "Project"
        case .voices: return "Voices"
        case .models: return "Models"
        case .diagnostics: return "Diagnostics"
        }
    }

    var symbolName: String {
        switch self {
        case .generate: return "waveform"
        case .project: return "book.pages"
        case .voices: return "person.wave.2"
        case .models: return "cpu"
        case .diagnostics: return "stethoscope"
        }
    }

    var isPrimary: Bool {
        self != .diagnostics
    }
}
