import SwiftUI

extension KeyboardShortcut {
    static let generate = KeyboardShortcut("g", modifiers: .command)
    static let toggleInspector = KeyboardShortcut("i", modifiers: [.command, .option])
    static let renderChapter = KeyboardShortcut("r", modifiers: [.command, .shift])
    static let renderProject = KeyboardShortcut("r", modifiers: [.command, .option])
    static let exportProject = KeyboardShortcut("e", modifiers: [.command, .shift])
}
