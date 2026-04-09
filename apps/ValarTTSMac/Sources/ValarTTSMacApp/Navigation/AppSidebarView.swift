import SwiftUI

struct AppSidebarView: View {
    @Binding var selection: AppSection

    var body: some View {
        List(selection: $selection) {
            Section("Workspace") {
                ForEach(AppSection.allCases.filter(\.isPrimary)) { section in
                    Label(section.title, systemImage: section.symbolName)
                        .tag(section)
                }
            }
            Section("Tools") {
                Label(AppSection.diagnostics.title, systemImage: AppSection.diagnostics.symbolName)
                    .tag(AppSection.diagnostics)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Valar")
    }
}
