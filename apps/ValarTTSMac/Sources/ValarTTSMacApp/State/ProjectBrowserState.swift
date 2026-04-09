import SwiftUI
import ValarPersistence

@Observable
@MainActor
final class ProjectBrowserState {
    var projects: [ProjectRecord] = []
    var draftTitle = "Untitled Project"
    var loadError: String?
    var selectedProjectID: UUID?

    private let services: ValarServiceHub

    init(services: ValarServiceHub) {
        self.services = services
    }

    func apply(snapshot: ValarDashboardSnapshot, selectedProjectID: UUID?) {
        projects = snapshot.projects
        self.selectedProjectID = resolvedSelectedProjectID(
            in: snapshot.projects,
            preferred: selectedProjectID ?? self.selectedProjectID
        )
        loadError = nil
    }

    func loadProjects() async {
        do {
            let loadedProjects = try await services.grdbProjectStore.fetchAll()
            projects = loadedProjects
            selectedProjectID = resolvedSelectedProjectID(in: loadedProjects)
            loadError = nil
        } catch {
            loadError = PathRedaction.redactMessage(error.localizedDescription)
            projects = []
            selectedProjectID = nil
        }
    }

    func createProject() async {
        let title = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else { return }
        guard let record = await services.createProject(title: title) else {
            loadError = services.projectCreationErrorMessage ?? "Failed to create project."
            return
        }

        loadError = nil

        selectedProjectID = record.id
        draftTitle = "Untitled Project"
        await loadProjects()
    }

    func deleteProject(_ id: UUID) async {
        do {
            try await services.deleteProject(id)
            loadError = nil
        } catch {
            loadError = PathRedaction.redactMessage(error.localizedDescription)
            return
        }

        if selectedProjectID == id { selectedProjectID = nil }
        await loadProjects()
    }

    private func resolvedSelectedProjectID(in projects: [ProjectRecord], preferred: UUID? = nil) -> UUID? {
        if let preferred,
           projects.contains(where: { $0.id == preferred }) {
            return preferred
        }

        return projects.first?.id
    }
}
