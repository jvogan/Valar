import Foundation
import ValarCore
import ValarPersistence
import XCTest
@testable import valartts

final class ProjectsCommandTests: XCTestCase {
    func testExportPackManifestUsesBundleRelativeArtifactPaths() throws {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? fileManager.removeItem(at: root) }

        let bundleURL = root.appendingPathComponent("Book.valarproject", isDirectory: true)
        let exportsURL = bundleURL.appendingPathComponent("Exports", isDirectory: true)
        try fileManager.createDirectory(at: exportsURL, withIntermediateDirectories: true)
        try Data([0x01, 0x02, 0x03]).write(to: exportsURL.appendingPathComponent("chapter.wav"))

        let projectID = UUID()
        let chapterID = UUID()
        let exportID = UUID()
        let snapshot = ProjectBundleSnapshot(
            project: ProjectRecord(id: projectID, title: "Book"),
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapters: [
                ChapterRecord(
                    id: chapterID,
                    projectID: projectID,
                    index: 0,
                    title: "Chapter",
                    script: "Narrator: Hello.",
                    speakerLabel: "Narrator"
                ),
            ],
            renderJobs: [],
            exports: [
                ExportRecord(
                    id: exportID,
                    projectID: projectID,
                    fileName: "../chapter.wav",
                    createdAt: Date(timeIntervalSince1970: 0)
                ),
            ],
            speakers: [
                ProjectSpeakerRecord(projectID: projectID, name: "Narrator"),
            ]
        )

        let manifest = try ProjectsCommand.exportPackManifest(
            snapshot: snapshot,
            bundleURL: bundleURL,
            generatedAt: Date(timeIntervalSince1970: 0),
            fileManager: fileManager
        )

        let artifact = try XCTUnwrap(manifest.artifacts.first)
        XCTAssertEqual(artifact.path, "Exports/chapter.wav")
        XCTAssertFalse(artifact.path.hasPrefix("/"))
        XCTAssertFalse(artifact.path.contains(".."))
        XCTAssertEqual(artifact.byteCount, 3)
        XCTAssertEqual(manifest.chapters.first?.sourceHash, ProjectScriptMarkup.sourceHash(
            title: "Chapter",
            text: "Narrator: Hello.",
            speakerLabel: "Narrator"
        ))
    }
}
