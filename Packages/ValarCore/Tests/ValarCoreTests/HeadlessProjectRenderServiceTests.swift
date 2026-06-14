import Foundation
import XCTest
import ValarAudio
@testable import ValarCore
import ValarModelKit
import ValarPersistence

final class HeadlessProjectRenderServiceTests: XCTestCase {
    func testOutputFileNameSanitizesChapterTitle() {
        let chapter = ChapterRecord(
            projectID: UUID(),
            index: 2,
            title: "  Act I: The Signal / Return?  ",
            script: "Renderable"
        )

        XCTAssertEqual(
            HeadlessProjectRenderService.outputFileName(for: chapter),
            "003-act-i-the-signal-return.wav"
        )
    }

    func testValidatedOutputURLRejectsPathTraversal() throws {
        let exportsDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)

        XCTAssertThrowsError(
            try HeadlessProjectRenderService.validatedOutputURL(
                for: "../escape.wav",
                in: exportsDirectory
            )
        ) { error in
            guard case HeadlessProjectRenderError.pathTraversal = error else {
                return XCTFail("Expected pathTraversal, got \(error)")
            }
        }
    }

    func testFailureReasonRedactsPaths() {
        let sensitivePath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("valar-secret/input.wav")
            .path
        let error = NSError(
            domain: "ValarRenderTest",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to read \(sensitivePath)"]
        )

        let reason = HeadlessProjectRenderService.failureReason(from: error)

        XCTAssertFalse(reason.contains(sensitivePath))
        XCTAssertTrue(reason.contains("~"))
    }

    func testEnqueueProjectRenderSkipsBlankChaptersAndWritesExport() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let appPaths = ValarAppPaths(baseURL: root)
        let projectStore = ProjectStore(paths: appPaths)
        let renderQueue = RenderQueue()
        let tracker = RenderInvocationTracker()

        let project = try await projectStore.create(title: "Shared Renderer")
        let bundleURL = root
            .appendingPathComponent("SharedRenderer", isDirectory: true)
            .appendingPathExtension("valarproject")
        await projectStore.updateBundleURL(bundleURL, for: project.id)
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 0, title: "Draft", script: "  \n")
        )
        await projectStore.addChapter(
            ChapterRecord(projectID: project.id, index: 1, title: "Opening Scene", script: "Render this")
        )

        let service = HeadlessProjectRenderService(
            renderQueue: renderQueue,
            projectStore: projectStore,
            audioPipeline: AudioPipeline()
        ) { _, options, text in
            await tracker.record(text: text, options: options)
            return AudioChunk(samples: [0, 0.1, -0.1, 0], sampleRate: 24_000)
        }

        let options = RenderSynthesisOptions(
            language: "en",
            temperature: 0.6,
            topP: 0.85,
            repetitionPenalty: 1.1,
            maxTokens: 2_048,
            voiceBehavior: .stableNarrator
        )
        let jobs = await service.enqueueProjectRender(
            project: project,
            modelID: "test-model",
            synthesisOptions: options
        )

        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs.first?.title, "Opening Scene")
        XCTAssertEqual(jobs.first?.outputFileName, "002-opening-scene.wav")
        XCTAssertEqual(jobs.first?.synthesisOptions, options)

        let finishedJobs = try await waitForTerminalJobs(expectedCount: 1, queue: renderQueue)
        let invocations = await tracker.invocations()
        let exports = await projectStore.exports(for: project.id)
        let bundleLocation = await projectStore.bundleLocation(for: project.id)
        let exportsDirectory = try XCTUnwrap(bundleLocation?.exportsDirectory)
        let outputURL = exportsDirectory.appendingPathComponent("002-opening-scene.wav")

        XCTAssertEqual(finishedJobs.map(\.state), [.completed])
        XCTAssertEqual(invocations.map(\.text), ["Render this"])
        XCTAssertEqual(invocations.map(\.options), [options])
        XCTAssertEqual(exports.map(\.fileName), ["002-opening-scene.wav"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))

        let header = try Data(contentsOf: outputURL).prefix(4)
        XCTAssertEqual(String(data: header, encoding: .ascii), "RIFF")
    }
}

private actor RenderInvocationTracker {
    struct Invocation: Sendable, Equatable {
        let text: String
        let options: RenderSynthesisOptions
    }

    private var recordedInvocations: [Invocation] = []

    func record(text: String, options: RenderSynthesisOptions) {
        recordedInvocations.append(Invocation(text: text, options: options))
    }

    func invocations() -> [Invocation] {
        recordedInvocations
    }
}

private func waitForTerminalJobs(
    expectedCount: Int,
    queue: RenderQueue,
    file: StaticString = #filePath,
    line: UInt = #line
) async throws -> [RenderJob] {
    for _ in 0..<100 {
        let jobs = await queue.jobs(matching: nil)
        let terminalJobs = jobs.filter { [.completed, .cancelled, .failed].contains($0.state) }
        if jobs.count == expectedCount, terminalJobs.count == expectedCount {
            return jobs.sorted { $0.createdAt < $1.createdAt }
        }
        try await Task.sleep(nanoseconds: 20_000_000)
    }

    XCTFail("Timed out waiting for \(expectedCount) terminal render job(s).", file: file, line: line)
    return await queue.jobs(matching: nil).sorted { $0.createdAt < $1.createdAt }
}
