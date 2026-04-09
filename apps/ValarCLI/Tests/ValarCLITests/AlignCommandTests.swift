import Foundation
import XCTest
@testable import valartts

final class AlignCommandTests: XCTestCase {
    func testResolvedTranscriptReturnsLiteralTranscript() throws {
        XCTAssertEqual(
            try AlignCommand.resolvedTranscript(from: " Hello world "),
            "Hello world"
        )
    }

    func testResolvedTranscriptLoadsAtFileSyntax() throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let transcriptURL = temporaryDirectory.appendingPathComponent("transcript.txt", isDirectory: false)
        try "Hello from file\n".write(to: transcriptURL, atomically: true, encoding: .utf8)

        let transcript = try AlignCommand.resolvedTranscript(from: "@\(transcriptURL.path)")

        XCTAssertEqual(transcript, "Hello from file")
    }
}
