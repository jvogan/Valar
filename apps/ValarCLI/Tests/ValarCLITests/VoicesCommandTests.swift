import XCTest
@testable import valartts

final class VoicesCommandTests: XCTestCase {
    func testCreateParsesSavedTADAVoiceFlags() throws {
        let command = try VoicesCommand.Create.parse([
            "--model", "HumeAI/mlx-tada-3b",
            "--reference-audio", "/tmp/ref.wav",
            "--reference-transcript", "hello world",
            "--name", "My Voice",
        ])

        XCTAssertEqual(command.model, "HumeAI/mlx-tada-3b")
        XCTAssertEqual(command.referenceAudio, "/tmp/ref.wav")
        XCTAssertEqual(command.referenceTranscript, "hello world")
        XCTAssertEqual(command.name, "My Voice")
    }
}
