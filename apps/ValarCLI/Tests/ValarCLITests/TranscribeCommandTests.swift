import Foundation
import XCTest
@testable import valartts

final class TranscribeCommandTests: XCTestCase {
    func testStreamFlagDefaultsFalse() throws {
        let command = try TranscribeCommand.parse(["audio.wav"])
        XCTAssertFalse(command.stream)
    }

    func testStreamFlagCanBeEnabled() throws {
        let command = try TranscribeCommand.parse(["audio.wav", "--stream"])
        XCTAssertTrue(command.stream)
    }

    func testStreamFlagWithOtherOptions() throws {
        let command = try TranscribeCommand.parse([
            "audio.wav",
            "--stream",
            "--format", "text",
            "--model", "Qwen3-ASR-0.6B",
        ])
        XCTAssertTrue(command.stream)
    }
}
