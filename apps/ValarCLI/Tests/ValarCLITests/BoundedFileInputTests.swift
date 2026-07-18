import Foundation
import XCTest
@testable import valartts

final class BoundedFileInputTests: XCTestCase {
    func testBoundedReadAcceptsExactLimitAndRejectsLargerFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data([1, 2, 3, 4]).write(to: url)

        XCTAssertEqual(
            try BoundedFileInput.readData(
                from: url,
                maximumByteCount: 4,
                label: "fixture"
            ),
            Data([1, 2, 3, 4])
        )
        XCTAssertThrowsError(
            try BoundedFileInput.readData(
                from: url,
                maximumByteCount: 3,
                label: "fixture"
            )
        ) { error in
            XCTAssertEqual(
                error as? BoundedFileInputError,
                .exceedsLimit(label: "fixture", observedBytes: 4, maximumBytes: 3)
            )
        }
    }

    func testBoundedReadRejectsDirectories() throws {
        XCTAssertThrowsError(
            try BoundedFileInput.readData(
                from: FileManager.default.temporaryDirectory,
                maximumByteCount: 4,
                label: "fixture"
            )
        ) { error in
            XCTAssertEqual(error as? BoundedFileInputError, .notRegularFile(label: "fixture"))
        }
    }

    func testAudioLimitsFitDaemonBodyAndBase64Expansion() {
        XCTAssertLessThan(CLIAudioInputLimits.daemonMultipartFileBytes, 15_000_000)
        let base64UpperBound = ((CLIAudioInputLimits.daemonInlineReferenceBytes + 2) / 3) * 4
        XCTAssertLessThan(base64UpperBound, 15_000_000)
    }
}
