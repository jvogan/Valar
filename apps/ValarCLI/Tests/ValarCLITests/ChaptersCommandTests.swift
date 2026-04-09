import Foundation
import XCTest
import ValarPersistence
@testable import valartts

final class ChaptersCommandTests: XCTestCase {
    func testAttachAudioCommandParsesArguments() throws {
        let command = try XCTUnwrap(
            try ValarCLI.parseAsRoot([
                "chapters",
                "attach-audio",
                "00000000-0000-0000-0000-000000000001",
                "--audio", "recording.wav",
            ]) as? ChaptersCommand.AttachAudio
        )

        XCTAssertEqual(command.id, "00000000-0000-0000-0000-000000000001")
        XCTAssertEqual(command.audio, "recording.wav")
    }

    func testRenderChapterSummaryLineIncludesSourceAudioFlag() {
        let chapter = ChapterRecord(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000010")!,
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000020")!,
            index: 1,
            title: "Intro",
            script: "Hello there",
            sourceAudioAssetName: "00000000-0000-0000-0000-000000000010.wav",
            sourceAudioSampleRate: 24_000,
            sourceAudioDurationSeconds: 1.5
        )

        XCTAssertEqual(
            ChaptersCommand.renderChapterSummaryLine(chapter),
            "00000000-0000-0000-0000-000000000010 | 2 | Intro | 11 chars | hasSourceAudio: true"
        )
    }

    func testStagedAssetNameUsesChapterIDAndLowercasedExtension() {
        let chapterID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
        let sourceURL = URL(fileURLWithPath: "/tmp/Recording.WAV")

        XCTAssertEqual(
            ChaptersCommand.stagedAssetName(chapterID: chapterID, sourceURL: sourceURL),
            "00000000-0000-0000-0000-000000000001.wav"
        )
    }
}
