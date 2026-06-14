import XCTest
@testable import ValarCore

final class ProjectTextImporterTests: XCTestCase {
    func testMarkdownHeadingsBecomeSegments() {
        let text = """
        # Opening

        First chapter body.

        ## Second

        Second chapter body.
        """

        let segments = ProjectTextImporter.parse(
            text: text,
            fallbackTitle: "Book",
            splitMode: "markdown-headings",
            defaultSpeakerLabel: "Narrator"
        )

        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].title, "Opening")
        XCTAssertEqual(segments[0].text, "First chapter body.")
        XCTAssertEqual(segments[0].speakerLabel, "Narrator")
        XCTAssertEqual(segments[1].title, "Second")
        XCTAssertEqual(segments[1].text, "Second chapter body.")
        XCTAssertNotEqual(segments[0].sourceHash, segments[1].sourceHash)
    }

    func testDialogueModeParsesSpeakerTagsAndColonLines() {
        let text = """
        [S1] We should leave now.
        [S2|style:calm] Not before the signal.
        Narrator: The room went quiet.
        """

        let segments = ProjectTextImporter.parse(
            text: text,
            fallbackTitle: "Scene",
            splitMode: "dialogue"
        )

        XCTAssertEqual(segments.map(\.speakerLabel), ["S1", "S2", "Narrator"])
        XCTAssertEqual(segments.map(\.text), [
            "We should leave now.",
            "Not before the signal.",
            "The room went quiet.",
        ])
        XCTAssertEqual(segments[1].title, "S2: Not before the signal")
    }

    func testScriptMarkupParsesAttributesAndExpressiveTags() {
        let line = ProjectScriptMarkup.parseLine(
            "[S2|style:calm|lang:en|whisper] (sighs) We wait for the signal. <laugh>",
            lineNumber: 7
        )

        XCTAssertEqual(line.lineNumber, 7)
        XCTAssertEqual(line.speakerLabel, "S2")
        XCTAssertEqual(line.attributes["style"], "calm")
        XCTAssertEqual(line.attributes["lang"], "en")
        XCTAssertEqual(line.tags, ["whisper", "laugh", "sighs"])
        XCTAssertEqual(line.text, "(sighs) We wait for the signal. <laugh>")
    }

    func testParagraphModeUsesFirstSentenceTitles() {
        let segments = ProjectTextImporter.parse(
            text: "One paragraph.\n\nSecond paragraph.",
            fallbackTitle: "Article",
            splitMode: "paragraphs",
            defaultSpeakerLabel: "Reader"
        )

        XCTAssertEqual(segments.map(\.title), ["One paragraph", "Second paragraph"])
        XCTAssertEqual(segments.map(\.speakerLabel), ["Reader", "Reader"])
        XCTAssertFalse(segments[0].sourceHash.isEmpty)
        XCTAssertNotEqual(segments[0].sourceHash, segments[1].sourceHash)
    }
}
