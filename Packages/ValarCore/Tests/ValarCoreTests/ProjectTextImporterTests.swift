import XCTest
import ValarModelKit
import ValarPersistence
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

    func testProjectLintBuildsVoiceBibleAndModelWarnings() throws {
        let projectID = UUID()
        let project = ProjectRecord(id: projectID, title: "Cast Scene")
        let chapters = [
            ChapterRecord(
                projectID: projectID,
                index: 0,
                title: "Opening",
                script: """
                [Narrator|style:calm] (sighs) The hall was empty.
                Guest: I heard something.
                """
            ),
        ]
        let speakers = [
            ProjectSpeakerRecord(
                projectID: projectID,
                name: "Narrator",
                voiceModelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
            ),
        ]
        let sopranoModel = try Self.catalogModel(from: XCTUnwrap(SopranoCatalog.supportedEntries.first))

        let payload = ProjectScriptMarkup.lintProject(
            project: project,
            chapters: chapters,
            speakers: speakers,
            model: sopranoModel,
            generatedAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(payload.projectTitle, "Cast Scene")
        XCTAssertEqual(payload.lines.count, 2)
        XCTAssertTrue(payload.issueCount >= 3)
        XCTAssertTrue(payload.issues.contains { $0.code == "speaker_missing_voice_profile" && $0.speakerLabel == "guest" })
        XCTAssertTrue(payload.issues.contains { $0.code == "model_may_not_hold_cast_consistency" })
        XCTAssertTrue(payload.issues.contains { $0.code == "model_may_ignore_expression" && $0.tag == "style" })
        XCTAssertEqual(Set(payload.voiceBible?.profiles.map(\.name) ?? []), Set(["guest", "Narrator"]))
    }

    private static func catalogModel(from entry: SupportedModelCatalogEntry) -> CatalogModel {
        let manifest = entry.manifest
        return CatalogModel(
            id: manifest.id,
            descriptor: ModelDescriptor(manifest: manifest),
            familyID: manifest.familyID,
            installState: .supported,
            providerName: "test",
            providerURL: entry.remoteURL,
            sourceKind: nil,
            isRecommended: entry.isRecommended,
            manifestPath: nil,
            installedPath: nil,
            artifactCount: manifest.artifacts.count,
            supportedBackends: manifest.supportedBackends.map(\.backendKind),
            licenseName: manifest.licenses.first?.name,
            licenseURL: manifest.licenses.first?.sourceURL,
            supportTier: manifest.supportTier,
            releaseEligible: manifest.releaseEligible,
            qualityTierByLanguage: manifest.qualityTierByLanguage,
            distributionTier: entry.distributionTier,
            notes: manifest.notes
        )
    }
}
