import Foundation
import XCTest
import ValarModelKit
@testable import valartts

final class SpeakCommandTests: XCTestCase {
    func testFormatDefaultsToWav() throws {
        let command = try SpeakCommand.parse(["Hello", "--output", "out.wav"])
        XCTAssertEqual(command.format, .wav)
    }

    func testFormatWavExplicit() throws {
        let command = try SpeakCommand.parse([
            "Hello",
            "--output", "out.wav",
            "--format", "wav",
        ])
        XCTAssertEqual(command.format, .wav)
    }

    func testFormatOggOpus() throws {
        let command = try SpeakCommand.parse([
            "Hello",
            "--output", "out.ogg",
            "--format", "ogg_opus",
        ])
        XCTAssertEqual(command.format, .oggOpus)
    }

    func testFormatOggOpusWithOtherOptions() throws {
        let command = try SpeakCommand.parse([
            "Hello",
            "--output", "out.ogg",
            "--format", "ogg_opus",
            "--model", "Qwen3-TTS-12Hz-1.7B-Base",
        ])
        XCTAssertEqual(command.format, .oggOpus)
    }

    func testInvalidFormatThrows() throws {
        XCTAssertThrowsError(
            try SpeakCommand.parse([
                "Hello",
                "--output", "out.wav",
                "--format", "mp3",
            ])
        )
    }

    func testLongFormOptionsParseForVoxtralStyleInvocation() throws {
        let command = try SpeakCommand.parse([
            "--model", "mistralai/Voxtral-4B-TTS-2603",
            "--voice", "neutral_female",
            "--language", "en",
            "--text", "Hello from Voxtral.",
            "--output", "/tmp/voxtral.wav",
        ])

        XCTAssertEqual(command.model, "mistralai/Voxtral-4B-TTS-2603")
        XCTAssertEqual(command.voice, "neutral_female")
        XCTAssertEqual(command.language, "en")
        XCTAssertEqual(command.text, "Hello from Voxtral.")
        XCTAssertEqual(command.output, "/tmp/voxtral.wav")
    }

    func testVoxtralAliasVoiceParses() throws {
        let command = try SpeakCommand.parse([
            "--model", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            "--voice", "emma",
            "--language", "en",
            "--text", "Alias path.",
            "--output", "/tmp/voxtral-alias.wav",
        ])

        XCTAssertEqual(command.model, "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
        XCTAssertEqual(command.voice, "emma")
    }

    func testVoxtralRandomVoiceParses() throws {
        let command = try SpeakCommand.parse([
            "--model", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            "--voice", "random",
            "--language", "en",
            "--text", "Random path.",
            "--output", "/tmp/voxtral-random.wav",
        ])

        XCTAssertEqual(command.voice, "random")
    }

    func testVoxtralExcludedRandomPresetParses() throws {
        let command = try SpeakCommand.parse([
            "--model", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            "--voice", "ar_male",
            "--language", "ar",
            "--text", "Arabic preset path.",
            "--output", "/tmp/voxtral-arabic.wav",
        ])

        XCTAssertEqual(command.voice, "ar_male")
        XCTAssertEqual(command.language, "ar")
    }

    func testStyleControlsParse() throws {
        let command = try SpeakCommand.parse([
            "--model", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            "--text", "Hello from Qwen.",
            "--instruct", "Warm, close-mic narration with a calm cadence.",
            "--exaggeration", "0.7",
            "--cfg-weight", "0.4",
            "--output", "/tmp/qwen.wav",
        ])

        XCTAssertEqual(command.instruct, "Warm, close-mic narration with a calm cadence.")
        XCTAssertEqual(command.exaggeration, 0.7)
        XCTAssertEqual(command.cfgWeight, 0.4)
    }

    func testVibeVoicePresetInfersFamilyModelQuery() {
        XCTAssertEqual(
            SpeakCommand.inferredSpeechModelQuery(fromExplicitVoiceIdentifier: "en-Carter_man"),
            ModelFamilyID.vibevoiceRealtimeTTS.rawValue
        )
        XCTAssertEqual(
            SpeakCommand.inferredSpeechModelQuery(fromExplicitVoiceIdentifier: "Emma"),
            ModelFamilyID.vibevoiceRealtimeTTS.rawValue
        )
        XCTAssertNil(SpeakCommand.inferredSpeechModelQuery(fromExplicitVoiceIdentifier: UUID().uuidString))
        XCTAssertNil(SpeakCommand.inferredSpeechModelQuery(fromExplicitVoiceIdentifier: "not_a_vibe_voice"))
    }

    func testVibeVoiceRandomInfersFamilyModelQuery() {
        XCTAssertEqual(
            SpeakCommand.inferredSpeechModelQuery(fromExplicitVoiceIdentifier: "random"),
            ModelFamilyID.vibevoiceRealtimeTTS.rawValue
        )
    }

    func testVibeVoicePresetResolutionBeatsSavedVoiceLabelsWhenFamilyMatches() {
        XCTAssertTrue(
            SpeakCommand.prefersVibeVoicePresetResolution(
                rawIdentifier: "en-Emma_woman",
                familyID: .vibevoiceRealtimeTTS
            )
        )
        XCTAssertTrue(
            SpeakCommand.prefersVibeVoicePresetResolution(
                rawIdentifier: "Emma",
                familyID: .vibevoiceRealtimeTTS
            )
        )
        XCTAssertFalse(
            SpeakCommand.prefersVibeVoicePresetResolution(
                rawIdentifier: "Emma",
                familyID: .qwen3TTS
            )
        )
    }
}
