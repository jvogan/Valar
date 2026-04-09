import Foundation
import XCTest
@testable import ValarCore
import ValarModelKit
import ValarPersistence

final class DTOsTests: XCTestCase {
    func testCommandEnvelopeEncodesTypedPayload() throws {
        let envelope = ValarCommandEnvelope(
            ok: true,
            command: "valartts models list",
            data: ModelListPayloadDTO(
                message: "Loaded 1 supported model(s).",
                models: [
                    ModelSummaryDTO(
                        id: "demo-model",
                        displayName: "Demo Model",
                    family: "demo-family",
                    provider: "Valar",
                    installState: "supported",
                    installedPath: nil,
                    isRecommended: true,
                    licenseName: "CC BY-NC 4.0"
                ),
                ]
            ),
            error: nil
        )

        let data = try JSONEncoder().encode(envelope)
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let payload = try XCTUnwrap(object["data"] as? [String: Any])
        let models = try XCTUnwrap(payload["models"] as? [[String: Any]])

        XCTAssertEqual(object["ok"] as? Bool, true)
        XCTAssertEqual(object["command"] as? String, "valartts models list")
        XCTAssertEqual(payload["message"] as? String, "Loaded 1 supported model(s).")
        XCTAssertEqual(models.first?["id"] as? String, "demo-model")
        XCTAssertEqual(models.first?["isRecommended"] as? Bool, true)
    }

    func testDaemonErrorEnvelopeEncodesSharedShape() throws {
        let envelope = DaemonErrorEnvelopeDTO(
            error: ValarCommandErrorDTO(
                code: 500,
                kind: "daemon_error",
                message: "Something failed",
                help: "Retry later."
            )
        )

        let object = try XCTUnwrap(jsonObject(for: envelope))
        let error = try XCTUnwrap(object["error"] as? [String: Any])

        XCTAssertEqual(object["ok"] as? Bool, false)
        XCTAssertEqual(error["code"] as? Int, 500)
        XCTAssertEqual(error["kind"] as? String, "daemon_error")
        XCTAssertEqual(error["message"] as? String, "Something failed")
        XCTAssertEqual(error["help"] as? String, "Retry later.")
    }

    func testModelSummaryDTOMapsCatalogModel() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let model = CatalogModel(
            id: descriptor.id,
            descriptor: descriptor,
            familyID: descriptor.familyID,
            installState: .installed,
            providerName: "ModelScope",
            providerURL: URL(string: "https://example.com/model"),
            sourceKind: .remoteURL,
            isRecommended: true,
            manifestPath: "/tmp/manifest.json",
            installedPath: "/tmp/model",
            artifactCount: 4,
            supportedBackends: [.mlx],
            licenseName: "CC BY-NC 4.0",
            licenseURL: URL(string: "https://example.com/license"),
            notes: "demo"
        )

        let dto = ModelSummaryDTO(from: model)

        XCTAssertEqual(dto.id, descriptor.id.rawValue)
        XCTAssertEqual(dto.displayName, descriptor.displayName)
        XCTAssertEqual(dto.family, descriptor.familyID.rawValue)
        XCTAssertEqual(dto.provider, "ModelScope")
        XCTAssertEqual(dto.installState, "installed")
        XCTAssertEqual(dto.installedPath, "/tmp/model")
        XCTAssertTrue(dto.isRecommended)
        XCTAssertEqual(dto.licenseName, "CC BY-NC 4.0")
        XCTAssertEqual(dto.licenseURL, "https://example.com/license")
    }

    func testModelSummaryDTOMapsStaleInstallPathStatus() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS CustomVoice",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let model = CatalogModel(
            id: descriptor.id,
            descriptor: descriptor,
            familyID: descriptor.familyID,
            installState: .supported,
            providerName: "Valar",
            providerURL: nil,
            sourceKind: .localFile,
            isRecommended: false,
            manifestPath: nil,
            installedPath: nil,
            artifactCount: 4,
            supportedBackends: [.mlx],
            notes: nil,
            cachedOnDisk: false,
            installPathStatus: .missingInstalledPath
        )

        let dto = ModelSummaryDTO(from: model)

        XCTAssertEqual(dto.installPathStatus, "missingInstalledPath")
    }

    func testModelRouteDescriptorDTOMapsInstallState() {
        let descriptor = ModelDescriptor(
            id: "HumeAI/mlx-tada-3b",
            familyID: .tadaTTS,
            displayName: "TADA 3B",
            domain: .tts,
            capabilities: [.speechSynthesis, .voiceCloning, .audioConditioning]
        )
        let model = CatalogModel(
            id: descriptor.id,
            descriptor: descriptor,
            familyID: descriptor.familyID,
            installState: .cached,
            providerName: "huggingface.co",
            providerURL: URL(string: "https://huggingface.co/HumeAI/mlx-tada-3b"),
            sourceKind: .remoteURL,
            isRecommended: false,
            manifestPath: nil,
            installedPath: nil,
            artifactCount: 5,
            supportedBackends: [.mlx],
            licenseName: "Llama 3.2 Community",
            licenseURL: URL(string: "https://huggingface.co/HumeAI/mlx-tada-3b"),
            notes: "demo",
            cachedOnDisk: true
        )

        let dto = ModelRouteDescriptorDTO(from: model)

        XCTAssertEqual(dto.id, descriptor.id.rawValue)
        XCTAssertEqual(dto.installState, "cached")
        XCTAssertFalse(dto.installed)
        XCTAssertTrue(dto.cachedOnDisk)
        XCTAssertEqual(dto.voiceFeatures, ["referenceAudio"])
        XCTAssertTrue(dto.supportsReferenceAudio)
    }

    func testModelInstallRequestDTOAcceptsLegacyModelIDKey() throws {
        let data = #"{"model_id":"HumeAI/mlx-tada-1b"}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelInstallRequestDTO.self, from: data)
        XCTAssertEqual(decoded.model, "HumeAI/mlx-tada-1b")
        XCTAssertFalse(decoded.allowDownload)
        XCTAssertFalse(decoded.refreshCache)
    }

    func testModelInstallRequestDTOPrefersCanonicalModelKey() throws {
        let data = #"{"model":"HumeAI/mlx-tada-3b","model_id":"ignored"}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelInstallRequestDTO.self, from: data)
        XCTAssertEqual(decoded.model, "HumeAI/mlx-tada-3b")
    }

    func testSpeechSynthesisPayloadDTOEncodesVibeVoiceSelectionMetadata() throws {
        let payload = SpeechSynthesisPayloadDTO(
            message: "Wrote WAV to /tmp/out.wav",
            modelID: "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            outputPath: "/tmp/out.wav",
            text: "Hola desde Valar.",
            voiceID: "sp-Spk0_woman",
            effectiveVoiceID: "sp-Spk0_woman",
            effectiveLanguage: "es",
            voiceSelectionMode: "auto_default"
        )

        let object = try XCTUnwrap(jsonObject(for: payload))

        XCTAssertEqual(object["voiceID"] as? String, "sp-Spk0_woman")
        XCTAssertEqual(object["effectiveVoiceID"] as? String, "sp-Spk0_woman")
        XCTAssertEqual(object["effectiveLanguage"] as? String, "es")
        XCTAssertEqual(object["voiceSelectionMode"] as? String, "auto_default")
    }

    func testDaemonRuntimeTrimRequestDTODefaultsMissingWarmStartFlag() throws {
        let data = #"{"modelIDs":["mlx-community/VibeVoice-Realtime-0.5B-4bit"]}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(DaemonRuntimeTrimRequestDTO.self, from: data)

        XCTAssertEqual(decoded.modelIDs, ["mlx-community/VibeVoice-Realtime-0.5B-4bit"])
        XCTAssertFalse(decoded.includeWarmStartModels)
    }

    func testModelInstallRequestDTODecodesInstallFlags() throws {
        let data = #"{"model":"mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16","allow_download":true,"refresh_cache":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelInstallRequestDTO.self, from: data)
        XCTAssertEqual(decoded.model, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        XCTAssertTrue(decoded.allowDownload)
        XCTAssertTrue(decoded.refreshCache)
    }

    func testVoiceAndChapterDTOsPreserveDetailFields() {
        let voice = VoiceLibraryRecord(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000010")!,
            label: "Narrator",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            runtimeModelID: "runtime-model",
            sourceAssetName: "source.wav",
            referenceAudioAssetName: "reference.wav",
            referenceTranscript: "Hello world",
            referenceDurationSeconds: 1.5,
            referenceSampleRate: 24_000,
            referenceChannelCount: 1,
            conditioningFormat: "tada.reference/v1",
            voiceKind: "tadaReference",
            voicePrompt: "Warm narrator",
            createdAt: Date(timeIntervalSince1970: 1_700_000_000)
        )
        let chapter = ChapterRecord(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000020")!,
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000021")!,
            index: 2,
            title: "Chapter 3",
            script: "Once upon a time",
            speakerLabel: "Narrator",
            estimatedDurationSeconds: 3.2,
            sourceAudioAssetName: "chapter.wav",
            sourceAudioSampleRate: 24_000,
            sourceAudioDurationSeconds: 3.2,
            transcriptionJSON: "{\"text\":\"Once upon a time\"}",
            transcriptionModelID: "asr-model",
            alignmentJSON: "{\"tokens\":[]}",
            alignmentModelID: "align-model",
            derivedTranslationText: "Habia una vez"
        )

        let voiceDTO = VoiceDetailDTO(from: voice, preview: "reference.wav")
        let chapterDTO = ChapterDTO(from: chapter)

        XCTAssertEqual(voiceDTO.id, voice.id.uuidString)
        XCTAssertEqual(voiceDTO.runtimeModelID, "runtime-model")
        XCTAssertEqual(voiceDTO.referenceTranscript, "Hello world")
        XCTAssertEqual(voiceDTO.conditioningFormat, "tada.reference/v1")
        XCTAssertEqual(voiceDTO.voiceKind, "tadaReference")
        XCTAssertNil(voiceDTO.conditioningData)
        XCTAssertNil(voiceDTO.conditioningAssetName)
        XCTAssertNil(voiceDTO.conditioningSourceModel)
        XCTAssertNil(voiceDTO.conditioningMetadata)
        XCTAssertEqual(voiceDTO.preview, "reference.wav")
        XCTAssertTrue(voiceDTO.isClonedVoice)

        XCTAssertEqual(chapterDTO.id, chapter.id.uuidString)
        XCTAssertEqual(chapterDTO.projectID, chapter.projectID.uuidString)
        XCTAssertEqual(chapterDTO.index, 2)
        XCTAssertEqual(chapterDTO.textLength, chapter.script.count)
        XCTAssertEqual(chapterDTO.speakerLabel, "Narrator")
        XCTAssertTrue(chapterDTO.hasSourceAudio)
        XCTAssertEqual(chapterDTO.sourceAudioAssetName, "chapter.wav")
        XCTAssertEqual(chapterDTO.transcriptionModelID, "asr-model")
        XCTAssertEqual(chapterDTO.alignmentModelID, "align-model")
        XCTAssertEqual(chapterDTO.derivedTranslationText, "Habia una vez")
    }

    func testRenderJobAndDiagnosticsDTOsMapNestedStructures() {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-ASR-0.6B-8bit",
            familyID: .qwen3ASR,
            displayName: "Qwen3 ASR 0.6B",
            domain: .stt,
            capabilities: [.speechRecognition]
        )
        let snapshot = ModelResidencySnapshot(
            descriptor: descriptor,
            state: .resident,
            lastTouchedAt: Date(timeIntervalSince1970: 1_700_000_100),
            residentRank: 1,
            estimatedResidentBytes: 1234,
            actualResidentBytes: 4321,
            residencyPolicy: .automatic,
            activeSessionCount: 2
        )
        let renderJob = RenderJob(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000030")!,
            projectID: UUID(uuidString: "00000000-0000-0000-0000-000000000031")!,
            modelID: descriptor.id,
            chapterIDs: [UUID(uuidString: "00000000-0000-0000-0000-000000000032")!],
            outputFileName: "003-chapter.wav",
            createdAt: Date(timeIntervalSince1970: 1_700_000_200),
            state: .running,
            priority: 1,
            progress: 0.42,
            title: "Chapter 3",
            failureReason: nil,
            queuePosition: 0
        )
        let project = ProjectRecord(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000040")!,
            title: "Book",
            createdAt: Date(timeIntervalSince1970: 1_700_000_300),
            updatedAt: Date(timeIntervalSince1970: 1_700_000_400),
            notes: "draft"
        )
        let voice = VoiceLibraryRecord(
            id: UUID(uuidString: "00000000-0000-0000-0000-000000000050")!,
            label: "Lead",
            modelID: descriptor.id.rawValue
        )

        let renderDTO = RenderJobDTO(from: renderJob)
        let diagnostics = DiagnosticsDTO(
            appPaths: AppPathsDTO(from: ValarAppPaths(baseURL: URL(fileURLWithPath: "/tmp/ValarTTS", isDirectory: true))),
            runtimeConfiguration: RuntimeConfigurationDTO(from: RuntimeConfiguration(warmStartModelIDs: ["model-a"])),
            models: [
                ModelSummaryDTO(
                    id: descriptor.id.rawValue,
                    displayName: descriptor.displayName,
                    family: descriptor.familyID.rawValue,
                    provider: "Valar",
                    installState: "installed",
                    installedPath: "/tmp/model",
                    installPathStatus: "valid",
                    isRecommended: true
                ),
            ],
            modelSnapshots: [ModelResidencySnapshotDTO(from: snapshot)],
            projects: [ProjectSummaryDTO(from: project)],
            voices: [
                VoiceSummaryDTO(
                    id: voice.id.uuidString,
                    label: voice.label,
                    modelID: voice.modelID,
                    preview: "preview.wav"
                ),
            ],
            renders: [renderDTO],
            lastUpdatedAt: "2026-03-21T00:00:00Z"
        )

        XCTAssertEqual(renderDTO.chapterID, "00000000-0000-0000-0000-000000000032")
        XCTAssertEqual(renderDTO.state, "running")
        XCTAssertEqual(renderDTO.progressLabel, "42%")
        XCTAssertEqual(diagnostics.modelSnapshots.first?.actualResidentBytes, 4321)
        XCTAssertEqual(diagnostics.projects.first?.notes, "draft")
        XCTAssertEqual(diagnostics.renders.first?.outputFileName, "003-chapter.wav")
        XCTAssertEqual(diagnostics.lastUpdatedAt, "2026-03-21T00:00:00Z")
        XCTAssertEqual(diagnostics.runtimeConfiguration.warmStartModelIDs, ["model-a"])
        XCTAssertEqual(diagnostics.models.first?.installPathStatus, "valid")
    }

    func testDaemonRuntimeStatusDTOExposesWarmStartSource() throws {
        let descriptor = ModelDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            familyID: .qwen3TTS,
            displayName: "Qwen3 TTS Base",
            domain: .tts,
            capabilities: [.speechSynthesis]
        )
        let snapshot = ModelResidencySnapshotDTO(
            id: descriptor.id.rawValue,
            displayName: descriptor.displayName,
            domain: descriptor.domain.rawValue,
            state: "resident",
            lastTouchedAt: "2026-03-21T00:00:00Z",
            residentRank: 1,
            estimatedResidentBytes: 1234,
            actualResidentBytes: 4321,
            residencyPolicy: "automatic",
            activeSessionCount: 0,
            isWarmStartModel: true,
            idleTrimEligible: false
        )
        let status = DaemonRuntimeStatusDTO(
            processIdentifier: 42,
            daemonPIDFilePath: "/tmp/valarttsd.pid",
            daemonPIDFilePresent: true,
            daemonPIDFileMatchesProcess: true,
            residentModels: [snapshot],
            totalResidentBytes: 4321,
            memoryBudgetBytes: 9_999,
            warmPolicy: "eager",
            warmStartModelSource: .default,
            configuredWarmStartModels: [],
            effectiveWarmStartModels: ["mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "mlx-community/Qwen3-ASR-0.6B-8bit"],
            orphanedModelPackPaths: [],
            idleResidentExtraModels: [],
            idleTrimEligibleModels: [],
            idleResidentExtraModelsLikelyDrift: false,
            idleResidentExtraModelsAdvisory: nil,
            lastIdleTrimResult: nil,
            metalDeviceName: nil,
            processFootprintBytes: 5_000,
            processFootprintHighWaterBytes: nil,
            processCPUPercent: 1.5,
            processCPUCurrentPercent: 1.2,
            processCPUCurrentHighWaterPercent: nil,
            processCPUAveragePercentSinceStart: nil,
            processCPUAverageHighWaterPercentSinceStart: nil,
            availableDiskBytes: nil,
            availableDiskLowWaterBytes: nil,
            prewarmedModels: ["mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"],
            warmingModels: [],
            activeSynthesisCount: 0,
            oldestActiveSynthesisAgeSeconds: nil,
            stalledSynthesisCount: 0,
            activeSynthesisRequests: [],
            lastSynthesisCompletionReason: nil,
            uptimeSeconds: 12.5
        )

        let object = try XCTUnwrap(jsonObject(for: status))
        let data = try JSONEncoder().encode(status)
        let decoded = try JSONDecoder().decode(DaemonRuntimeStatusDTO.self, from: data)

        XCTAssertEqual(object["warmStartModelSource"] as? String, "default")
        XCTAssertEqual((object["effectiveWarmStartModels"] as? [String])?.count, 2)
        XCTAssertEqual(decoded.warmStartModelSource, .default)
    }

    func testPayloadDTOsEncodeStableTopLevelKeys() throws {
        let alignmentPayload = AlignmentPayloadDTO(
            message: "Aligned transcript.",
            modelID: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
            audioPath: "/tmp/demo.wav",
            outputPath: "/tmp/alignment.json",
            transcript: "Hello world",
            tokens: [
                AlignmentTokenDTO(
                    text: "Hello",
                    startTime: 0.0,
                    endTime: 0.5,
                    confidence: 0.98
                ),
            ]
        )
        let projectPayload = ProjectInfoPayloadDTO(
            message: "Loaded project metadata.",
            project: ProjectInfoDTO(
                title: "Book",
                projectID: "project-123",
                bundlePath: "/tmp/Book.valarproject",
                createdAt: "2026-03-21T00:00:00Z",
                openedAt: "2026-03-21T00:05:00Z",
                chapters: 3,
                renderJobs: 2,
                exports: 1,
                speakers: 4
            )
        )
        let renderPayload = RenderStatusPayloadDTO(
            message: "Loaded render status.",
            projectTitle: "Book",
            watched: true,
            generatedAt: "2026-03-21T00:10:00Z",
            processedCount: 1,
            queuedCount: 2,
            remainingPendingCount: 1,
            renders: [
                RenderJobDTO(
                    id: "render-123",
                    projectID: "project-123",
                    chapterID: "chapter-123",
                    modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                    title: "Intro",
                    outputFileName: "001-intro.wav",
                    state: "queued",
                    progress: 0.0,
                    progressLabel: "0%",
                    failureReason: nil,
                    createdAt: "2026-03-21T00:09:00Z"
                ),
            ]
        )
        let exportPayload = ExportListPayloadDTO(
            message: "Loaded exports.",
            projectTitle: "Book",
            exports: [
                ExportDTO(
                    id: "export-123",
                    projectID: "project-123",
                    fileName: "book.zip",
                    createdAt: "2026-03-21T00:11:00Z",
                    checksum: "abc123"
                ),
            ]
        )

        let alignmentObject = try XCTUnwrap(jsonObject(for: alignmentPayload))
        let projectObject = try XCTUnwrap(jsonObject(for: projectPayload))
        let renderObject = try XCTUnwrap(jsonObject(for: renderPayload))
        let exportObject = try XCTUnwrap(jsonObject(for: exportPayload))

        XCTAssertEqual(alignmentObject["modelID"] as? String, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        XCTAssertEqual(alignmentObject["audioPath"] as? String, "/tmp/demo.wav")
        XCTAssertEqual((alignmentObject["tokens"] as? [[String: Any]])?.first?["text"] as? String, "Hello")

        XCTAssertEqual(projectObject["message"] as? String, "Loaded project metadata.")
        XCTAssertEqual((projectObject["project"] as? [String: Any])?["projectID"] as? String, "project-123")

        XCTAssertEqual(renderObject["projectTitle"] as? String, "Book")
        XCTAssertEqual(renderObject["watched"] as? Bool, true)
        XCTAssertEqual((renderObject["renders"] as? [[String: Any]])?.first?["outputFileName"] as? String, "001-intro.wav")

        XCTAssertEqual(exportObject["projectTitle"] as? String, "Book")
        XCTAssertEqual((exportObject["exports"] as? [[String: Any]])?.first?["fileName"] as? String, "book.zip")
    }

    private func jsonObject<Value: Encodable>(for value: Value) throws -> [String: Any]? {
        let data = try JSONEncoder().encode(value)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}
