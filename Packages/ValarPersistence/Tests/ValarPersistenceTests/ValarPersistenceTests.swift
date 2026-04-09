import XCTest
@testable import ValarPersistence
import GRDB
import ValarModelKit

final class ValarPersistenceTests: XCTestCase {
    private func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func fileSystemNumber(at url: URL) throws -> NSNumber {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let fileNumber = attributes[.systemFileNumber] as? NSNumber else {
            throw CocoaError(.fileReadUnknown)
        }
        return fileNumber
    }

    func testProjectBundleUsesValarProjectExtension() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let store = ProjectStore()
        let project = try await store.create(title: "Narrated Chapter One")
        let documentURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        await store.updateBundleURL(documentURL, for: project.id)
        let bundleURL = await store.bundleURL(for: project.id)

        XCTAssertEqual(bundleURL?.pathExtension, "valarproject")
    }

    func testProjectBundleLocationIncludesManifestAndAssetDirectories() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let store = ProjectStore()
        let project = try await store.create(title: "Narrated Chapter One")
        let documentURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        await store.updateBundleURL(documentURL, for: project.id)
        let location = await store.bundleLocation(for: project.id)

        XCTAssertEqual(location?.bundleURL.pathExtension, "valarproject")
        XCTAssertEqual(location?.manifestURL.lastPathComponent, "manifest.json")
        XCTAssertEqual(location?.sqliteURL.pathExtension, "sqlite")
        XCTAssertEqual(location?.assetsDirectory.lastPathComponent, "Assets")
        XCTAssertEqual(location?.exportsDirectory.lastPathComponent, "Exports")
        XCTAssertEqual(location?.cacheDirectory.lastPathComponent, "Cache")
    }

    func testProjectBundleWriterCreatesDirectoryManifestAndSQLite() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let project = ProjectRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            title: "Narrated Chapter One",
            createdAt: Date(timeIntervalSince1970: 1_710_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_710_000_600),
            notes: "Opening chapter"
        )
        let chapter = ChapterRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "A beginning.",
            speakerLabel: "Narrator",
            estimatedDurationSeconds: 12.5
        )
        let snapshot = ProjectBundleSnapshot(
            project: project,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapter],
            renderJobs: [],
            exports: []
        )
        let bundleURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        let location = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: bundleURL
        )

        let createdAt = Date(timeIntervalSince1970: 1_710_001_000)
        let manifest = try ProjectBundleWriter().write(snapshot, to: location, createdAt: createdAt)

        XCTAssertTrue(FileManager.default.fileExists(atPath: bundleURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: location.manifestURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: location.sqliteURL.path))
        XCTAssertEqual(manifest.version, 1)
        XCTAssertEqual(manifest.createdAt, createdAt)
        XCTAssertEqual(manifest.modelID, snapshot.modelID)
        XCTAssertEqual(manifest.chapters, [
            ProjectBundleManifest.ChapterSummary(
                id: chapter.id,
                index: chapter.index,
                title: chapter.title
            )
        ])
    }

    func testProjectBundleRoundTripPreservesAllProjectState() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let project = ProjectRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            title: "Narrated Chapter One",
            createdAt: Date(timeIntervalSince1970: 1_710_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_710_000_600),
            notes: "Opening chapter"
        )
        let chapterOne = ChapterRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "A beginning.",
            speakerLabel: "Narrator",
            estimatedDurationSeconds: 12.5,
            sourceAudioAssetName: "chapter-1.wav",
            sourceAudioSampleRate: 24_000,
            sourceAudioDurationSeconds: 11.2,
            transcriptionJSON: #"{"segments":[{"text":"A beginning."}]}"#,
            transcriptionModelID: "whisper-large-v3",
            alignmentJSON: #"{"tokens":[{"start":0.0,"end":0.8}]}"#,
            alignmentModelID: "ctc-aligner-v1",
            derivedTranslationText: "Un commencement."
        )
        let chapterTwo = ChapterRecord(
            id: UUID(uuidString: "66666666-7777-8888-9999-AAAAAAAAAAAA")!,
            projectID: project.id,
            index: 1,
            title: "Chapter 2",
            script: "A continuation.",
            speakerLabel: "Companion",
            estimatedDurationSeconds: 18.75
        )
        let renderJob = RenderJobRecord(
            id: UUID(uuidString: "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF")!,
            projectID: project.id,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [chapterOne.id, chapterTwo.id],
            outputFileName: "narrated.wav",
            createdAt: Date(timeIntervalSince1970: 1_710_002_000),
            updatedAt: Date(timeIntervalSince1970: 1_710_002_300),
            state: "completed",
            priority: 2,
            progress: 1,
            title: "Full narration",
            synthesisOptions: RenderSynthesisOptions(
                language: "en",
                temperature: 0.6,
                topP: 0.88,
                repetitionPenalty: 1.05,
                maxTokens: 3_072,
                voiceBehavior: .stableNarrator
            )
        )
        let export = ExportRecord(
            id: UUID(uuidString: "12345678-90AB-CDEF-1234-567890ABCDEF")!,
            projectID: project.id,
            fileName: "chapter-one.wav",
            createdAt: Date(timeIntervalSince1970: 1_710_003_000),
            checksum: "abc123"
        )
        let speaker = ProjectSpeakerRecord(
            id: UUID(uuidString: "99999999-8888-7777-6666-555555555555")!,
            projectID: project.id,
            name: "Narrator",
            voiceModelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            language: "en"
        )
        let snapshot = ProjectBundleSnapshot(
            project: project,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapterTwo, chapterOne],
            renderJobs: [renderJob],
            exports: [export],
            speakers: [speaker]
        )
        let bundleURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        let location = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: bundleURL
        )

        _ = try ProjectBundleWriter().write(
            snapshot,
            to: location,
            createdAt: Date(timeIntervalSince1970: 1_710_004_000)
        )
        let bundle = try ProjectBundleReader().read(from: bundleURL)

        XCTAssertEqual(bundle.snapshot.project, project)
        XCTAssertEqual(bundle.snapshot.modelID, snapshot.modelID)
        XCTAssertEqual(bundle.snapshot.chapters, [chapterOne, chapterTwo])
        XCTAssertEqual(bundle.snapshot.renderJobs, [renderJob])
        XCTAssertEqual(bundle.snapshot.exports, [export])
        XCTAssertEqual(bundle.snapshot.speakers, [speaker])
        XCTAssertEqual(bundle.manifest.projectID, project.id)
        XCTAssertEqual(bundle.manifest.title, project.title)
        XCTAssertEqual(bundle.manifest.modelID, snapshot.modelID)
        XCTAssertEqual(bundle.manifest.chapters.map(\.title), ["Chapter 1", "Chapter 2"])
    }

    func testProjectBundleSavePreservesExistingBundleDirectoriesAcrossResave() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let project = ProjectRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            title: "Narrated Chapter One",
            createdAt: Date(timeIntervalSince1970: 1_710_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_710_000_600),
            notes: "Opening chapter"
        )
        let chapter = ChapterRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "A beginning.",
            speakerLabel: "Narrator",
            estimatedDurationSeconds: 12.5
        )
        let export = ExportRecord(
            id: UUID(uuidString: "12345678-90AB-CDEF-1234-567890ABCDEF")!,
            projectID: project.id,
            fileName: "chapter-one.wav",
            createdAt: Date(timeIntervalSince1970: 1_710_003_000),
            checksum: "abc123"
        )
        let bundleURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        let location = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: bundleURL
        )
        let initialSnapshot = ProjectBundleSnapshot(
            project: project,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapter],
            renderJobs: [],
            exports: [export]
        )

        _ = try ProjectBundleWriter().write(
            initialSnapshot,
            to: location,
            createdAt: Date(timeIntervalSince1970: 1_710_004_000)
        )

        let assetURL = location.assetsDirectory.appendingPathComponent("voice-reference.wav", isDirectory: false)
        let exportURL = location.exportsDirectory.appendingPathComponent(export.fileName, isDirectory: false)
        let cacheURL = location.cacheDirectory.appendingPathComponent("render.tmp", isDirectory: false)
        let assetData = Data("VOICE".utf8)
        let exportData = Data("RIFF".utf8)
        let cacheData = Data("CACHE".utf8)
        try assetData.write(to: assetURL, options: .atomic)
        try exportData.write(to: exportURL, options: .atomic)
        try cacheData.write(to: cacheURL, options: .atomic)

        let assetsDirectoryFileNumber = try fileSystemNumber(at: location.assetsDirectory)
        let exportsDirectoryFileNumber = try fileSystemNumber(at: location.exportsDirectory)
        let cacheDirectoryFileNumber = try fileSystemNumber(at: location.cacheDirectory)

        let updatedProject = ProjectRecord(
            id: project.id,
            title: project.title,
            createdAt: project.createdAt,
            updatedAt: Date(timeIntervalSince1970: 1_710_005_000),
            notes: "Updated chapter"
        )
        let updatedSnapshot = ProjectBundleSnapshot(
            project: updatedProject,
            modelID: initialSnapshot.modelID,
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapter],
            renderJobs: [],
            exports: [export]
        )

        _ = try ProjectBundleWriter().write(
            updatedSnapshot,
            to: location,
            createdAt: Date(timeIntervalSince1970: 1_710_006_000)
        )

        let bundle = try ProjectBundleReader().read(from: bundleURL)

        XCTAssertEqual(bundle.snapshot.project, updatedProject)
        XCTAssertEqual(bundle.snapshot.exports, [export])
        XCTAssertEqual(try Data(contentsOf: assetURL), assetData)
        XCTAssertEqual(try Data(contentsOf: exportURL), exportData)
        XCTAssertEqual(try Data(contentsOf: cacheURL), cacheData)
        XCTAssertEqual(try fileSystemNumber(at: location.assetsDirectory), assetsDirectoryFileNumber)
        XCTAssertEqual(try fileSystemNumber(at: location.exportsDirectory), exportsDirectoryFileNumber)
        XCTAssertEqual(try fileSystemNumber(at: location.cacheDirectory), cacheDirectoryFileNumber)
        XCTAssertFalse(FileManager.default.fileExists(atPath: bundleURL.appendingPathExtension("saving").path))
    }

    func testProjectBundleSaveFailureCleansTempAndKeepsOriginalBundle() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let project = ProjectRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            title: "Narrated Chapter One",
            createdAt: Date(timeIntervalSince1970: 1_710_000_000),
            updatedAt: Date(timeIntervalSince1970: 1_710_000_600),
            notes: "Opening chapter"
        )
        let chapter = ChapterRecord(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "A beginning.",
            speakerLabel: "Narrator",
            estimatedDurationSeconds: 12.5
        )
        let export = ExportRecord(
            id: UUID(uuidString: "12345678-90AB-CDEF-1234-567890ABCDEF")!,
            projectID: project.id,
            fileName: "chapter-one.wav",
            createdAt: Date(timeIntervalSince1970: 1_710_003_000),
            checksum: "abc123"
        )
        let bundleURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        let location = ValarProjectBundleLocation(
            projectID: project.id,
            title: project.title,
            bundleURL: bundleURL
        )
        let initialSnapshot = ProjectBundleSnapshot(
            project: project,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapter],
            renderJobs: [],
            exports: [export]
        )

        _ = try ProjectBundleWriter().write(
            initialSnapshot,
            to: location,
            createdAt: Date(timeIntervalSince1970: 1_710_004_000)
        )

        let exportURL = location.exportsDirectory.appendingPathComponent(export.fileName, isDirectory: false)
        let exportData = Data("RIFF".utf8)
        try exportData.write(to: exportURL, options: .atomic)

        let updatedProject = ProjectRecord(
            id: project.id,
            title: project.title,
            createdAt: project.createdAt,
            updatedAt: Date(timeIntervalSince1970: 1_710_005_000),
            notes: "Updated chapter"
        )
        let updatedSnapshot = ProjectBundleSnapshot(
            project: updatedProject,
            modelID: initialSnapshot.modelID,
            renderSynthesisOptions: RenderSynthesisOptions(),
            chapters: [chapter],
            renderJobs: [],
            exports: [export]
        )
        var didAttemptReplace = false

        XCTAssertThrowsError(
            try ProjectBundleWriter(
                bundleCommitter: { _, _, _ in
                    didAttemptReplace = true
                    throw CocoaError(.fileWriteUnknown)
                }
            ).write(
                updatedSnapshot,
                to: location,
                createdAt: Date(timeIntervalSince1970: 1_710_006_000)
            )
        )

        let bundle = try ProjectBundleReader().read(from: bundleURL)

        XCTAssertTrue(didAttemptReplace)
        XCTAssertEqual(bundle.snapshot.project, project)
        XCTAssertEqual(bundle.snapshot.exports, [export])
        XCTAssertEqual(try Data(contentsOf: exportURL), exportData)
        XCTAssertFalse(FileManager.default.fileExists(atPath: bundleURL.appendingPathExtension("saving").path))
    }

    func testSanitizedBundleNameRemovesIllegalSeparators() {
        let paths = ValarAppPaths()
        let bundleName = paths.sanitizeBundleName("A/B: The Rewrite")
        XCTAssertEqual(bundleName, "a-b-the-rewrite")
    }

    func testModelPackPathsLandUnderApplicationSupport() {
        let paths = ValarAppPaths()

        XCTAssertTrue(paths.modelPacksDirectory.path.contains("ValarTTS"))
        XCTAssertEqual(try? paths.modelPackManifestURL(familyID: "qwen3_tts", modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16").lastPathComponent, "manifest.json")
        XCTAssertEqual(try? paths.modelPackAssetsDirectory(familyID: "qwen3_tts", modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16").lastPathComponent, "assets")
        XCTAssertTrue((try? paths.importedModelAssetDirectory(modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16").path.contains("Imports")) ?? false)
        XCTAssertEqual(
            try? paths.voiceAssetURL(voiceID: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!).lastPathComponent,
            "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE.wav"
        )
    }

    func testModelPackRejectsTraversalIdentifier() throws {
        let paths = ValarAppPaths()

        XCTAssertThrowsError(try paths.modelPackDirectory(familyID: "qwen3_tts", modelID: "../escape")) { error in
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .pathTraversalDetected(label: "model identifier", value: "../escape")
            )
        }
    }

    func testVoiceAssetURLAllowlistAcceptsKnownExtensions() throws {
        let paths = ValarAppPaths()
        let id = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "wav").pathExtension, "wav")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "m4a").pathExtension, "m4a")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "MP3").pathExtension, "mp3")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "  aiff  ").pathExtension, "aiff")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "caf").pathExtension, "caf")
    }

    func testVoiceAssetURLAllowlistRejectsUnknownExtensions() throws {
        let paths = ValarAppPaths()
        let id = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "exe").pathExtension, "wav")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "ogg").pathExtension, "wav")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "sh").pathExtension, "wav")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "").pathExtension, "wav")
        XCTAssertEqual(try paths.voiceAssetURL(voiceID: id, fileExtension: "../../../etc/passwd").pathExtension, "wav")
    }

    func testResolveSystemPathsReturnsValidPaths() throws {
        let paths = try ValarAppPaths.resolveSystemPaths()
        XCTAssertTrue(paths.applicationSupport.path.contains("ValarTTS"))
        XCTAssertTrue(paths.modelPacksDirectory.path.contains("ModelPacks"))
        XCTAssertTrue(paths.databaseURL.path.contains("valar.db"))
    }

    func testResolveSystemPathsThrowsWhenDirectoryUnavailable() {
        let emptyManager = EmptyURLsFileManager()
        XCTAssertThrowsError(try ValarAppPaths.resolveSystemPaths(fileManager: emptyManager)) { error in
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .applicationSupportDirectoryUnavailable
            )
        }
    }

    func testDefaultPathsHonorCLIHomeOverride() throws {
        let override = try makeTemporaryDirectory().path
        setenv("VALARTTS_CLI_HOME", override, 1)
        defer { unsetenv("VALARTTS_CLI_HOME") }

        let paths = ValarAppPaths()

        XCTAssertEqual(paths.applicationSupport.path, override)
        XCTAssertEqual(paths.databaseURL.path, "\(override)/valar.db")
    }

    func testValarPathRedactionSanitizesEmbeddedPathsAndFileURLs() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let homeFileURL = URL(fileURLWithPath: home)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("voice.wav", isDirectory: false)
            .absoluteString
        let volumePath = "/Volumes" + "/External/audio.wav"
        let message = "Failed to open \(home)/Secrets/voice.wav from \(volumePath) via \(homeFileURL)"

        let sanitized = ValarPathRedaction.sanitizeMessage(message)

        XCTAssertFalse(sanitized.contains(home))
        XCTAssertFalse(sanitized.contains(volumePath))
        XCTAssertTrue(sanitized.contains("~/Secrets/voice.wav"))
        XCTAssertTrue(sanitized.contains("/Volumes/<volume>/audio.wav"))
        XCTAssertTrue(sanitized.contains("file://~/Library/voice.wav"))
    }

    func testValidateContainmentRejectsSymlinkEscape() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let outside = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: outside) }

        let symlink = root.appendingPathComponent("Projects", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: outside)

        let escaped = symlink.appendingPathComponent("escape.sqlite", isDirectory: false)
        XCTAssertThrowsError(try ValarAppPaths.validateContainment(escaped, within: root)) { error in
            guard case .pathEscapesContainment = error as? ValarPathValidationError else {
                return XCTFail("Expected containment failure, got \(error)")
            }
        }
    }

    func testValidateContainmentRejectsTraversalInAppendedComponent() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let staging = root.appendingPathComponent("staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)

        let escaped = staging.appendingPathComponent("../escape.bin", isDirectory: false)
        XCTAssertThrowsError(try ValarAppPaths.validateContainment(escaped, within: staging)) { error in
            guard case .pathEscapesContainment = error as? ValarPathValidationError else {
                return XCTFail("Expected containment failure, got \(error)")
            }
        }
    }

    func testValidateContainmentAcceptsNestedSubpath() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let staging = root.appendingPathComponent("staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)

        let nested = staging.appendingPathComponent("weights/model.safetensors", isDirectory: false)
        XCTAssertNoThrow(try ValarAppPaths.validateContainment(nested, within: staging))
    }

    func testValidateRelativePathRejectsAbsolutePath() {
        XCTAssertThrowsError(try ValarAppPaths.validateRelativePath("/etc/passwd", label: "test")) { error in
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .absolutePathNotAllowed(label: "test", value: "/etc/passwd")
            )
        }
    }

    func testValidateRelativePathRejectsEmbeddedTraversal() {
        XCTAssertThrowsError(try ValarAppPaths.validateRelativePath("weights/../../escape.bin", label: "artifact")) { error in
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .pathTraversalDetected(label: "artifact", value: "weights/../../escape.bin")
            )
        }
    }

    func testAppDatabaseRejectsSymlinkEscape() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let outside = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: outside) }

        let symlink = root.appendingPathComponent("db-link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: outside)
        let databaseURL = symlink.appendingPathComponent("Valar.sqlite", isDirectory: false)

        XCTAssertThrowsError(
            try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        ) { error in
            guard case .pathEscapesContainment = error as? ValarPathValidationError else {
                return XCTFail("Expected containment failure, got \(error)")
            }
        }
    }

    func testDocumentBundleManifestRoundTrips() throws {
        let manifest = DocumentBundleManifest(title: "Novel")
        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(DocumentBundleManifest.self, from: data)
        XCTAssertEqual(decoded.title, "Novel")
        XCTAssertEqual(decoded.schemaVersion, 1)
    }

    func testModelPackStoreReportsMalformedManifestJSON() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)
        let manifest = ValarPersistence.ModelPackManifest(
            id: "manifest-1",
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speechSynthesis"],
            backendKinds: ["mlx"],
            artifactSpecs: []
        )
        try await store.saveManifest(manifest)

        try await db.writer.write { database in
            try database.execute(
                sql: "UPDATE modelPack SET manifestJSON = ? WHERE id = ?",
                arguments: ["not-json", manifest.id]
            )
        }

        do {
            _ = try await store.manifest(for: manifest.modelID)
            XCTFail("Expected malformed JSON to throw when decoding modelPack.manifestJSON")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("Failed to decode JSON for modelPack.manifestJSON"),
                "Expected modelPack.manifestJSON decode failure, got: \(error.localizedDescription)"
            )
        }
    }

    func testRenderJobStoreReportsMalformedChapterIDsJSON() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)
        let job = RenderJobRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            projectID: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            chapterIDs: [UUID(uuidString: "66666666-7777-8888-9999-AAAAAAAAAAAA")!],
            outputFileName: "chapter.wav"
        )
        try await store.save(job)

        try await db.writer.write { database in
            try database.execute(
                sql: "UPDATE renderJob SET chapterIDs = ? WHERE id = ?",
                arguments: ["not-json", job.id.uuidString]
            )
        }

        do {
            _ = try await store.job(id: job.id)
            XCTFail("Expected malformed JSON to throw when decoding renderJob.chapterIDs")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("Failed to decode JSON for renderJob.chapterIDs"),
                "Expected renderJob.chapterIDs decode failure, got: \(error.localizedDescription)"
            )
        }
    }

    func testMigrationLedgerCapturesStep() async {
        let ledger = MigrationLedger()
        _ = await ledger.record(sourcePath: "/tmp/legacy", note: "Imported")
        let steps = await ledger.steps()
        XCTAssertEqual(steps.count, 1)
        XCTAssertEqual(steps.first?.message, "Imported")
    }

    func testModelPackManifestAndReceiptRoundTrip() throws {
        let artifact = ModelPackArtifact(
            id: "weights",
            kind: "weights",
            relativePath: "weights/model.safetensors",
            checksum: "abc123",
            byteCount: 42
        )
        let manifest = ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis", "voice.cloning"],
            backendKinds: ["mlx"],
            tokenizerType: "qwen3_tts_tokenizer_12hz",
            sampleRate: 24_000,
            artifactSpecs: [artifact],
            licenseName: "MIT",
            licenseURL: "https://example.com/license"
        )
        let receipt = ModelInstallReceipt(
            modelID: manifest.modelID,
            familyID: manifest.familyID,
            sourceKind: .localFile,
            sourceLocation: "/tmp/Qwen3TTS.valarmodel",
            installedModelPath: "/Library/Application Support/ValarTTS/ModelPacks/qwen3_tts/model",
            manifestPath: "/Library/Application Support/ValarTTS/ModelPacks/qwen3_tts/model/manifest.json",
            checksum: artifact.checksum,
            artifactCount: manifest.artifactSpecs.count,
            notes: "Installed locally"
        )

        let manifestData = try JSONEncoder().encode(manifest)
        let decodedManifest = try JSONDecoder().decode(ValarPersistence.ModelPackManifest.self, from: manifestData)
        XCTAssertEqual(decodedManifest.modelID, manifest.modelID)
        XCTAssertEqual(decodedManifest.familyID, "qwen3_tts")

        let receiptData = try JSONEncoder().encode(receipt)
        let decodedReceipt = try JSONDecoder().decode(ModelInstallReceipt.self, from: receiptData)
        XCTAssertEqual(decodedReceipt.modelID, manifest.modelID)
        XCTAssertEqual(decodedReceipt.artifactCount, 1)
    }

    func testModelPackRegistryTracksSupportedAndInstalledModels() async throws {
        let registry = ModelPackRegistry()
        let supported = SupportedModelCatalogRecord(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
                providerName: "Valar",
            providerURL: "https://huggingface.co",
            installHint: "Import a local pack or open the provider link.",
            sourceKind: .remoteURL,
            isRecommended: true
        )

        await registry.registerSupported(supported)
        let catalog = await registry.supportedModels()
        XCTAssertEqual(catalog.first?.modelID, supported.modelID)

        let manifest = ModelPackManifest(
            familyID: supported.familyID,
            modelID: supported.modelID,
            displayName: supported.displayName,
            capabilities: ["speech.synthesis", "voice.cloning"],
            backendKinds: ["mlx"],
            tokenizerType: "qwen3_tts_tokenizer_12hz",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(id: "weights", kind: "weights", relativePath: "weights/model.safetensors")
            ],
            licenseName: "MIT",
            licenseURL: "https://example.com/license",
            minimumAppVersion: "1.0.0"
        )

        let receipt = try await registry.install(
            manifest: manifest,
            sourceKind: .localFile,
            sourceLocation: "/tmp/Qwen3TTS.valarmodel"
        )

        let installed = await registry.installedRecord(for: manifest.modelID)
        XCTAssertEqual(installed?.modelID, manifest.modelID)
        XCTAssertEqual(installed?.artifactCount, 1)
        XCTAssertEqual(receipt.manifestPath.hasSuffix("manifest.json"), true)

        let receipts = await registry.receipts()
        XCTAssertEqual(receipts.count, 1)
        let ledger = await registry.ledgerEntries()
        XCTAssertEqual(ledger.count, 1)
        XCTAssertTrue(ledger.first?.succeeded ?? false)
    }

    // MARK: - GRDB Store Tests

    func testAppDatabaseCreatesInMemory() throws {
        let db = try AppDatabase.inMemory()
        XCTAssertNotNil(db)
    }

    func testGRDBProjectStoreCRUD() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)

        let project = try await store.insert(title: "Test Project", notes: "Some notes")
        XCTAssertEqual(project.title, "Test Project")
        XCTAssertEqual(project.notes, "Some notes")

        let fetched = try await store.project(id: project.id)
        XCTAssertEqual(fetched?.title, "Test Project")

        let all = try await store.fetchAll()
        XCTAssertEqual(all.count, 1)

        var updated = project
        updated.title = "Updated Title"
        try await store.update(updated)
        let refetched = try await store.project(id: project.id)
        XCTAssertEqual(refetched?.title, "Updated Title")

        try await store.delete(project.id)
        let gone = try await store.project(id: project.id)
        XCTAssertNil(gone)
    }

    func testGRDBProjectStorePersistsAcrossDatabaseReopen() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = root.appendingPathComponent("Valar.sqlite", isDirectory: false)
        let writer = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let initialStore = GRDBProjectStore(db: writer, paths: ValarAppPaths(baseURL: root))

        let created = try await initialStore.insert(title: "Persistent Project", notes: "Reopen me")
        XCTAssertEqual(created.title, "Persistent Project")

        let reopened = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let reopenedStore = GRDBProjectStore(db: reopened, paths: ValarAppPaths(baseURL: root))
        let projects = try await reopenedStore.fetchAll()

        XCTAssertEqual(projects.count, 1)
        XCTAssertEqual(projects.first?.id, created.id)
        XCTAssertEqual(projects.first?.title, "Persistent Project")
        XCTAssertEqual(projects.first?.notes, "Reopen me")
    }

    func testGRDBProjectStorePersistsChapterSpeechMetadataAcrossDatabaseReopen() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = root.appendingPathComponent("Valar.sqlite", isDirectory: false)
        let writer = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let initialStore = GRDBProjectStore(db: writer, paths: ValarAppPaths(baseURL: root))

        let project = try await initialStore.insert(title: "Speech Project")
        let chapter = ChapterRecord(
            projectID: project.id,
            index: 0,
            title: "Chapter 1",
            script: "A beginning."
        )
        try await initialStore.insert(chapter)
        try await initialStore.attachAudio(
            to: chapter.id,
            in: project.id,
            assetName: "chapter-1.wav",
            sampleRate: 24_000,
            durationSeconds: 11.2
        )
        try await initialStore.setTranscription(
            for: chapter.id,
            in: project.id,
            transcriptionJSON: #"{"segments":[{"text":"A beginning."}]}"#,
            modelID: "whisper-large-v3"
        )
        try await initialStore.setAlignment(
            for: chapter.id,
            in: project.id,
            alignmentJSON: #"{"tokens":[{"start":0.0,"end":0.8}]}"#,
            modelID: "ctc-aligner-v1"
        )

        let translatedChapterCandidate = try await initialStore.chapters(for: project.id).first
        var translatedChapter = try XCTUnwrap(translatedChapterCandidate)
        translatedChapter.derivedTranslationText = "Un commencement."
        try await initialStore.update(translatedChapter)

        let reopened = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let reopenedStore = GRDBProjectStore(db: reopened, paths: ValarAppPaths(baseURL: root))
        let reloadedChapterCandidate = try await reopenedStore.chapters(for: project.id).first
        let reloadedChapter = try XCTUnwrap(reloadedChapterCandidate)

        XCTAssertEqual(reloadedChapter.sourceAudioAssetName, "chapter-1.wav")
        XCTAssertEqual(reloadedChapter.sourceAudioSampleRate, 24_000)
        XCTAssertEqual(reloadedChapter.sourceAudioDurationSeconds, 11.2)
        XCTAssertEqual(reloadedChapter.transcriptionJSON, #"{"segments":[{"text":"A beginning."}]}"#)
        XCTAssertEqual(reloadedChapter.transcriptionModelID, "whisper-large-v3")
        XCTAssertEqual(reloadedChapter.alignmentJSON, #"{"tokens":[{"start":0.0,"end":0.8}]}"#)
        XCTAssertEqual(reloadedChapter.alignmentModelID, "ctc-aligner-v1")
        XCTAssertEqual(reloadedChapter.derivedTranslationText, "Un commencement.")
    }

    func testGRDBProjectStoreRejectsTraversalTitle() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)

        do {
            _ = try await store.create(title: "../Secrets", notes: nil)
            XCTFail("Expected traversal title to be rejected")
        } catch {
            XCTAssertEqual(
                error as? ValarPathValidationError,
                .pathTraversalDetected(label: "project title", value: "../Secrets")
            )
        }
    }

    func testGRDBProjectStoreBundleLocation() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBProjectStore(db: db)
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let project = try await store.insert(title: "Narrated Chapter One")
        let documentURL = root
            .appendingPathComponent("Narrated Chapter One", isDirectory: true)
            .appendingPathExtension("valarproject")
        await store.updateBundleURL(documentURL, for: project.id)
        let location = try await store.bundleLocation(for: project.id)

        XCTAssertEqual(location?.bundleURL.pathExtension, "valarproject")
        XCTAssertEqual(location?.manifestURL.lastPathComponent, "manifest.json")
        XCTAssertEqual(location?.sqliteURL.pathExtension, "sqlite")
    }

    func testVoiceStorePersistsCloneMetadata() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)
        let embedding = Data([0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40])
        let record = VoiceLibraryRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            label: "Studio Clone",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            runtimeModelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            sourceAssetName: "reference.m4a",
            referenceAudioAssetName: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE.wav",
            referenceTranscript: "Judge Victor quickly mixed a beige waxy potion.",
            referenceDurationSeconds: 8.4,
            referenceSampleRate: 24_000,
            referenceChannelCount: 2,
            speakerEmbedding: embedding,
            createdAt: Date(timeIntervalSince1970: 1_710_000_000)
        )

        _ = try await store.save(record)
        let voices = try await store.fetchAll()

        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(voices.first, record)
        XCTAssertTrue(voices.first?.isClonedVoice == true)
        XCTAssertEqual(voices.first?.speakerEmbedding, embedding)
        XCTAssertEqual(
            voices.first?.conditioningFormat,
            VoiceLibraryRecord.qwenSpeakerEmbeddingConditioningFormat
        )
    }

    func testVoiceStorePersistsDesignPrompt() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)
        let record = VoiceLibraryRecord(
            label: "British Guide",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            voicePrompt: "warm female voice, British accent, mid-30s"
        )

        _ = try await store.save(record)
        let voices = try await store.fetchAll()

        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(voices.first?.voicePrompt, "warm female voice, British accent, mid-30s")
        XCTAssertEqual(voices.first?.label, "British Guide")
    }

    func testVoiceStorePersistsBackendVoiceID() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)
        let record = VoiceLibraryRecord(
            label: "Neutral Female",
            modelID: "mistralai/Voxtral-4B-TTS-2603",
            runtimeModelID: "mistralai/Voxtral-4B-TTS-2603",
            backendVoiceID: "neutral_female"
        )

        _ = try await store.save(record)
        let voices = try await store.fetchAll()

        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(voices.first?.backendVoiceID, "neutral_female")
        XCTAssertTrue(voices.first?.isModelDeclaredPreset == true)
        XCTAssertFalse(voices.first?.isMutable ?? true)
    }

    func testQwenBackendVoiceIDInfersNamedSpeakerKind() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)
        let record = VoiceLibraryRecord(
            label: "Cherry",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            runtimeModelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            backendVoiceID: "Cherry"
        )

        _ = try await store.save(record)
        let voices = try await store.fetchAll()

        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(voices.first?.effectiveVoiceKind, "namedSpeaker")
        XCTAssertEqual(voices.first?.typeDisplayName, "Named Speaker")
    }

    func testGRDBVoiceStoreCRUD() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)

        let voice = try await store.insert(VoiceLibraryRecord(label: "Narrator", modelID: "qwen3-tts", sourceAssetName: "narrator.wav"))
        XCTAssertEqual(voice.label, "Narrator")
        XCTAssertEqual(voice.modelID, "qwen3-tts")

        let voices = try await store.fetchAll()
        XCTAssertEqual(voices.count, 1)

        let fetched = try await store.voice(id: voice.id)
        XCTAssertEqual(fetched?.label, "Narrator")

        try await store.delete(voice.id)
        let gone = try await store.voice(id: voice.id)
        XCTAssertNil(gone)
    }

    func testGRDBVoiceStorePersistsAcrossDatabaseReopen() async throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let databaseURL = root.appendingPathComponent("Valar.sqlite", isDirectory: false)
        let writer = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let initialStore = GRDBVoiceStore(db: writer)
        let created = try await initialStore.insert(
            VoiceLibraryRecord(
                label: "Persistent Narrator",
                modelID: "qwen3-tts",
                voicePrompt: "calm documentary narration"
            )
        )

        let reopened = try AppDatabase(path: databaseURL.path, allowedDirectories: [root])
        let reopenedStore = GRDBVoiceStore(db: reopened)
        let voices = try await reopenedStore.fetchAll()

        XCTAssertEqual(voices.count, 1)
        XCTAssertEqual(voices.first?.id, created.id)
        XCTAssertEqual(voices.first?.label, "Persistent Narrator")
        XCTAssertEqual(voices.first?.voicePrompt, "calm documentary narration")
    }

    func testGRDBRenderJobStoreCRUD() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBRenderJobStore(db: db)

        let projectID = UUID()
        let chapterID = UUID()
        let job = RenderJobRecord(
            projectID: projectID,
            chapterIDs: [chapterID],
            outputFileName: "output.wav"
        )
        try await store.save(job)

        let jobs = try await store.loadJobs()
        XCTAssertEqual(jobs.count, 1)
        XCTAssertEqual(jobs.first?.outputFileName, "output.wav")
        XCTAssertEqual(jobs.first?.chapterIDs, [chapterID])

        let fetched = try await store.job(id: job.id)
        XCTAssertEqual(fetched?.projectID, projectID)

        let projectJobs = try await store.jobs(forProject: projectID)
        XCTAssertEqual(projectJobs.count, 1)

        try await store.remove(id: job.id)
        let gone = try await store.job(id: job.id)
        XCTAssertNil(gone)
    }

    func testGRDBModelPackStoreManifestRoundTrip() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let manifest = ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis", "voice.cloning"],
            backendKinds: ["mlx"],
            tokenizerType: "qwen3_tts_tokenizer_12hz",
            sampleRate: 24_000,
            artifactSpecs: [
                ModelPackArtifact(id: "weights", kind: "weights", relativePath: "weights/model.safetensors", checksum: "abc123", byteCount: 42)
            ],
            licenseName: "MIT",
            licenseURL: "https://example.com/license",
            minimumAppVersion: "1.0.0"
        )

        try await store.saveManifest(manifest)
        let fetched = try await store.manifest(for: manifest.modelID)
        XCTAssertEqual(fetched?.modelID, manifest.modelID)
        XCTAssertEqual(fetched?.familyID, "qwen3_tts")
        XCTAssertEqual(fetched?.capabilities, ["speech.synthesis", "voice.cloning"])
        XCTAssertEqual(fetched?.artifactSpecs.count, 1)
        XCTAssertEqual(fetched?.artifactSpecs.first?.checksum, "abc123")
        XCTAssertEqual(fetched?.sampleRate, 24_000)
    }

    func testGRDBModelPackStoreCatalogAndInstall() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let catalogEntry = SupportedModelCatalogRecord(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            providerName: "Valar",
            providerURL: "https://huggingface.co",
            installHint: "Import a local pack",
            sourceKind: .remoteURL,
            isRecommended: true
        )
        try await store.saveCatalogEntry(catalogEntry)

        let supported = try await store.supportedModels()
        XCTAssertEqual(supported.count, 1)
        XCTAssertEqual(supported.first?.isRecommended, true)

        let found = try await store.supportedModel(for: catalogEntry.modelID)
        XCTAssertEqual(found?.displayName, "Qwen3 TTS Base")

        let installed = InstalledModelRecord(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            installedPath: "/path/to/model",
            manifestPath: "/path/to/manifest.json",
            artifactCount: 1,
            sourceKind: .localFile
        )
        try await store.saveInstalledRecord(installed)

        let fetchedInstalled = try await store.installedRecord(for: installed.modelID)
        XCTAssertEqual(fetchedInstalled?.artifactCount, 1)

        let receipt = ModelInstallReceipt(
            modelID: installed.modelID,
            familyID: "qwen3_tts",
            sourceKind: .localFile,
            sourceLocation: "/tmp/model.valarmodel",
            installedModelPath: "/path/to/model",
            manifestPath: "/path/to/manifest.json",
            artifactCount: 1
        )
        try await store.saveReceipt(receipt)

        let receipts = try await store.receipts()
        XCTAssertEqual(receipts.count, 1)

        let ledgerEntry = ModelInstallLedgerEntry(
            receiptID: receipt.id,
            sourceKind: .localFile,
            sourceLocation: "/tmp/model.valarmodel",
            succeeded: true,
            message: "Installed successfully"
        )
        try await store.saveLedgerEntry(ledgerEntry)

        let ledger = try await store.ledgerEntries()
        XCTAssertEqual(ledger.count, 1)
        XCTAssertTrue(ledger.first?.succeeded ?? false)
    }

    func testGRDBModelPackStoreUninstallRemovesInstalledArtifactsMetadata() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBModelPackStore(db: db)

        let manifest = ModelPackManifest(
            familyID: "qwen3_tts",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 TTS Base",
            capabilities: ["speech.synthesis"],
            backendKinds: ["mlx"],
            artifactSpecs: [
                ModelPackArtifact(id: "weights", kind: "weights", relativePath: "weights/model.safetensors")
            ],
            licenseName: "MIT",
            licenseURL: "https://example.com/license"
        )
        let installed = InstalledModelRecord(
            familyID: manifest.familyID,
            modelID: manifest.modelID,
            displayName: manifest.displayName,
            installedPath: "/path/to/model",
            manifestPath: "/path/to/manifest.json",
            artifactCount: 1,
            sourceKind: .remoteURL
        )
        let receipt = ModelInstallReceipt(
            modelID: manifest.modelID,
            familyID: manifest.familyID,
            sourceKind: .remoteURL,
            sourceLocation: "https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            installedModelPath: installed.installedPath,
            manifestPath: installed.manifestPath,
            artifactCount: installed.artifactCount
        )

        try await store.saveManifest(manifest)
        try await store.saveInstalledRecord(installed)
        try await store.saveReceipt(receipt)

        let removed = try await store.uninstall(modelID: manifest.modelID)

        XCTAssertEqual(removed?.modelID, manifest.modelID)
        let manifestAfter = try await store.manifest(for: manifest.modelID)
        let installedAfter = try await store.installedRecord(for: manifest.modelID)
        let receiptsAfter = try await store.receipts()
        XCTAssertNil(manifestAfter)
        XCTAssertNil(installedAfter)
        XCTAssertTrue(receiptsAfter.isEmpty)
    }
}

private class EmptyURLsFileManager: FileManager {
    override func urls(
        for directory: FileManager.SearchPathDirectory,
        in domainMask: FileManager.SearchPathDomainMask
    ) -> [URL] {
        []
    }
}
