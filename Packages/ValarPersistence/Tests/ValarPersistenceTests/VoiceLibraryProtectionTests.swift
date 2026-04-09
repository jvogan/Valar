import XCTest
import GRDB
import ValarModelKit
@testable import ValarPersistence

final class VoiceLibraryProtectionTests: XCTestCase {
    private static let testKeyEnvironmentVariable = "VALARTTS_TEST_VOICE_KEY_B64"
    private static let testKeyBase64 = Data((0 ..< 32).map(UInt8.init)).base64EncodedString()
    private static let keyPathEnvironmentVariable = "VALARTTS_VOICE_KEY_PATH"
    private static let disableKeychainEnvironmentVariable = "VALARTTS_VOICE_KEY_DISABLE_KEYCHAIN"

    override func setUp() {
        super.setUp()
        setenv(Self.testKeyEnvironmentVariable, Self.testKeyBase64, 1)
    }

    override func tearDown() {
        unsetenv(Self.testKeyEnvironmentVariable)
        super.tearDown()
    }

    func testProtectRoundTripsAndLegacyPlaintextStillLoads() throws {
        let plaintext = Data("voice-material".utf8)

        let protected = try VoiceLibraryProtection.protect(plaintext)

        XCTAssertTrue(VoiceLibraryProtection.isProtected(protected))
        XCTAssertNotEqual(protected, plaintext)
        XCTAssertEqual(try VoiceLibraryProtection.unprotectIfNeeded(protected), plaintext)
        XCTAssertEqual(try VoiceLibraryProtection.unprotectIfNeeded(plaintext), plaintext)
    }

    func testProtectFallsBackToFileKeyWhenKeychainIsDisabled() throws {
        unsetenv(Self.testKeyEnvironmentVariable)
        let keyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
        setenv(Self.keyPathEnvironmentVariable, keyURL.path, 1)
        setenv(Self.disableKeychainEnvironmentVariable, "1", 1)
        defer {
            unsetenv(Self.keyPathEnvironmentVariable)
            unsetenv(Self.disableKeychainEnvironmentVariable)
            setenv(Self.testKeyEnvironmentVariable, Self.testKeyBase64, 1)
            try? FileManager.default.removeItem(at: keyURL)
        }

        let plaintext = Data("daemon-friendly-voice-material".utf8)
        let protected = try VoiceLibraryProtection.protect(plaintext)

        XCTAssertTrue(FileManager.default.fileExists(atPath: keyURL.path))
        XCTAssertEqual(try Data(contentsOf: keyURL).count, 32)
        XCTAssertEqual(try VoiceLibraryProtection.unprotectIfNeeded(protected), plaintext)
    }

    func testProtectPrefersExistingFallbackKeyWithoutDisableFlag() throws {
        unsetenv(Self.testKeyEnvironmentVariable)
        let keyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
        let expectedKey = Data((0 ..< 32).map { UInt8(($0 * 7) % 256) })
        setenv(Self.keyPathEnvironmentVariable, keyURL.path, 1)
        defer {
            unsetenv(Self.keyPathEnvironmentVariable)
            setenv(Self.testKeyEnvironmentVariable, Self.testKeyBase64, 1)
            try? FileManager.default.removeItem(at: keyURL)
        }

        try expectedKey.write(to: keyURL, options: .atomic)
        let plaintext = Data("prefer-existing-fallback-key".utf8)
        let protected = try VoiceLibraryProtection.protect(plaintext)

        XCTAssertEqual(try Data(contentsOf: keyURL), expectedKey)
        XCTAssertEqual(try VoiceLibraryProtection.unprotectIfNeeded(protected), plaintext)
    }

    func testVoiceStoreEncryptsSpeakerEmbeddingAtRestAndRewritesLegacyPlaintext() async throws {
        let db = try AppDatabase.inMemory()
        let store = GRDBVoiceStore(db: db)
        let embedding = Data([0xDE, 0xAD, 0xBE, 0xEF])

        let record = VoiceLibraryRecord(
            id: UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!,
            label: "Encrypted Clone",
            modelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            runtimeModelID: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            speakerEmbedding: embedding
        )

        _ = try await store.save(record)

        let storedCiphertext = try await rawSpeakerEmbedding(in: db, id: record.id)
        let protectedEmbedding = try XCTUnwrap(storedCiphertext)
        XCTAssertNotNil(storedCiphertext)
        XCTAssertNotEqual(storedCiphertext, embedding)
        XCTAssertTrue(VoiceLibraryProtection.isProtected(protectedEmbedding))

        let fetched = try await store.voice(id: record.id)
        XCTAssertEqual(fetched?.speakerEmbedding, embedding)

        let legacyID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!
        try await db.writer.write { database in
            try database.execute(
                sql: """
                INSERT INTO voice (
                    id, label, modelID, runtimeModelID, speakerEmbedding, createdAt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    legacyID.uuidString,
                    "Legacy Clone",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                    embedding,
                    "2026-03-28T00:00:00Z",
                ]
            )
        }

        let legacyFetched = try await store.voice(id: legacyID)
        XCTAssertEqual(legacyFetched?.speakerEmbedding, embedding)

        _ = try await store.save(XCTUnwrap(legacyFetched))

        let upgradedCiphertext = try await rawSpeakerEmbedding(in: db, id: legacyID)
        let upgradedEmbedding = try XCTUnwrap(upgradedCiphertext)
        XCTAssertNotNil(upgradedCiphertext)
        XCTAssertNotEqual(upgradedCiphertext, embedding)
        XCTAssertTrue(VoiceLibraryProtection.isProtected(upgradedEmbedding))
    }

    func testTADABundleWritesProtectedFilesAndLoadsLegacyPlaintext() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let encryptedBundleURL = root.appendingPathComponent("encrypted-bundle", isDirectory: true)
        let conditioning = makeTADAConditioning()
        _ = try TADAConditioningBundleIO.write(conditioning: conditioning, to: encryptedBundleURL)

        let protectedManifest = try Data(contentsOf: encryptedBundleURL.appendingPathComponent(TADAConditioningBundleIO.manifestFilename))
        XCTAssertTrue(VoiceLibraryProtection.isProtected(protectedManifest))

        for asset in conditioning.assetFiles {
            let storedAsset = try Data(contentsOf: encryptedBundleURL.appendingPathComponent(asset.filename))
            XCTAssertTrue(VoiceLibraryProtection.isProtected(storedAsset))
            XCTAssertNotEqual(storedAsset, asset.data)
        }

        let encryptedLoaded = try TADAConditioningBundleIO.load(from: encryptedBundleURL)
        XCTAssertEqual(encryptedLoaded.format, VoiceConditioning.tadaReferenceV1)
        XCTAssertEqual(encryptedLoaded.assetFiles, conditioning.assetFiles)
        XCTAssertEqual(encryptedLoaded.metadata, conditioning.metadata)
        XCTAssertEqual(encryptedLoaded.sourceModel, ModelIdentifier("HumeAI/mlx-tada-3b"))

        let legacyBundleURL = root.appendingPathComponent("legacy-bundle", isDirectory: true)
        try FileManager.default.createDirectory(at: legacyBundleURL, withIntermediateDirectories: true)

        let manifest = TADAConditioningManifest(metadata: conditioning.metadata)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(XCTUnwrap(manifest)).write(
            to: legacyBundleURL.appendingPathComponent(TADAConditioningBundleIO.manifestFilename),
            options: .atomic
        )
        for asset in conditioning.assetFiles {
            try asset.data.write(
                to: legacyBundleURL.appendingPathComponent(asset.filename),
                options: .atomic
            )
        }

        let legacyLoaded = try TADAConditioningBundleIO.load(from: legacyBundleURL)
        XCTAssertEqual(legacyLoaded.assetFiles, conditioning.assetFiles)
        XCTAssertEqual(legacyLoaded.metadata, conditioning.metadata)
        XCTAssertEqual(legacyLoaded.sourceModel, ModelIdentifier("HumeAI/mlx-tada-3b"))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func rawSpeakerEmbedding(in db: AppDatabase, id: UUID) async throws -> Data? {
        try await db.reader.read { database in
            guard let row = try Row.fetchOne(
                database,
                sql: "SELECT speakerEmbedding FROM voice WHERE id = ?",
                arguments: [id.uuidString]
            ) else {
                return nil
            }
            let stored: Data? = row["speakerEmbedding"]
            return stored
        }
    }

    // MARK: - Adversarial tests

    func testDecryptWithWrongKeyFails() throws {
        let plaintext = Data("sensitive-voice-material".utf8)
        let protectedData = try VoiceLibraryProtection.protect(plaintext)

        // Swap to a different key (bytes 1–32 instead of 0–31).
        let wrongKeyBase64 = Data((1 ... 32).map(UInt8.init)).base64EncodedString()
        setenv(Self.testKeyEnvironmentVariable, wrongKeyBase64, 1)
        defer { setenv(Self.testKeyEnvironmentVariable, Self.testKeyBase64, 1) }

        XCTAssertThrowsError(try VoiceLibraryProtection.unprotectIfNeeded(protectedData)) { error in
            guard let protectionError = error as? VoiceLibraryProtectionError,
                  case .decryptionFailed = protectionError else {
                XCTFail("Expected VoiceLibraryProtectionError.decryptionFailed, got \(error)")
                return
            }
        }
    }

    func testTamperedCiphertextFails() throws {
        let plaintext = Data("tamper-me".utf8)
        var protectedData = try VoiceLibraryProtection.protect(plaintext)

        // Flip a byte well into the ciphertext region (past the magic prefix and JSON preamble).
        let flipIndex = protectedData.index(protectedData.startIndex, offsetBy: protectedData.count / 2)
        protectedData[flipIndex] ^= 0xFF

        XCTAssertThrowsError(try VoiceLibraryProtection.unprotectIfNeeded(protectedData)) { error in
            guard let protectionError = error as? VoiceLibraryProtectionError else {
                XCTFail("Expected VoiceLibraryProtectionError, got \(error)")
                return
            }
            switch protectionError {
            case .decryptionFailed, .malformedEnvelope:
                break // Both are acceptable outcomes for tampered data.
            default:
                XCTFail("Unexpected VoiceLibraryProtectionError case: \(protectionError)")
            }
        }
    }

    func testMalformedJSONEnvelopeThrowsMalformedError() throws {
        // Build a blob that starts with the magic prefix but has garbage JSON after it.
        let magicPrefix = Data("VALARSEC1\n".utf8)
        let garbage = Data("{\"version\":1,\"algorithm\":\"AES.GCM\",\"nonce\":\"!!!\"}".utf8)
        let malformed = magicPrefix + garbage

        XCTAssertTrue(VoiceLibraryProtection.isProtected(malformed))
        XCTAssertThrowsError(try VoiceLibraryProtection.unprotectIfNeeded(malformed)) { error in
            guard let protectionError = error as? VoiceLibraryProtectionError,
                  case .malformedEnvelope = protectionError else {
                XCTFail("Expected VoiceLibraryProtectionError.malformedEnvelope, got \(error)")
                return
            }
        }
    }

    func testUnsupportedEnvelopeVersionThrowsError() throws {
        // Construct a valid-looking envelope JSON with version 99.
        let magicPrefix = Data("VALARSEC1\n".utf8)
        let fakeEnvelope = """
        {"version":99,"algorithm":"AES.GCM","nonce":"AAAAAAAAAAAAAAAAAAAAAA==","ciphertext":"dGVzdA==","tag":"AAAAAAAAAAAAAAAAAAAAAA=="}
        """
        let blob = magicPrefix + Data(fakeEnvelope.utf8)

        XCTAssertTrue(VoiceLibraryProtection.isProtected(blob))
        XCTAssertThrowsError(try VoiceLibraryProtection.unprotectIfNeeded(blob)) { error in
            guard let protectionError = error as? VoiceLibraryProtectionError,
                  case .unsupportedEnvelopeVersion(let version) = protectionError else {
                XCTFail("Expected VoiceLibraryProtectionError.unsupportedEnvelopeVersion, got \(error)")
                return
            }
            XCTAssertEqual(version, 99)
        }
    }

    func testProtectIsIdempotentOnAlreadyProtectedData() throws {
        let plaintext = Data("idempotent-check".utf8)
        let protectedOnce = try VoiceLibraryProtection.protect(plaintext)

        XCTAssertTrue(VoiceLibraryProtection.isProtected(protectedOnce))

        let protectedTwice = try VoiceLibraryProtection.protect(protectedOnce)

        // The second protect() call must return the data unchanged.
        XCTAssertEqual(protectedOnce, protectedTwice)

        // And it must still decrypt to the original plaintext.
        XCTAssertEqual(try VoiceLibraryProtection.unprotectIfNeeded(protectedTwice), plaintext)
    }

    private func makeTADAConditioning() -> VoiceConditioning {
        VoiceConditioning.tadaReference(
            assetFiles: [
                VoiceConditioningAssetFile(
                    filename: TADAConditioningBundleIO.tokenValuesFilename,
                    data: Data([0x01, 0x02, 0x03, 0x04])
                ),
                VoiceConditioningAssetFile(
                    filename: TADAConditioningBundleIO.tokenPositionsFilename,
                    data: Data([0x05, 0x06, 0x07, 0x08])
                ),
                VoiceConditioningAssetFile(
                    filename: TADAConditioningBundleIO.textTokensFilename,
                    data: Data([0x09, 0x0A, 0x0B, 0x0C])
                ),
                VoiceConditioningAssetFile(
                    filename: TADAConditioningBundleIO.tokenMasksFilename,
                    data: Data([0x01, 0x00, 0x01, 0x00])
                ),
            ],
            assetName: "fixture-bundle",
            sourceModel: ModelIdentifier("HumeAI/mlx-tada-3b"),
            metadata: VoiceConditioningMetadata(
                modelID: "HumeAI/mlx-tada-3b",
                transcript: "A compact multilingual clone fixture.",
                language: "en",
                sampleRate: 24_000,
                tokenCount: 4,
                acousticDimension: 512,
                frameCount: 4
            )
        )
    }
}
