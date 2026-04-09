import CryptoKit
import Foundation
import Security

public enum VoiceLibraryProtectionError: LocalizedError {
    case invalidKeyMaterial
    case malformedEnvelope
    case unsupportedEnvelopeVersion(Int)
    case keychainFailure(OSStatus, String)
    case decryptionFailed

    public var errorDescription: String? {
        switch self {
        case .invalidKeyMaterial:
            return "Voice library protection key is invalid."
        case .malformedEnvelope:
            return "Stored voice data is encrypted but malformed."
        case .unsupportedEnvelopeVersion(let version):
            return "Stored voice data uses unsupported encryption version \(version)."
        case .keychainFailure(let status, let operation):
            return "Voice library keychain \(operation) failed (\(status))."
        case .decryptionFailed:
            return "Stored voice data could not be decrypted."
        }
    }
}

private struct VoiceLibraryEncryptedEnvelope: Codable, Sendable, Equatable {
    let version: Int
    let algorithm: String
    let nonce: Data
    let ciphertext: Data
    let tag: Data
}

private enum VoiceLibraryProtectionKeychain {
    static let service = "com.valar.tts.voice-library"
    static let account = "default"
    static let keyPathEnvironmentVariable = "VALARTTS_VOICE_KEY_PATH"
    static let disableKeychainEnvironmentVariable = "VALARTTS_VOICE_KEY_DISABLE_KEYCHAIN"
    static let forceKeychainEnvironmentVariable = "VALARTTS_VOICE_KEY_FORCE_KEYCHAIN"
    static let fallbackKeyFilename = "voice-library.key"

    static func symmetricKey() throws -> SymmetricKey {
        let environment = ProcessInfo.processInfo.environment
        #if DEBUG
        if let override = environment["VALARTTS_TEST_VOICE_KEY_B64"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty {
            guard let data = Data(base64Encoded: override), data.count == 32 else {
                throw VoiceLibraryProtectionError.invalidKeyMaterial
            }
            return SymmetricKey(data: data)
        }
        #endif

        let fallbackURL = fallbackKeyURL(environment: environment)
        if let fallback = try readFallbackKeyData(from: fallbackURL) {
            guard fallback.count == 32 else {
                throw VoiceLibraryProtectionError.invalidKeyMaterial
            }
            if shouldUseKeychain(environment: environment) {
                try? writeKeyData(fallback)
            }
            return SymmetricKey(data: fallback)
        }

        if shouldUseKeychain(environment: environment) == false {
            let generated = Data((0 ..< 32).map { _ in UInt8.random(in: 0 ... 255) })
            try writeFallbackKeyData(generated, to: fallbackURL)
            return SymmetricKey(data: generated)
        }

        if let existing = try readKeyData() {
            guard existing.count == 32 else {
                throw VoiceLibraryProtectionError.invalidKeyMaterial
            }
            // Fallback file is a convenience mirror — failure here must not discard the
            // successfully-read keychain key, so we use try? intentionally.
            try? writeFallbackKeyData(existing, to: fallbackURL)
            return SymmetricKey(data: existing)
        }

        let generated = Data((0 ..< 32).map { _ in UInt8.random(in: 0 ... 255) })
        do {
            try writeKeyData(generated)
        } catch VoiceLibraryProtectionError.keychainFailure(let status, _)
                where status == errSecDuplicateItem {
            // Another process beat us to SecItemAdd — read back the winner's key instead
            // of overwriting it, so both processes end up with the same key.
            if let winner = try readKeyData(), winner.count == 32 {
                try? writeFallbackKeyData(winner, to: fallbackURL)
                return SymmetricKey(data: winner)
            }
            // Read-back also failed; fall through and use the locally-generated key.
        } catch {
            try writeFallbackKeyData(generated, to: fallbackURL)
            return SymmetricKey(data: generated)
        }
        try? writeFallbackKeyData(generated, to: fallbackURL)
        return SymmetricKey(data: generated)
    }

    private static func shouldUseKeychain(environment: [String: String]) -> Bool {
        if environment[disableKeychainEnvironmentVariable] == "1" {
            return false
        }
        if environment[forceKeychainEnvironmentVariable] == "1" {
            return true
        }
        return Bundle.main.bundleURL.pathExtension.lowercased() == "app"
    }

    private static func fallbackKeyURL(environment: [String: String]) -> URL {
        if let override = environment[keyPathEnvironmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: false).standardizedFileURL
        }

        return ValarAppPaths()
            .applicationSupport
            .appendingPathComponent(fallbackKeyFilename, isDirectory: false)
    }

    private static func readFallbackKeyData(from url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try Data(contentsOf: url)
    }

    private static func writeFallbackKeyData(_ data: Data, to url: URL) throws {
        guard data.count == 32 else {
            throw VoiceLibraryProtectionError.invalidKeyMaterial
        }
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    private static func readKeyData() throws -> Data? {
        let query: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
            kSecReturnData: true,
            kSecMatchLimit: kSecMatchLimitOne,
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        switch status {
        case errSecSuccess:
            guard let data = item as? Data else {
                throw VoiceLibraryProtectionError.invalidKeyMaterial
            }
            return data
        case errSecItemNotFound:
            return nil
        default:
            throw VoiceLibraryProtectionError.keychainFailure(status, "read")
        }
    }

    private static func writeKeyData(_ data: Data) throws {
        let attributes: [CFString: Any] = [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
            kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecValueData: data,
        ]

        let status = SecItemAdd(attributes as CFDictionary, nil)
        // errSecDuplicateItem is thrown as-is so the caller can decide whether to
        // read back the existing key (race-safe) rather than overwriting it.
        guard status == errSecSuccess else {
            throw VoiceLibraryProtectionError.keychainFailure(status, "write")
        }
    }
}

public enum VoiceLibraryProtection {
    public static let currentVersion = 1

    private static let magicPrefix = Data("VALARSEC1\n".utf8)
    private static let algorithm = "AES.GCM"

    public static func isProtected(_ data: Data) -> Bool {
        data.starts(with: magicPrefix)
    }

    public static func protect(_ plaintext: Data) throws -> Data {
        if isProtected(plaintext) {
            return plaintext
        }

        let key = try VoiceLibraryProtectionKeychain.symmetricKey()
        let sealedBox = try AES.GCM.seal(plaintext, using: key)
        let envelope = VoiceLibraryEncryptedEnvelope(
            version: currentVersion,
            algorithm: algorithm,
            nonce: sealedBox.nonce.withUnsafeBytes { Data($0) },
            ciphertext: sealedBox.ciphertext,
            tag: sealedBox.tag
        )
        let encoded = try JSONEncoder().encode(envelope)
        return magicPrefix + encoded
    }

    public static func unprotectIfNeeded(_ stored: Data) throws -> Data {
        guard isProtected(stored) else { return stored }

        let encoded = stored.dropFirst(magicPrefix.count)
        let envelope: VoiceLibraryEncryptedEnvelope
        do {
            envelope = try JSONDecoder().decode(VoiceLibraryEncryptedEnvelope.self, from: Data(encoded))
        } catch {
            throw VoiceLibraryProtectionError.malformedEnvelope
        }
        guard envelope.version == currentVersion else {
            throw VoiceLibraryProtectionError.unsupportedEnvelopeVersion(envelope.version)
        }
        guard envelope.algorithm == algorithm else {
            throw VoiceLibraryProtectionError.malformedEnvelope
        }
        let nonce: AES.GCM.Nonce
        let sealedBox: AES.GCM.SealedBox
        do {
            nonce = try AES.GCM.Nonce(data: envelope.nonce)
            sealedBox = try AES.GCM.SealedBox(
                nonce: nonce,
                ciphertext: envelope.ciphertext,
                tag: envelope.tag
            )
        } catch {
            throw VoiceLibraryProtectionError.malformedEnvelope
        }
        let key = try VoiceLibraryProtectionKeychain.symmetricKey()
        do {
            return try AES.GCM.open(sealedBox, using: key)
        } catch {
            throw VoiceLibraryProtectionError.decryptionFailed
        }
    }

    public static func writeProtectedFile(
        _ plaintext: Data,
        to url: URL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try protect(plaintext).write(to: url, options: .atomic)
    }

    public static func readProtectedFile(from url: URL) throws -> Data {
        try unprotectIfNeeded(Data(contentsOf: url))
    }
}
