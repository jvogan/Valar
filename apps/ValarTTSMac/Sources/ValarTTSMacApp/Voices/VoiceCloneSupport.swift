import Foundation
import ValarAudio
import ValarModelKit

struct VoiceCloneDraft: Sendable, Equatable {
    let label: String
    let referenceAudioURL: URL
    let referenceTranscript: String
    let modelID: ModelIdentifier?
}

struct VoiceCloneAssessment: Sendable, Equatable {
    let normalizedBuffer: AudioPCMBuffer
    let durationSeconds: Double
    let sampleRate: Double
    let originalChannelCount: Int
    let warningMessage: String?
}

enum VoiceCloneModels {
    static let profileModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    static let defaultRuntimeModelID: ModelIdentifier = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
}

enum VoiceCloneError: LocalizedError, Equatable {
    case emptyLabel
    case emptyTranscript
    case unsupportedFileType(String)
    case fileTooLarge(bytes: Int)
    case unreadableFile
    case invalidAudioHeader(String)
    case clipTooShort(actual: Double)
    case clipTooLong(actual: Double)
    case sampleRateTooLow(actual: Double)
    case missingReferenceAudio(String)

    var errorDescription: String? {
        switch self {
        case .emptyLabel:
            return "Enter a label for the cloned voice."
        case .emptyTranscript:
            return "Enter the transcript for the reference clip. Valar currently requires the spoken text when saving cloned voices."
        case .unsupportedFileType(let fileExtension):
            return "Unsupported audio file type '.\(fileExtension)'. Choose a \(VoiceCloneFileValidator.allowedExtensionsChoiceText) clip."
        case .fileTooLarge(let bytes):
            let megabytes = Double(bytes) / (1_024 * 1_024)
            return "Reference audio file is too large (\(megabytes.formatted(.number.precision(.fractionLength(1)))) MB). Maximum allowed size is \(VoiceCloneFileValidator.maximumFileSizeMB) MB."
        case .unreadableFile:
            return "The selected file could not be read. Check that the file exists and is accessible."
        case .invalidAudioHeader(let expected):
            return "The selected file does not appear to be a valid \(expected) audio file. The file may be corrupted or mislabeled."
        case .clipTooShort(let actual):
            return "Reference audio must be at least 5 seconds. Selected clip is \(actual.formatted(.number.precision(.fractionLength(1)))) seconds."
        case .clipTooLong(let actual):
            return "Reference audio must be 30 seconds or less. Selected clip is \(actual.formatted(.number.precision(.fractionLength(1)))) seconds."
        case .sampleRateTooLow(let actual):
            return "Reference audio must be at least 16 kHz. Selected clip is \(actual.formatted(.number.precision(.fractionLength(0)))) Hz."
        case .missingReferenceAudio(let assetName):
            return "The saved reference audio '\(assetName)' could not be found."
        }
    }
}

enum VoiceCloneAudioValidator {
    static func validate(_ buffer: AudioPCMBuffer) throws -> VoiceCloneAssessment {
        let duration = buffer.duration
        guard duration >= 5 else {
            throw VoiceCloneError.clipTooShort(actual: duration)
        }
        guard duration <= 30 else {
            throw VoiceCloneError.clipTooLong(actual: duration)
        }
        guard buffer.format.sampleRate >= 16_000 else {
            throw VoiceCloneError.sampleRateTooLow(actual: buffer.format.sampleRate)
        }

        let originalChannelCount = max(buffer.format.channelCount, buffer.channels.count)
        let normalizedBuffer = downmixToMono(buffer)
        let warning = originalChannelCount == 1
            ? nil
            : "Stereo and multi-channel clips are accepted, but Valar will downmix them to mono before cloning."

        return VoiceCloneAssessment(
            normalizedBuffer: normalizedBuffer,
            durationSeconds: duration,
            sampleRate: normalizedBuffer.format.sampleRate,
            originalChannelCount: originalChannelCount,
            warningMessage: warning
        )
    }

    private static func downmixToMono(_ buffer: AudioPCMBuffer) -> AudioPCMBuffer {
        guard buffer.channels.count > 1 else { return buffer }

        let frameCount = buffer.frameCount
        guard frameCount > 0 else {
            return AudioPCMBuffer(mono: [], sampleRate: buffer.format.sampleRate, container: buffer.format.container)
        }

        var mixed = Array(repeating: Float.zero, count: frameCount)
        let channels = buffer.channels.filter { !$0.isEmpty }
        guard !channels.isEmpty else { return buffer }

        for channel in channels {
            for index in 0 ..< min(channel.count, frameCount) {
                mixed[index] += channel[index]
            }
        }

        let divisor = Float(channels.count)
        mixed = mixed.map { $0 / divisor }
        return AudioPCMBuffer(mono: mixed, sampleRate: buffer.format.sampleRate, container: buffer.format.container)
    }
}

enum VoiceCloneFileValidator {
    static let allowedExtensionsInDisplayOrder = ["wav", "m4a"]
    static let allowedExtensions = Set(allowedExtensionsInDisplayOrder)
    static let allowedExtensionsDisplayText = allowedExtensionsInDisplayOrder
        .map { $0.uppercased() }
        .joined(separator: ", ")
    static let allowedExtensionsChoiceText = allowedExtensionsInDisplayOrder
        .map { $0.uppercased() }
        .joined(separator: " or ")
    static let maximumFileSizeBytes: Int = 50 * 1_024 * 1_024
    static let maximumFileSizeMB: Int = 50

    /// Validates a file URL at selection time (extension + readability + size).
    /// Call this from the UI layer when the user picks or drops a file.
    static func validateFileSelection(_ url: URL) throws {
        let fileExtension = url.pathExtension.lowercased()
        guard allowedExtensions.contains(fileExtension) else {
            throw VoiceCloneError.unsupportedFileType(fileExtension)
        }

        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = attributes[.size] as? Int else {
            throw VoiceCloneError.unreadableFile
        }

        guard fileSize <= maximumFileSizeBytes else {
            throw VoiceCloneError.fileTooLarge(bytes: fileSize)
        }
    }

    /// Validates that raw file data has a header consistent with the claimed extension.
    /// Call this after reading the file but before decoding.
    static func validateFileHeader(_ data: Data, hint: String) throws {
        guard data.count >= 12 else {
            throw VoiceCloneError.invalidAudioHeader(hint.uppercased())
        }

        switch hint.lowercased() {
        case "wav":
            // WAV files start with "RIFF" at offset 0 and "WAVE" at offset 8
            let riff = data.prefix(4)
            let wave = data[data.startIndex + 8 ..< data.startIndex + 12]
            guard riff.elementsEqual("RIFF".utf8),
                  wave.elementsEqual("WAVE".utf8) else {
                throw VoiceCloneError.invalidAudioHeader("WAV")
            }
        case "m4a":
            // M4A/MP4 files have "ftyp" at offset 4
            let ftyp = data[data.startIndex + 4 ..< data.startIndex + 8]
            guard ftyp.elementsEqual("ftyp".utf8) else {
                throw VoiceCloneError.invalidAudioHeader("M4A")
            }
        default:
            throw VoiceCloneError.unsupportedFileType(hint)
        }
    }
}
