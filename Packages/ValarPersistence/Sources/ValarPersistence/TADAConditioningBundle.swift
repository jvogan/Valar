import Foundation
import ValarModelKit

public struct TADAConditioningManifest: Codable, Sendable, Equatable, Hashable {
    public let format: String
    public let modelID: String
    public let transcript: String
    public let language: String
    public let sampleRate: Int
    public let tokenCount: Int
    public let acousticDimension: Int
    public let frameCount: Int?

    enum CodingKeys: String, CodingKey {
        case format
        case modelID = "model_id"
        case transcript
        case language
        case sampleRate = "sample_rate"
        case tokenCount = "token_count"
        case acousticDimension = "acoustic_dim"
        case frameCount = "frame_count"
    }

    public init(
        format: String = VoiceConditioning.tadaReferenceV1,
        modelID: String,
        transcript: String,
        language: String,
        sampleRate: Int,
        tokenCount: Int,
        acousticDimension: Int,
        frameCount: Int? = nil
    ) {
        self.format = format
        self.modelID = modelID
        self.transcript = transcript
        self.language = language
        self.sampleRate = sampleRate
        self.tokenCount = tokenCount
        self.acousticDimension = acousticDimension
        self.frameCount = frameCount
    }

    public init?(metadata: VoiceConditioningMetadata?) {
        guard
            let metadata,
            let modelID = metadata.modelID,
            let transcript = metadata.transcript,
            let language = metadata.language,
            let sampleRate = metadata.sampleRate,
            let tokenCount = metadata.tokenCount,
            let acousticDimension = metadata.acousticDimension
        else {
            return nil
        }

        self.init(
            modelID: modelID,
            transcript: transcript,
            language: language,
            sampleRate: Int(sampleRate.rounded()),
            tokenCount: tokenCount,
            acousticDimension: acousticDimension,
            frameCount: metadata.frameCount
        )
    }
}

public enum TADAConditioningBundleError: Error, LocalizedError {
    case assetTooLarge(String)

    public var errorDescription: String? {
        switch self {
        case let .assetTooLarge(filename):
            return "TADA conditioning asset '\(filename)' exceeds the maximum allowed file size."
        }
    }
}

public enum TADAConditioningBundleIO {
    public static let manifestFilename = "conditioning.json"
    public static let tokenValuesFilename = "token_values.f16"
    public static let tokenPositionsFilename = "token_positions.i32"
    public static let textTokensFilename = "text_tokens.i32"
    public static let tokenMasksFilename = "token_masks.u8"

    private static let requiredBinaryFilenames: Set<String> = [
        tokenValuesFilename,
        tokenPositionsFilename,
        textTokensFilename,
    ]

    public static func write(
        conditioning: VoiceConditioning,
        to directoryURL: URL,
        fileManager: FileManager = .default
    ) throws -> String {
        guard conditioning.format == VoiceConditioning.tadaReferenceV1 else {
            throw NSError(domain: "TADAConditioningBundleIO", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Expected \(VoiceConditioning.tadaReferenceV1) conditioning."
            ])
        }
        guard let manifest = TADAConditioningManifest(metadata: conditioning.metadata) else {
            throw NSError(domain: "TADAConditioningBundleIO", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "TADA conditioning metadata is incomplete."
            ])
        }

        let filenames = Set(conditioning.assetFiles.map(\.filename))
        guard requiredBinaryFilenames.isSubset(of: filenames) else {
            throw NSError(domain: "TADAConditioningBundleIO", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "TADA conditioning is missing one or more required binary payloads."
            ])
        }

        try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let manifestURL = directoryURL.appendingPathComponent(manifestFilename, isDirectory: false)
        try VoiceLibraryProtection.writeProtectedFile(encoder.encode(manifest), to: manifestURL, fileManager: fileManager)

        for asset in conditioning.assetFiles {
            let assetURL = directoryURL.appendingPathComponent(asset.filename, isDirectory: false)
            try VoiceLibraryProtection.writeProtectedFile(asset.data, to: assetURL, fileManager: fileManager)
        }

        return directoryURL.lastPathComponent
    }

    public static func load(
        from directoryURL: URL,
        fileManager: FileManager = .default
    ) throws -> VoiceConditioning {
        let decoder = JSONDecoder()
        let manifestURL = directoryURL.appendingPathComponent(manifestFilename, isDirectory: false)
        let manifest = try decoder.decode(
            TADAConditioningManifest.self,
            from: VoiceLibraryProtection.readProtectedFile(from: manifestURL)
        )

        let filenames = [
            tokenValuesFilename,
            tokenPositionsFilename,
            textTokensFilename,
            tokenMasksFilename,
        ]
        let assets = try filenames.compactMap { filename -> VoiceConditioningAssetFile? in
            let url = directoryURL.appendingPathComponent(filename, isDirectory: false)
            guard fileManager.fileExists(atPath: url.path) else {
                if filename == tokenMasksFilename {
                    return nil
                }
                throw NSError(domain: "TADAConditioningBundleIO", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Missing required TADA conditioning file '\(filename)'."
                ])
            }
            let attrs = try fileManager.attributesOfItem(atPath: url.path)
            let fileSize = (attrs[.size] as? Int) ?? 0
            guard fileSize <= 64_000_000 else { // 64 MB cap
                throw TADAConditioningBundleError.assetTooLarge(filename)
            }
            return VoiceConditioningAssetFile(
                filename: filename,
                data: try VoiceLibraryProtection.readProtectedFile(from: url)
            )
        }

        let metadata = VoiceConditioningMetadata(
            modelID: manifest.modelID,
            transcript: manifest.transcript,
            language: manifest.language,
            sampleRate: Double(manifest.sampleRate),
            tokenCount: manifest.tokenCount,
            acousticDimension: manifest.acousticDimension,
            frameCount: manifest.frameCount
        )

        return VoiceConditioning(
            format: manifest.format,
            payload: nil,
            assetFiles: assets,
            assetName: directoryURL.lastPathComponent,
            sourceModel: ModelIdentifier(manifest.modelID),
            metadata: metadata
        )
    }
}
