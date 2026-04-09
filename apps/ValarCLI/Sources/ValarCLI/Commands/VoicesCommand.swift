@preconcurrency import ArgumentParser
import Foundation
import ValarCore
import ValarModelKit
import ValarPersistence

struct VoicesCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voices",
        abstract: "Manage, create, and audition voices.",
        subcommands: [List.self, Create.self, Duplicate.self, CloneFile.self, Design.self, Stabilize.self, Delete.self, Audition.self]
    )

    mutating func run() throws {
        throw CleanExit.helpRequest(Self.self)
    }
}

extension VoicesCommand {
    struct List: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "Print all voices stored in the local GRDB database."
        )

        mutating func run() async throws {
            let runtime = try ValarRuntime()
            let voices = await runtime.listVoices()

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices list"),
                    data: VoiceListPayloadDTO(
                        message: voices.isEmpty
                            ? "No voices found in \(runtime.paths.databaseURL.path)."
                            : "Loaded \(voices.count) voice(s).",
                        databasePath: runtime.paths.databaseURL.path,
                        voices: voices.map { VoicesCommand.voiceSummaryDTO($0) }
                    )
                )
                return
            }

            guard voices.isEmpty == false else {
                print("No voices found in \(runtime.paths.databaseURL.path).")
                return
            }

            let presetVoices = voices.filter(\.isModelDeclaredPreset)
            let savedQwenVoices = voices.filter { !$0.isModelDeclaredPreset && $0.inferredFamilyID == ModelFamilyID.qwen3TTS.rawValue }
            let savedTADAVoices = voices.filter { !$0.isModelDeclaredPreset && $0.inferredFamilyID == ModelFamilyID.tadaTTS.rawValue }
            let otherSavedVoices = voices.filter {
                !$0.isModelDeclaredPreset
                    && $0.inferredFamilyID != ModelFamilyID.qwen3TTS.rawValue
                    && $0.inferredFamilyID != ModelFamilyID.tadaTTS.rawValue
            }

            Self.printSection(title: "Preset Voices", voices: presetVoices)
            if !presetVoices.isEmpty, (!savedQwenVoices.isEmpty || !savedTADAVoices.isEmpty || !otherSavedVoices.isEmpty) {
                print("")
            }
            Self.printSection(title: "Saved Qwen Voices", voices: savedQwenVoices)
            if !savedQwenVoices.isEmpty, (!savedTADAVoices.isEmpty || !otherSavedVoices.isEmpty) {
                print("")
            }
            Self.printSection(title: "Saved TADA Voices", voices: savedTADAVoices)
            if !savedTADAVoices.isEmpty, !otherSavedVoices.isEmpty {
                print("")
            }
            Self.printSection(title: "Other Saved Voices", voices: otherSavedVoices)
        }

        private static func sanitize(_ value: String) -> String {
            value.replacingOccurrences(of: "\t", with: " ").replacingOccurrences(of: "\n", with: " ")
        }

        private static func printSection(title: String, voices: [VoiceLibraryRecord]) {
            guard !voices.isEmpty else { return }
            print(title)
            let idWidth = 36
            let labelWidth = 24
            let typeWidth = 16
            let modelWidth = 48
            let header = pad("ID", toLength: idWidth)
                + "  " + pad("Label", toLength: labelWidth)
                + "  " + pad("Type", toLength: typeWidth)
                + "  " + pad("Model", toLength: modelWidth)
                + "  Source"
            print(header)
            for voice in voices {
                let line = pad(voice.id.uuidString, toLength: idWidth)
                    + "  " + pad(sanitize(voice.label), toLength: labelWidth)
                    + "  " + pad(sanitize(voice.typeDisplayName), toLength: typeWidth)
                    + "  " + pad(sanitize(voice.modelID), toLength: modelWidth)
                    + "  " + sanitize(VoicesCommand.previewLabel(for: voice))
                print(line)
            }
        }

        private static func pad(_ value: String, toLength length: Int) -> String {
            if value.count >= length {
                return String(value.prefix(length))
            }
            return value + String(repeating: " ", count: length - value.count)
        }
    }

    struct Create: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "create",
            abstract: "Create a new voice record in the local voice library. For cloning from audio, prefer 'clone-file'."
        )

        @Option(name: .long, help: "Display name for the new voice.")
        var name: String

        @Option(name: .long, help: "Model identifier used to create or classify the saved voice.")
        var model: String?

        @Option(name: .long, help: "Reference audio file for creating a cloned voice.")
        var referenceAudio: String?

        @Option(name: .long, help: "Reference transcript for the audio clip. Required when saving a cloned voice.")
        var referenceTranscript: String?

        @Option(name: .long, help: "Optional description or prompt to store with the voice. If provided without --model, Valar defaults to the Qwen VoiceDesign lane.")
        var description: String?

        mutating func run() async throws {
            let runtime = try ValarRuntime()
            let voice: VoiceLibraryRecord

            if let referenceAudio = referenceAudio?.trimmingCharacters(in: .whitespacesAndNewlines),
               !referenceAudio.isEmpty {
                let audioURL = URL(fileURLWithPath: referenceAudio).standardizedFileURL
                guard FileManager.default.fileExists(atPath: audioURL.path) else {
                    throw CLICommandError(message: "Audio file not found: \(audioURL.path)")
                }
                let resolvedTranscript = referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !resolvedTranscript.isEmpty else {
                    throw ValidationError("Reference transcript is required when saving a cloned voice.")
                }

                let requestedModel = model.map { ModelIdentifier($0) }

                voice = try await runtime.cloneVoice(
                    VoiceCloneRequest(
                        label: name,
                        referenceTranscript: resolvedTranscript,
                        audioData: try Data(contentsOf: audioURL),
                        audioFileExtension: audioURL.pathExtension.lowercased(),
                        sourceAssetName: audioURL.lastPathComponent,
                        modelID: requestedModel
                    )
                )
            } else {
                voice = try await runtime.createVoice(
                    VoiceCreateRequest(
                        label: name,
                        modelID: model,
                        voicePrompt: description
                    )
                )
            }
            let message = "Created voice \(voice.label) (\(voice.id.uuidString))."

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices create"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: voice,
                            preview: previewLabel(for: voice)
                        ),
                        previewPath: ""
                    )
                )
                return
            }

            print(message)
        }
    }

    struct Duplicate: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "duplicate",
            abstract: "Duplicate an existing saved voice by UUID."
        )

        @Argument(help: "Voice UUID from `valartts voices list`.")
        var id: String

        @Option(name: .long, help: "Optional display name for the cloned voice.")
        var name: String?

        mutating func run() async throws {
            guard let voiceID = UUID(uuidString: id) else {
                throw ValidationError("Voice id must be a UUID.")
            }

            if let name, name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw ValidationError("Cloned voice name must not be empty.")
            }

            let runtime = try ValarRuntime()
            guard let sourceVoice = await runtime.voiceRecord(id: voiceID) else {
                throw ValidationError("No voice exists with id \(id).")
            }

            guard
                let transcript = sourceVoice.referenceTranscript?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                transcript.isEmpty == false
            else {
                throw CLICommandError(
                    message: "Voice \(id) cannot be cloned because it has no stored reference transcript."
                )
            }

            let previewURL = try VoicesCommand.resolvePreviewURL(for: sourceVoice, paths: runtime.paths)
            let audioData = try VoiceLibraryProtection.readProtectedFile(from: previewURL)
            let clonedVoice = try await runtime.cloneVoice(
                VoiceCloneRequest(
                    label: resolvedCloneName(sourceVoice.label),
                    referenceTranscript: transcript,
                    audioData: audioData,
                    audioFileExtension: previewURL.pathExtension,
                    sourceAssetName: sourceVoice.sourceAssetName
                )
            )
            let clonedPreviewURL = try VoicesCommand.resolvePreviewURL(for: clonedVoice, paths: runtime.paths)
            let message = "Duplicated voice '\(sourceVoice.label)' as '\(clonedVoice.label)' (\(clonedVoice.id.uuidString))."

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices duplicate"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: clonedVoice,
                            preview: previewLabel(for: clonedVoice)
                        ),
                        previewPath: clonedPreviewURL.path
                    )
                )
                return
            }

            print(message)
        }

        private func resolvedCloneName(_ sourceLabel: String) -> String {
            guard let name else {
                return "\(sourceLabel) Copy"
            }

            return name.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    struct CloneFile: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "clone-file",
            abstract: "Create a new voice from a reference audio file (does not require an existing voice record)."
        )

        @Option(name: .long, help: "Path to the reference audio file (WAV or M4A, 5–30 seconds).")
        var audio: String

        @Option(name: .long, help: "Display name for the new voice.")
        var name: String

        @Option(name: .long, help: "Reference transcript for the audio clip. Required when saving a cloned voice.")
        var transcript: String?

        @Option(name: .long, help: "Optional model ID or alias to use for cloning (for example HumeAI/mlx-tada-3b). If omitted, Valar uses the Qwen Base clone-prompt lane and only falls back to TADA when Base is unavailable.")
        var model: String?

        mutating func run() async throws {
            let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmedName.isEmpty == false else {
                throw ValidationError("Voice name must not be empty.")
            }

            let audioURL = URL(fileURLWithPath: audio).standardizedFileURL
            guard FileManager.default.fileExists(atPath: audioURL.path) else {
                throw CLICommandError(message: "Audio file not found: \(audioURL.path)")
            }

            let audioData = try Data(contentsOf: audioURL)
            let fileExtension = audioURL.pathExtension.lowercased()
            let resolvedTranscript = transcript?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !resolvedTranscript.isEmpty else {
                throw ValidationError("Reference transcript is required when saving a cloned voice.")
            }
            let requestedModel = model.map { ModelIdentifier($0) }

            let runtime = try ValarRuntime()
            let voice = try await runtime.cloneVoice(
                VoiceCloneRequest(
                    label: trimmedName,
                    referenceTranscript: resolvedTranscript,
                    audioData: audioData,
                    audioFileExtension: fileExtension,
                    sourceAssetName: audioURL.lastPathComponent,
                    modelID: requestedModel
                )
            )
            let message = "Created voice '\(voice.label)' (\(voice.id.uuidString)) from \(audioURL.lastPathComponent)."

            if OutputContext.jsonRequested {
                let previewPath = (try? VoicesCommand.resolvePreviewURL(for: voice, paths: runtime.paths))?.path ?? ""
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices clone-file"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: voice,
                            preview: VoicesCommand.previewLabel(for: voice)
                        ),
                        previewPath: previewPath
                    )
                )
                return
            }

            print(message)
        }
    }

    struct Design: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "design",
            abstract: "Create a designed voice from a text description."
        )

        @Option(name: .long, help: "Display name for the designed voice.")
        var name: String

        @Option(name: .long, help: "Text description of the voice style, tone, and characteristics.")
        var description: String

        mutating func run() async throws {
            let trimmedDescription = description.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmedDescription.isEmpty == false else {
                throw ValidationError("Description must not be empty.")
            }

            let runtime = try ValarRuntime()
            let voice = try await runtime.createVoice(
                VoiceCreateRequest(
                    label: name,
                    voicePrompt: trimmedDescription
                )
            )
            let message = "Created designed voice \(voice.label) (\(voice.id.uuidString))."

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices design"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: voice,
                            preview: previewLabel(for: voice)
                        ),
                        previewPath: ""
                    )
                )
                return
            }

            print(message)
        }
    }

    struct Stabilize: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "stabilize",
            abstract: "Convert an expressive designed Qwen voice into a stable narrator voice backed by the Base clone-prompt lane."
        )

        @Argument(help: "Source voice UUID from `valartts voices list`.")
        var id: String

        @Option(name: .long, help: "Optional display name for the new stable narrator voice.")
        var name: String?

        @Option(name: .long, help: "Optional anchor text to synthesize for the stabilization reference clip.")
        var anchorText: String?

        @Option(name: .long, help: "Optional Base-model override to use for the stable narrator clone.")
        var model: String?

        mutating func run() async throws {
            guard let voiceID = UUID(uuidString: id) else {
                throw ValidationError("Voice id must be a UUID.")
            }

            let runtime = try ValarRuntime()
            let voice = try await runtime.stabilizeVoice(
                VoiceStabilizeRequest(
                    sourceVoiceID: voiceID,
                    label: name,
                    anchorText: anchorText,
                    modelID: model.map { ModelIdentifier($0) }
                )
            )
            let message = "Created stable narrator voice \(voice.label) (\(voice.id.uuidString))."

            if OutputContext.jsonRequested {
                let previewPath = (try? VoicesCommand.resolvePreviewURL(for: voice, paths: runtime.paths))?.path ?? ""
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices stabilize"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: voice,
                            preview: previewLabel(for: voice)
                        ),
                        previewPath: previewPath
                    )
                )
                return
            }

            print(message)
        }
    }

    struct Delete: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "delete",
            abstract: "Delete a saved voice and any retained reference-audio or conditioning assets."
        )

        @Argument(help: "Voice UUID from `valartts voices list`.")
        var id: String

        mutating func run() async throws {
            guard let voiceID = UUID(uuidString: id) else {
                throw ValidationError("Voice id must be a UUID.")
            }

            let runtime = try ValarRuntime()
            let voiceRecord = await runtime.voiceRecord(id: voiceID)
            try await runtime.deleteVoice(voiceID)
            let label = voiceRecord?.label ?? voiceID.uuidString
            let message = "Deleted voice '\(label)' (\(voiceID.uuidString))."

            if OutputContext.jsonRequested {
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices delete"),
                    data: ValarCommandSuccessPayloadDTO(message: message)
                )
                return
            }

            print(message)
        }
    }

    struct Audition: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "audition",
            abstract: "Play the stored preview clip for a voice."
        )

        @Argument(help: "Voice UUID from `valartts voices list`.")
        var id: String

        mutating func run() async throws {
            guard let voiceID = UUID(uuidString: id) else {
                throw ValidationError("Voice id must be a UUID.")
            }

            let runtime = try ValarRuntime()
            guard let voice = await runtime.voiceRecord(id: voiceID) else {
                throw ValidationError("No voice exists with id \(id).")
            }
            guard !voice.isModelDeclaredPreset else {
                throw CLICommandError(message: "Voice \(id) is a model-declared preset and has no stored preview clip. Use `valartts speak --voice \(voice.label) 'Your text here'` to audition it.")
            }

            let previewURL = try VoicesCommand.resolvePreviewURL(for: voice, paths: runtime.paths)
            let message = "Playing preview for \(voice.label): \(previewURL.path)"
            if OutputContext.jsonRequested {
                try VoicesCommand.playPreview(at: previewURL)
                try OutputFormat.writeSuccess(
                    command: OutputFormat.commandPath("voices audition"),
                    data: VoiceDetailPayloadDTO(
                        message: message,
                        voice: VoiceDetailDTO(
                            from: voice,
                            preview: previewLabel(for: voice)
                        ),
                        previewPath: previewURL.path
                    )
                )
                return
            }

            print(message)
            try VoicesCommand.playPreview(at: previewURL)
        }
    }
}

private extension VoicesCommand {
    static let playableExtensions = ["wav", "aiff", "m4a", "mp3", "caf"]

    static func previewLabel(for voice: VoiceLibraryRecord) -> String {
        if let backendVoiceID = voice.backendVoiceID {
            return backendVoiceID
        }
        return voice.referenceAudioAssetName
            ?? voice.conditioningAssetName
            ?? voice.sourceAssetName
            ?? "\(voice.id.uuidString).<audio>"
    }

    static func voiceSummaryDTO(_ voice: VoiceLibraryRecord) -> VoiceSummaryDTO {
        VoiceSummaryDTO(from: voice, preview: previewLabel(for: voice))
    }

    static func resolvePreviewURL(
        for voice: VoiceLibraryRecord,
        paths: ValarAppPaths,
        fileManager: FileManager = .default
    ) throws -> URL {
        let namedCandidates = [voice.referenceAudioAssetName, voice.sourceAssetName].compactMap { assetName in
            assetName?.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { $0.isEmpty == false }

        for assetName in namedCandidates {
            let candidateURL = try resolveNamedAsset(assetName, voiceLibraryDirectory: paths.voiceLibraryDirectory)
            if fileManager.fileExists(atPath: candidateURL.path) {
                return candidateURL
            }
        }

        let preferredExtensions = namedCandidates
            .map { URL(fileURLWithPath: $0).pathExtension.lowercased() }
            .filter { $0.isEmpty == false }

        for fileExtension in unique(preferredExtensions + playableExtensions) {
            let candidateURL = try paths.voiceAssetURL(voiceID: voice.id, fileExtension: fileExtension)
            if fileManager.fileExists(atPath: candidateURL.path) {
                return candidateURL
            }
        }

        let hint = namedCandidates.isEmpty
            ? " This voice was created from a text description. Generate audio first with: valartts speak --voice \(voice.id.uuidString) 'preview text' --output preview.wav"
            : ""
        throw CLICommandError(
            message: "No preview audio clip was found for voice \(voice.id.uuidString). Checked \(paths.voiceLibraryDirectory.path).\(hint)"
        )
    }

    static func resolveNamedAsset(
        _ assetName: String,
        voiceLibraryDirectory: URL
    ) throws -> URL {
        let trimmed = assetName.trimmingCharacters(in: .whitespacesAndNewlines)
        let assetURL = URL(fileURLWithPath: trimmed)

        if assetURL.path.hasPrefix("/") {
            return assetURL.standardizedFileURL
        }

        try ValarAppPaths.validateRelativePath(trimmed, label: "voice asset")
        let resolvedURL = voiceLibraryDirectory.appendingPathComponent(trimmed, isDirectory: false)
        try ValarAppPaths.validateContainment(resolvedURL, within: voiceLibraryDirectory)
        return resolvedURL
    }

    static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []

        for value in values {
            if seen.insert(value).inserted {
                ordered.append(value)
            }
        }

        return ordered
    }

    static func playPreview(
        at fileURL: URL,
        processRunner: (Process) throws -> Void = { process in
            try process.run()
        }
    ) throws {
        let playableURL: URL
        var removeAfterPlayback = false
        if let encryptedData = try? Data(contentsOf: fileURL),
           VoiceLibraryProtection.isProtected(encryptedData) {
            let temporaryURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("valar-preview-\(UUID().uuidString)", isDirectory: false)
                .appendingPathExtension(fileURL.pathExtension.isEmpty ? "wav" : fileURL.pathExtension)
            let decrypted = try VoiceLibraryProtection.unprotectIfNeeded(encryptedData)
            try decrypted.write(to: temporaryURL, options: .atomic)
            playableURL = temporaryURL
            removeAfterPlayback = true
        } else {
            playableURL = fileURL
        }

        defer {
            if removeAfterPlayback {
                try? FileManager.default.removeItem(at: playableURL)
            }
        }

        let process = Process()
        let errorPipe = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        process.arguments = [playableURL.path]
        process.standardError = errorPipe

        try processRunner(process)
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorMessage = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw CLICommandError(
                message: errorMessage?.isEmpty == false ? errorMessage! : "Failed to play preview audio."
            )
        }
    }
}
