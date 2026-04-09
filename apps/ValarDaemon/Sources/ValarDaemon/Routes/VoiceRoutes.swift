import Foundation
import Hummingbird
import HTTPTypes
import ValarCore
import ValarModelKit
import ValarPersistence

private actor VoiceMutationRateLimiter {
    struct Decision: Sendable {
        let allowed: Bool
        let retryAfterSeconds: Int
    }

    private let window: TimeInterval
    private let limit: Int
    private var events: [String: [Date]] = [:]

    init(limit: Int = 6, window: TimeInterval = 60) {
        self.limit = limit
        self.window = window
    }

    func acquire(action: String, now: Date = Date()) -> Decision {
        let cutoff = now.addingTimeInterval(-window)
        var actionEvents = events[action, default: []].filter { $0 >= cutoff }
        guard actionEvents.count < limit else {
            let retryAfter = max(1, Int(ceil(window - now.timeIntervalSince(actionEvents[0]))))
            events[action] = actionEvents
            return Decision(allowed: false, retryAfterSeconds: retryAfter)
        }
        actionEvents.append(now)
        events[action] = actionEvents
        return Decision(allowed: true, retryAfterSeconds: 0)
    }
}

extension ValarDaemonRouter {
    private static let mutationRateLimiter = VoiceMutationRateLimiter()

    // MARK: - Route registration

    static func registerVoiceRoutes(on router: HTTPRouteGroup, runtime: ValarRuntime) {
        let voices = router.group("voices")

        // GET /v1/voices
        voices.get { _, _ async throws -> Response in
            let records = await runtime.listRouteVoices()
            let dtos = records.map { voice in
                VoiceSummaryDTO(
                    from: voice,
                    preview: voice.backendVoiceID
                        ?? voice.referenceAudioAssetName
                        ?? voice.conditioningAssetName
                        ?? voice.sourceAssetName
                        ?? "\(voice.id.uuidString).<audio>"
                )
            }
            return try jsonResponse(dtos)
        }

        // POST /v1/voices/create
        voices.post("create") { request, _ async throws -> Response in
            await handleVoiceCreateRequest(request, runtime: runtime)
        }

        // POST /v1/voices/clone
        voices.post("clone") { request, _ async throws -> Response in
            await handleVoiceCloneRequest(request, runtime: runtime)
        }

        // POST /v1/voices/design
        voices.post("design") { request, _ async throws -> Response in
            await handleVoiceDesignRequest(request, runtime: runtime)
        }

        // POST /v1/voices/stabilize
        voices.post("stabilize") { request, _ async throws -> Response in
            await handleVoiceStabilizeRequest(request, runtime: runtime)
        }

        // DELETE /v1/voices/{id}
        voices.delete(":id") { request, context async throws -> Response in
            await handleVoiceDeleteRequest(request, context: context, runtime: runtime)
        }
    }

    // MARK: - Create handler

    private static func handleVoiceCreateRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let decision = await mutationRateLimiter.acquire(action: "create")
        guard decision.allowed else {
            return voiceRateLimitResponse(action: "create", retryAfterSeconds: decision.retryAfterSeconds)
        }

        struct CreateBody: Decodable {
            let name: String
            let description: String?
        }

        let body: CreateBody
        do {
            let buffer = try await request.body.collect(upTo: 1_000_000)
            let data = Data(buffer: buffer)
            body = try JSONDecoder().decode(CreateBody.self, from: data)
        } catch {
            return voiceErrorResponse(
                "Invalid JSON body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        guard body.name.count <= 200 else {
            return voiceErrorResponse("Field 'name' must not exceed 200 characters.", status: .badRequest)
        }
        if let description = body.description, description.count > 2000 {
            return voiceErrorResponse("Field 'description' must not exceed 2000 characters.", status: .badRequest)
        }

        let voice: VoiceLibraryRecord
        do {
            voice = try await runtime.createRouteVoice(name: body.name, description: body.description)
        } catch let error as ValarVoiceError {
            return voiceErrorResponse(error.localizedDescription, status: .badRequest)
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .internalServerError)
        }

        let dto = VoiceSummaryDTO(from: voice, preview: voice.backendVoiceID ?? voice.voicePrompt ?? voice.label)
        do {
            return try jsonResponse(dto, status: .created)
        } catch {
            return voiceErrorResponse("Failed to encode response.", status: .internalServerError)
        }
    }

    // MARK: - Design handler

    private static func handleVoiceDesignRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let decision = await mutationRateLimiter.acquire(action: "design")
        guard decision.allowed else {
            return voiceRateLimitResponse(action: "design", retryAfterSeconds: decision.retryAfterSeconds)
        }

        struct DesignBody: Decodable {
            let name: String
            let description: String?
        }

        let body: DesignBody
        do {
            let buffer = try await request.body.collect(upTo: 1_000_000)
            let data = Data(buffer: buffer)
            body = try JSONDecoder().decode(DesignBody.self, from: data)
        } catch {
            return voiceErrorResponse(
                "Invalid JSON body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        guard body.name.count <= 200 else {
            return voiceErrorResponse("Field 'name' must not exceed 200 characters.", status: .badRequest)
        }
        if let description = body.description, description.count > 2000 {
            return voiceErrorResponse("Field 'description' must not exceed 2000 characters.", status: .badRequest)
        }

        guard body.description != nil else {
            return voiceErrorResponse("Field 'description' is required.", status: .badRequest)
        }

        let resolvedDescription = body.description!

        let voice: VoiceLibraryRecord
        do {
            voice = try await runtime.createRouteVoice(name: body.name, description: resolvedDescription)
        } catch let error as ValarVoiceError {
            return voiceErrorResponse(error.localizedDescription, status: .badRequest)
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .internalServerError)
        }

        let dto = VoiceSummaryDTO(from: voice, preview: voice.backendVoiceID ?? voice.voicePrompt ?? voice.label)
        do {
            return try jsonResponse(dto, status: .created)
        } catch {
            return voiceErrorResponse("Failed to encode response.", status: .internalServerError)
        }
    }

    // MARK: - Clone handler

    private static func handleVoiceCloneRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let decision = await mutationRateLimiter.acquire(action: "clone")
        guard decision.allowed else {
            return voiceRateLimitResponse(action: "clone", retryAfterSeconds: decision.retryAfterSeconds)
        }

        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: 15_000_000)
            body = Data(buffer: buffer)
        } catch {
            return voiceErrorResponse(
                "Failed to read request body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        let form: VoiceCloneFormData
        do {
            form = try VoiceCloneFormData.parse(
                body: body,
                contentType: request.headers[.contentType] ?? ""
            )
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .badRequest)
        }

        guard let name = voiceTrimmedOrNil(form.fields["name"]) else {
            return voiceErrorResponse("Multipart field 'name' is required.", status: .badRequest)
        }

        guard name.count <= 200 else {
            return voiceErrorResponse("Multipart field 'name' must not exceed 200 characters.", status: .badRequest)
        }

        guard let file = form.files["file"] else {
            return voiceErrorResponse("Multipart field 'file' is required.", status: .badRequest)
        }

        guard let transcript = voiceTrimmedOrNil(form.fields["transcript"]) else {
            return voiceErrorResponse("Multipart field 'transcript' is required.", status: .badRequest)
        }
        let fileExtension = voicePathExtension(from: file.filename) ?? file.filename.lowercased()
        let modelID = voiceTrimmedOrNil(form.fields["model"]).map { ModelIdentifier($0) }

        let voice: VoiceLibraryRecord
        do {
            voice = try await runtime.cloneVoice(
                VoiceCloneRequest(
                    label: name,
                    referenceTranscript: transcript,
                    audioData: file.data,
                    audioFileExtension: fileExtension,
                    sourceAssetName: URL(fileURLWithPath: file.filename).lastPathComponent,
                    modelID: modelID
                )
            )
        } catch let error as ValarVoiceError {
            return voiceErrorResponse(error.localizedDescription, status: .badRequest)
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .internalServerError)
        }

        let dto = VoiceSummaryDTO(from: voice, preview: voice.backendVoiceID ?? voice.voicePrompt ?? voice.label)
        do {
            return try jsonResponse(dto, status: .created)
        } catch {
            return voiceErrorResponse("Failed to encode response.", status: .internalServerError)
        }
    }

    private static func handleVoiceStabilizeRequest(
        _ request: Request,
        runtime: ValarRuntime
    ) async -> Response {
        let decision = await mutationRateLimiter.acquire(action: "stabilize")
        guard decision.allowed else {
            return voiceRateLimitResponse(action: "stabilize", retryAfterSeconds: decision.retryAfterSeconds)
        }

        struct StabilizeBody: Decodable {
            let id: String
            let name: String?
            let anchorText: String?
            let model: String?

            enum CodingKeys: String, CodingKey {
                case id
                case name
                case anchorText = "anchor_text"
                case model
            }
        }

        let body: StabilizeBody
        do {
            let buffer = try await request.body.collect(upTo: 1_000_000)
            let data = Data(buffer: buffer)
            body = try JSONDecoder().decode(StabilizeBody.self, from: data)
        } catch {
            return voiceErrorResponse(
                "Invalid JSON body: \(error.localizedDescription)",
                status: .badRequest
            )
        }

        guard let voiceID = UUID(uuidString: body.id) else {
            return voiceErrorResponse("Field 'id' must be a UUID.", status: .badRequest)
        }
        if let name = body.name, name.count > 200 {
            return voiceErrorResponse("Field 'name' must not exceed 200 characters.", status: .badRequest)
        }
        if let anchorText = body.anchorText, anchorText.count > 4_000 {
            return voiceErrorResponse("Field 'anchor_text' must not exceed 4000 characters.", status: .badRequest)
        }

        let voice: VoiceLibraryRecord
        do {
            voice = try await runtime.stabilizeVoice(
                VoiceStabilizeRequest(
                    sourceVoiceID: voiceID,
                    label: voiceTrimmedOrNil(body.name),
                    anchorText: voiceTrimmedOrNil(body.anchorText),
                    modelID: voiceTrimmedOrNil(body.model).map { ModelIdentifier($0) }
                )
            )
        } catch let error as ValarVoiceError {
            let status: HTTPResponse.Status
            switch error {
            case .voiceNotFound:
                status = .notFound
            case .immutablePresetVoice:
                status = .conflict
            default:
                status = .badRequest
            }
            return voiceErrorResponse(error.localizedDescription, status: status)
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .internalServerError)
        }

        let dto = VoiceSummaryDTO(from: voice, preview: voice.backendVoiceID ?? voice.voicePrompt ?? voice.label)
        do {
            return try jsonResponse(dto, status: .created)
        } catch {
            return voiceErrorResponse("Failed to encode response.", status: .internalServerError)
        }
    }

    private static func handleVoiceDeleteRequest(
        _ request: Request,
        context: some RequestContext,
        runtime: ValarRuntime
    ) async -> Response {
        let decision = await mutationRateLimiter.acquire(action: "delete")
        guard decision.allowed else {
            return voiceRateLimitResponse(action: "delete", retryAfterSeconds: decision.retryAfterSeconds)
        }

        let idValue = context.parameters.get("id") ?? ""
        guard let voiceID = UUID(uuidString: idValue) else {
            return voiceErrorResponse("Voice id must be a UUID.", status: .badRequest)
        }

        do {
            try await runtime.deleteVoice(voiceID)
            return Response(status: .noContent)
        } catch let error as ValarVoiceError {
            let status: HTTPResponse.Status
            switch error {
            case .voiceNotFound:
                status = .notFound
            case .immutablePresetVoice:
                status = .conflict
            default:
                status = .badRequest
            }
            return voiceErrorResponse(error.localizedDescription, status: status)
        } catch {
            return voiceErrorResponse(error.localizedDescription, status: .internalServerError)
        }
    }

    // MARK: - Private helpers

    private static func voiceTrimmedOrNil(_ value: String?) -> String? {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              value.isEmpty == false else {
            return nil
        }
        return value
    }

    private static func voicePathExtension(from filename: String) -> String? {
        let ext = URL(fileURLWithPath: filename).pathExtension.lowercased()
        return ext.isEmpty ? nil : ext
    }

    private static func voiceErrorResponse(
        _ message: String,
        status: HTTPResponse.Status
    ) -> Response {
        daemonErrorResponse(message: message, status: status, kind: "voice_error")
    }

    private static func voiceRateLimitResponse(
        action: String,
        retryAfterSeconds: Int
    ) -> Response {
        let message =
            "Too many voice \(action) requests in a short window. " +
            "Wait \(retryAfterSeconds)s and try again."
        var response = daemonErrorResponse(
            message: message,
            status: .tooManyRequests,
            kind: "rate_limit",
            help: "Retry after \(retryAfterSeconds) seconds."
        )
        response.headers[.retryAfter] = "\(retryAfterSeconds)"
        return response
    }
}

// MARK: - Multipart parsing

private struct VoiceCloneFormData: Sendable {
    let fields: [String: String]
    let files: [String: VoiceCloneFile]

    struct VoiceCloneFile: Sendable {
        let filename: String
        let data: Data
    }

    static func parse(body: Data, contentType: String) throws -> VoiceCloneFormData {
        guard let boundary = extractBoundary(from: contentType) else {
            throw VoiceCloneParseError.missingBoundary
        }

        let delimiter = Data(("--" + boundary).utf8)
        let terminal = Data("--".utf8)
        let parts = splitByDelimiter(body, delimiter: delimiter)

        var fields: [String: String] = [:]
        var files: [String: VoiceCloneFile] = [:]

        for rawPart in parts.dropFirst() {
            var part = trimLeadingCRLF(rawPart)
            if part.starts(with: terminal) { continue }
            part = trimTrailingCRLF(part)
            guard part.isEmpty == false else { continue }

            let headerSeparator = Data("\r\n\r\n".utf8)
            guard let sepRange = part.range(of: headerSeparator) else {
                throw VoiceCloneParseError.malformedPart
            }

            let headerData = part.subdata(in: part.startIndex ..< sepRange.lowerBound)
            let valueData = part.subdata(in: sepRange.upperBound ..< part.endIndex)

            guard let headerText = String(data: headerData, encoding: .utf8) else {
                throw VoiceCloneParseError.malformedPart
            }

            var dispositionAttrs: [String: String] = [:]
            for line in headerText.components(separatedBy: "\r\n") {
                guard let colonIdx = line.firstIndex(of: ":") else { continue }
                let headerName = line[..<colonIdx]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                let headerValue = line[line.index(after: colonIdx)...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if headerName == "content-disposition" {
                    dispositionAttrs = parseDisposition(headerValue)
                }
            }

            guard let fieldName = dispositionAttrs["name"] else { continue }

            if let filename = dispositionAttrs["filename"] {
                files[fieldName] = VoiceCloneFile(filename: filename, data: valueData)
            } else {
                let text = String(data: valueData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                fields[fieldName] = text
            }
        }

        return VoiceCloneFormData(fields: fields, files: files)
    }

    private static func extractBoundary(from contentType: String) -> String? {
        for segment in contentType.components(separatedBy: ";") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("boundary=") else { continue }
            let value = String(trimmed.dropFirst("boundary=".count))
            let boundary = value.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            guard boundary.count <= 70 else {
                return nil  // RFC 2046 limits boundary to 70 chars
            }
            return boundary
        }
        return nil
    }

    private static func parseDisposition(_ value: String) -> [String: String] {
        var attrs: [String: String] = [:]
        for segment in value.components(separatedBy: ";") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let eqIdx = trimmed.firstIndex(of: "=") else { continue }
            let key = trimmed[..<eqIdx].trimmingCharacters(in: .whitespacesAndNewlines)
            let val = trimmed[trimmed.index(after: eqIdx)...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            attrs[key] = val
        }
        return attrs
    }

    private static func splitByDelimiter(_ data: Data, delimiter: Data) -> [Data] {
        var parts: [Data] = []
        var searchStart = data.startIndex
        while let range = data.range(of: delimiter, in: searchStart ..< data.endIndex) {
            parts.append(data[searchStart ..< range.lowerBound])
            searchStart = range.upperBound
        }
        parts.append(data[searchStart ..< data.endIndex])
        return parts
    }

    private static func trimLeadingCRLF(_ data: Data) -> Data {
        var start = data.startIndex
        while start < data.endIndex,
              data[start] == UInt8(ascii: "\r") || data[start] == UInt8(ascii: "\n") {
            start = data.index(after: start)
        }
        return Data(data[start...])
    }

    private static func trimTrailingCRLF(_ data: Data) -> Data {
        var end = data.endIndex
        while end > data.startIndex {
            let prev = data.index(before: end)
            if data[prev] == UInt8(ascii: "\r") || data[prev] == UInt8(ascii: "\n") {
                end = prev
            } else {
                break
            }
        }
        return Data(data[..<end])
    }
}

private enum VoiceCloneParseError: LocalizedError {
    case missingBoundary
    case malformedPart

    var errorDescription: String? {
        switch self {
        case .missingBoundary:
            return "Missing multipart boundary in Content-Type."
        case .malformedPart:
            return "Malformed multipart section."
        }
    }
}
