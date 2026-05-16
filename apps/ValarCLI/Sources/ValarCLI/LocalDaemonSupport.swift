import Foundation

enum CLILocalDaemon {
    private static let allowedHosts = Set(["127.0.0.1", "::1", "localhost"])

    static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 2
        configuration.timeoutIntervalForResource = 4
        return URLSession(configuration: configuration)
    }()

    static func baseURL(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        let rawHost = trimmedNonEmpty(environment["VALARTTSD_BIND_HOST"]) ?? "127.0.0.1"
        let rawPort = trimmedNonEmpty(environment["VALARTTSD_BIND_PORT"]) ?? "8787"

        let host = rawHost.trimmingCharacters(in: CharacterSet(charactersIn: "[]")).lowercased()
        guard allowedHosts.contains(host),
              let port = Int(rawPort),
              (1...65_535).contains(port)
        else {
            return nil
        }

        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        return components.url
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else {
            return nil
        }
        return trimmed
    }
}
