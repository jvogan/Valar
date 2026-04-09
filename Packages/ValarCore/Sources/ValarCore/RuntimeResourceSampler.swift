import Foundation

public actor RuntimeResourceSampler {
    public struct Snapshot: Sendable, Equatable {
        public let processFootprintBytes: Int
        public let processFootprintHighWaterBytes: Int
        public let processCPUCurrentPercent: Double?
        public let processCPUCurrentHighWaterPercent: Double?
        public let processCPUAveragePercentSinceStart: Double?
        public let processCPUAverageHighWaterPercentSinceStart: Double?
        public let availableDiskBytes: Int?
        public let availableDiskLowWaterBytes: Int?
        public let collectedAt: Date
    }

    private var previousCPUSeconds: Double?
    private var previousCollectedAt: Date?
    private var processFootprintHighWaterBytes: Int = 0
    private var processCPUCurrentHighWaterPercent: Double?
    private var processCPUAverageHighWaterPercentSinceStart: Double?
    private var availableDiskLowWaterBytes: Int?

    public init() {}

    public func sample(
        applicationSupportURL: URL,
        uptimeSeconds: Double
    ) -> Snapshot {
        let now = Date()
        let footprint = RuntimeResourceMonitor.currentProcessFootprintBytes()
        let cpuSeconds = RuntimeResourceMonitor.currentProcessCPUTimeSeconds()
        let averageCPU = RuntimeResourceMonitor.currentProcessCPUPercent(uptimeSeconds: uptimeSeconds)
        let availableDisk = RuntimeResourceMonitor.availableDiskBytes(at: applicationSupportURL)

        let currentCPU: Double?
        if let cpuSeconds, let previousCPUSeconds, let previousCollectedAt {
            let wallDelta = now.timeIntervalSince(previousCollectedAt)
            let cpuDelta = cpuSeconds - previousCPUSeconds
            if wallDelta > 0, cpuDelta >= 0 {
                currentCPU = max(0, (cpuDelta / wallDelta) * 100)
            } else {
                currentCPU = averageCPU
            }
        } else {
            currentCPU = averageCPU
        }

        previousCPUSeconds = cpuSeconds
        previousCollectedAt = now
        processFootprintHighWaterBytes = max(processFootprintHighWaterBytes, footprint)
        if let currentCPU {
            processCPUCurrentHighWaterPercent = max(processCPUCurrentHighWaterPercent ?? currentCPU, currentCPU)
        }
        if let averageCPU {
            processCPUAverageHighWaterPercentSinceStart = max(
                processCPUAverageHighWaterPercentSinceStart ?? averageCPU,
                averageCPU
            )
        }
        if let availableDisk {
            if let existing = availableDiskLowWaterBytes {
                availableDiskLowWaterBytes = min(existing, availableDisk)
            } else {
                availableDiskLowWaterBytes = availableDisk
            }
        }

        return Snapshot(
            processFootprintBytes: footprint,
            processFootprintHighWaterBytes: processFootprintHighWaterBytes,
            processCPUCurrentPercent: currentCPU,
            processCPUCurrentHighWaterPercent: processCPUCurrentHighWaterPercent,
            processCPUAveragePercentSinceStart: averageCPU,
            processCPUAverageHighWaterPercentSinceStart: processCPUAverageHighWaterPercentSinceStart,
            availableDiskBytes: availableDisk,
            availableDiskLowWaterBytes: availableDiskLowWaterBytes,
            collectedAt: now
        )
    }
}
