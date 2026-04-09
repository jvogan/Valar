import SwiftUI

struct MemoryPressureView: View {
    let usedBytes: Int
    let budgetBytes: Int

    private var fraction: Double {
        guard budgetBytes > 0 else { return 0 }
        return min(Double(usedBytes) / Double(budgetBytes), 1.0)
    }

    private var tintColor: Color {
        if fraction < 0.6 { return StatusColor.success }
        if fraction < 0.85 { return StatusColor.warning }
        return StatusColor.error
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "memorychip")
                .font(.caption2)
                .foregroundStyle(tintColor.opacity(0.8))

            Gauge(value: fraction) {}
                .gaugeStyle(.accessoryLinearCapacity)
                .tint(tintColor)
                .frame(width: 56)

            Text("\(formatBytes(usedBytes)) / \(formatBytes(budgetBytes))")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .help("Process memory: \(formatBytes(usedBytes)) of \(formatBytes(budgetBytes)) headroom (\(Int(fraction * 100))%)")
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gib = Double(bytes) / (1024 * 1024 * 1024)
        return String(format: "%.1f GiB", gib)
    }
}
