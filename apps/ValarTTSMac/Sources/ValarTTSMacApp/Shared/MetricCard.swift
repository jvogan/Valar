import SwiftUI

struct MetricCard: View {
    let title: String
    let value: String
    let detail: String
    let symbol: String

    var body: some View {
        SurfaceCard(title: title, symbol: symbol) {
            VStack(alignment: .leading, spacing: 8) {
                Text(value)
                    .font(.title.bold())
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
