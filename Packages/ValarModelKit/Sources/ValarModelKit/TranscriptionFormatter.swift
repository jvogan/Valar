import Foundation

/// Formats a `RichTranscriptionResult` as SRT or WebVTT subtitle text.
///
/// Both formatters use the segment-level timing from `RichTranscriptionResult.segments`.
/// When segment timing is absent the formatters fall back to equal-duration distribution
/// across the total audio length (or a 5-second-per-segment estimate when duration is also
/// unknown).
public enum TranscriptionFormatter {

    // MARK: - Public API

    /// Renders the result as SRT (SubRip) subtitle text.
    ///
    /// Format:
    /// ```
    /// 1
    /// 00:00:00,000 --> 00:00:05,000
    /// First segment text
    ///
    /// 2
    /// 00:00:05,000 --> 00:00:10,000
    /// Second segment text
    /// ```
    public static func srt(from result: RichTranscriptionResult) -> String {
        let cues = timedCues(from: result)
        guard !cues.isEmpty else { return "" }

        return cues.enumerated().map { index, cue in
            let start = srtTimestamp(cue.start)
            let end   = srtTimestamp(cue.end)
            return "\(index + 1)\n\(start) --> \(end)\n\(cue.text)"
        }.joined(separator: "\n\n")
    }

    /// Renders the result as WebVTT subtitle text.
    ///
    /// Format:
    /// ```
    /// WEBVTT
    ///
    /// 00:00:00.000 --> 00:00:05.000
    /// First segment text
    ///
    /// 00:00:05.000 --> 00:00:10.000
    /// Second segment text
    /// ```
    public static func vtt(from result: RichTranscriptionResult) -> String {
        let cues = timedCues(from: result)

        var lines = ["WEBVTT", ""]
        for cue in cues {
            let start = vttTimestamp(cue.start)
            let end   = vttTimestamp(cue.end)
            lines.append("\(start) --> \(end)")
            lines.append(cue.text)
            lines.append("")
        }
        // Remove trailing blank line added after last cue
        if lines.last == "" { lines.removeLast() }
        return lines.joined(separator: "\n")
    }

    // MARK: - Internal helpers

    struct TimedCue {
        let start: Double
        let end: Double
        let text: String
    }

    /// Resolves a list of (start, end, text) triples from the result segments,
    /// filling in missing timing with estimated values.
    static func timedCues(from result: RichTranscriptionResult) -> [TimedCue] {
        let segments = result.segments.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !segments.isEmpty else { return [] }

        let totalDuration = result.durationSeconds

        // Check if ALL segments lack timing — if so, distribute evenly.
        let anyHasTiming = segments.contains { $0.startTime != nil || $0.endTime != nil }
        if !anyHasTiming {
            return distribute(segments: segments, over: totalDuration)
        }

        // At least some segments have timing. Fill gaps where needed.
        var cues: [TimedCue] = []
        for (i, segment) in segments.enumerated() {
            let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)

            let start: Double
            let end: Double

            if let s = segment.startTime {
                start = s
            } else if i == 0 {
                start = 0.0
            } else if let prevEnd = cues.last?.end {
                start = prevEnd
            } else {
                start = Double(i) * 5.0
            }

            if let e = segment.endTime {
                end = e
            } else {
                // Peek at the next segment's startTime to close the gap neatly.
                let nextStart = segments[safe: i + 1]?.startTime
                if let ns = nextStart {
                    end = ns
                } else if let dur = totalDuration {
                    // Distribute remaining duration over segments that lack endTime.
                    let remainingSegments = segments[(i)...].count
                    let perSegment = max(0.5, (dur - start) / Double(remainingSegments))
                    end = start + perSegment
                } else {
                    end = start + 5.0
                }
            }

            cues.append(TimedCue(start: max(0, start), end: max(max(0, start) + 0.001, end), text: text))
        }
        return cues
    }

    private static func distribute(segments: [TranscriptionSegment], over totalDuration: Double?) -> [TimedCue] {
        let count = segments.count
        let perSegment: Double
        if let dur = totalDuration, dur > 0 {
            perSegment = dur / Double(count)
        } else {
            perSegment = 5.0
        }
        return segments.enumerated().map { i, segment in
            let start = Double(i) * perSegment
            let end   = start + perSegment
            let text  = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return TimedCue(start: start, end: end, text: text)
        }
    }

    // MARK: - Timestamp formatting

    /// Formats seconds as `HH:MM:SS,mmm` (SRT).
    static func srtTimestamp(_ seconds: Double) -> String {
        formatTimestamp(seconds, decimalSeparator: ",")
    }

    /// Formats seconds as `HH:MM:SS.mmm` (VTT).
    static func vttTimestamp(_ seconds: Double) -> String {
        formatTimestamp(seconds, decimalSeparator: ".")
    }

    private static func formatTimestamp(_ seconds: Double, decimalSeparator: String) -> String {
        let clamped = max(0, seconds)
        let totalMs = Int((clamped * 1000).rounded())
        let ms = totalMs % 1000
        let totalSec = totalMs / 1000
        let sec = totalSec % 60
        let totalMin = totalSec / 60
        let min = totalMin % 60
        let hrs = totalMin / 60
        return String(format: "%02d:%02d:%02d%@%03d", hrs, min, sec, decimalSeparator, ms)
    }
}

// MARK: - Collection safe-subscript (file-private)

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
