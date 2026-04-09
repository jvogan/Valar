import Foundation
import Testing
@testable import ValarMLX
import ValarModelKit

@Suite("QwenTextChunker")
struct QwenTextChunkerTests {

    @Test("Empty text returns empty chunk plan")
    func emptyText() {
        let plan = QwenTextChunker.plan(text: "", behavior: .expressive)
        #expect(plan.chunks.isEmpty)
        #expect(!plan.isLongForm)
    }

    @Test("Short text stays single-shot")
    func shortTextSingleShot() {
        let text = "Hello, world!"
        let plan = QwenTextChunker.plan(text: text, behavior: .expressive)
        #expect(plan.chunks.count == 1)
        #expect(plan.chunks[0] == text)
        #expect(!plan.isLongForm)
    }

    @Test("Expressive text at target threshold stays single-shot")
    func expressiveTextAtThreshold() {
        let text = String(repeating: "a", count: 1_500)
        let plan = QwenTextChunker.plan(text: text, behavior: .expressive)
        #expect(plan.chunks.count == 1)
        #expect(!plan.isLongForm)
    }

    @Test("Expressive text can switch to long-form on word count alone")
    func expressiveWordThresholdTriggersLongForm() {
        let text = Array(repeating: "word", count: 260).joined(separator: " ")
        #expect(text.count < 1_500)
        let plan = QwenTextChunker.plan(text: text, behavior: .expressive)
        #expect(plan.isLongForm)
        #expect(plan.chunks.count > 1)
    }

    @Test("Long newline-separated text produces multiple chunks")
    func longNewlineTextMultiChunk() {
        let lines = (1...22).map {
            "Paragraph \($0): This paragraph is intentionally long so the expressive policy has to keep newline boundaries while planning long-form narration for Qwen."
        }
        let text = lines.joined(separator: "\n")
        let plan = QwenTextChunker.plan(text: text, behavior: .expressive)
        #expect(plan.isLongForm)
        #expect(plan.chunks.count > 1)
        let rejoined = plan.chunks.joined(separator: "\n\n")
        for line in lines {
            #expect(rejoined.contains(line))
        }
    }

    @Test("Only oversized paragraphs fall back to sentence splitting")
    func oversizedParagraphSentenceFallback() {
        let sentences = (1...80).map { "This is sentence number \($0) in a very long paragraph." }
        let text = sentences.joined(separator: " ")
        #expect(text.count > 3_400)
        let plan = QwenTextChunker.plan(text: text, behavior: .expressive)
        #expect(plan.isLongForm)
        #expect(plan.chunks.count > 1)
        for chunk in plan.chunks {
            #expect(chunk.count <= 2_600)
        }
    }

    @Test("Stable narrator policy prefers smaller continuation segments than expressive mode")
    func stableNarratorUsesSmallerTargetChunks() {
        let paragraph = String(repeating: "a", count: 700)
        let text = "\(paragraph)\n\n\(paragraph)\n\n\(paragraph)"
        let expressivePlan = QwenTextChunker.plan(text: text, behavior: .expressive)
        let stablePlan = QwenTextChunker.plan(text: text, behavior: .stableNarrator)
        #expect(expressivePlan.isLongForm)
        #expect(stablePlan.isLongForm)
        #expect(stablePlan.chunks.count > expressivePlan.chunks.count)
    }

    @Test("Chunk plan is deterministic")
    func deterministic() {
        let text = (1...20).map {
            "Paragraph \($0): The quick brown fox jumps over the lazy dog while the narrator maintains a very steady delivery across long-form sections."
        }
            .joined(separator: "\n")
        let plan1 = QwenTextChunker.plan(text: text, behavior: .stableNarrator)
        let plan2 = QwenTextChunker.plan(text: text, behavior: .stableNarrator)
        #expect(plan1.chunks == plan2.chunks)
        #expect(plan1.isLongForm == plan2.isLongForm)
    }

    @Test("Whitespace-only text returns empty plan")
    func whitespaceOnly() {
        let plan = QwenTextChunker.plan(text: "   \n\t  ", behavior: .expressive)
        #expect(plan.chunks.isEmpty)
        #expect(!plan.isLongForm)
    }

    @Test("Single long chunk count is greater than single short chunk count")
    func longFormHasMoreChunksThanShort() {
        let short = "Hello, this is a brief sentence."
        let long = (1...24).map {
            "Line \($0): The quick brown fox jumps over the lazy dog repeatedly while the speaker keeps a smooth and consistent cadence for a very long answer."
        }
            .joined(separator: "\n")
        let shortPlan = QwenTextChunker.plan(text: short, behavior: .expressive)
        let longPlan = QwenTextChunker.plan(text: long, behavior: .expressive)
        #expect(longPlan.chunks.count > shortPlan.chunks.count)
    }
}
