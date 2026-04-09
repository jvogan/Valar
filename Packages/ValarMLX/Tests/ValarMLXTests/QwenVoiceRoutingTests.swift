import Foundation
@testable import MLXAudioTTS
import MLXLMCommon
import Testing
@testable import ValarMLX
import ValarModelKit

@Suite("Qwen Voice Routing")
struct QwenVoiceRoutingTests {

    @Test("Legacy expressive Qwen voice routes through instruct")
    func legacyPromptRoutesToInstruct() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3 VoiceDesign",
            capabilities: [.speechSynthesis, .voiceDesign]
        )
        let voice = VoiceProfile(
            label: "Brooding noir narrator with smoked-baritone delivery.",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            voiceKind: .legacyPrompt,
            isLegacyExpressive: true
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Tell me a story.",
            voice: voice
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.voice == nil)
        #expect(input.speaker == nil)
        #expect(input.instruct == voice.voiceSelector)
        #expect(input.conditioning == nil)
        #expect(input.referenceAudio == nil)
        #expect(input.referenceText == nil)
    }

    @Test("Named-speaker Qwen request routes through speaker and preserves instruct")
    func namedSpeakerRoutesToSpeaker() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            displayName: "Qwen3 CustomVoice",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let voice = VoiceProfile(
            label: "Cherry",
            backendVoiceID: "Cherry",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            voiceKind: .namedSpeaker
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Announce the evening bulletin.",
            voice: voice,
            instruct: "Speak with deliberate emphasis."
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.voice == nil)
        #expect(input.speaker == "Cherry")
        #expect(input.instruct == "Speak with deliberate emphasis.")
        #expect(input.conditioning == nil)
        #expect(input.referenceAudio == nil)
        #expect(input.referenceText == nil)
    }

    @Test("Stable narrator reusable clone prompt keeps conditioning and skips heavyweight audio replay for short text")
    func stableNarratorReusableClonePromptUsesFastPathForShortText() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let payload = reusableClonePromptPayload()
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            referenceAudioAssetName: "architect-anchor.wav",
            referenceTranscript: "The bridge was silent except for the machines.",
            speakerEmbedding: payload,
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Continue the briefing.",
            voice: voice,
            referenceAudioAssetName: "architect-anchor.wav",
            referenceAudioSamples: [0.1, -0.1, 0.2, -0.2],
            referenceAudioSampleRate: 24_000,
            referenceTranscript: "This is the saved narrator anchor.",
            voiceBehavior: .stableNarrator
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.voice == nil)
        #expect(input.speaker == nil)
        #expect(input.instruct == nil)
        #expect(input.conditioning?.format == VoiceProfile.qwenClonePromptConditioningFormat)
        #expect(input.conditioning?.payload == payload)
        #expect(input.referenceAudio == nil)
        #expect(input.referenceText == "This is the saved narrator anchor.")
    }

    @Test("Stable narrator reusable clone prompt preserves reference replay for long-form text")
    func stableNarratorReusableClonePromptKeepsICLForLongForm() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let payload = reusableClonePromptPayload()
        let longText = Array(
            repeating: "The bridge stayed quiet while the machines kept watch over the lower deck, and the crew listened for the next instruction from the command deck.",
            count: 18
        ).joined(separator: " ")
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            referenceAudioAssetName: "architect-anchor.wav",
            referenceTranscript: "The bridge was silent except for the machines.",
            speakerEmbedding: payload,
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: longText,
            voice: voice,
            referenceTranscript: "This is the saved narrator anchor.",
            voiceBehavior: .stableNarrator
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.conditioning?.format == VoiceProfile.qwenClonePromptConditioningFormat)
        #expect(input.conditioning?.payload == payload)
        #expect(input.referenceAudio == nil)
        #expect(input.referenceText == "This is the saved narrator anchor.")
    }

    @Test("Clone prompt without reference audio keeps cached conditioning")
    func clonePromptWithoutReferenceAudioKeepsConditioning() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let payload = reusableClonePromptPayload()
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            speakerEmbedding: payload,
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Continue the briefing.",
            voice: voice,
            voiceBehavior: .stableNarrator
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.conditioning?.format == VoiceProfile.qwenClonePromptConditioningFormat)
        #expect(input.conditioning?.payload == payload)
        #expect(input.referenceAudio == nil)
        #expect(input.referenceText == nil)
    }

    @Test("Legacy clone prompt without reusable ref codes still keeps saved reference audio")
    func legacyClonePromptRetainsReferenceAudioReplay() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let legacyEmbedding = float32Data([0.25, -0.5, 1.0, 0.75])
        let voice = VoiceProfile(
            label: "Legacy Stable Voice",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            referenceAudioAssetName: "legacy-anchor.wav",
            referenceTranscript: "Legacy narrator anchor.",
            speakerEmbedding: legacyEmbedding,
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Continue the archive.",
            voice: voice,
            referenceAudioAssetName: "legacy-anchor.wav",
            referenceAudioSamples: [0.1, 0.2, -0.1, -0.2],
            referenceAudioSampleRate: 24_000,
            referenceTranscript: "Legacy narrator anchor.",
            voiceBehavior: .stableNarrator
        )

        let input = MLXModelHandle.resolvedSpeechGenerationInput(from: request, descriptor: descriptor)

        #expect(input.conditioning?.format == VoiceProfile.qwenClonePromptConditioningFormat)
        #expect(input.conditioning?.payload == legacyEmbedding)
        #expect(input.referenceAudio != nil)
        #expect(input.referenceText == "Legacy narrator anchor.")
    }

    @Test("Inline reference audio and transcript auto-select stable narrator")
    func inlineReferenceAutoSelectsStableNarrator() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "This should follow the same narrator.",
            referenceAudioSamples: [0.2, -0.2, 0.15, -0.15],
            referenceAudioSampleRate: 24_000,
            referenceTranscript: "Use this anchor voice for narration."
        )

        let behavior = MLXModelHandle.resolvedVoiceBehavior(for: request, descriptor: descriptor)
        let input = MLXModelHandle.resolvedSpeechGenerationInput(
            from: request,
            descriptor: descriptor,
            voiceBehavior: behavior
        )

        #expect(behavior == .stableNarrator)
        #expect(input.voice == nil)
        #expect(input.speaker == nil)
        #expect(input.instruct == nil)
        #expect(input.referenceAudio != nil)
        #expect(input.referenceText == "Use this anchor voice for narration.")
    }

    @Test("Stable narrator continuation can downgrade reusable clone prompts to speaker-embedding fast path")
    func stableNarratorContinuationUsesEmbeddingFastPath() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let payload = reusableClonePromptPayload()
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            referenceAudioAssetName: "architect-anchor.wav",
            referenceTranscript: "The bridge was silent except for the machines.",
            speakerEmbedding: payload,
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )

        let continuationVoice = MLXModelHandle.qwenEmbeddingOnlyContinuationVoice(from: voice)

        #expect(continuationVoice != nil)
        #expect(continuationVoice?.conditioningFormat == VoiceProfile.qwenSpeakerEmbeddingConditioningFormat)
        #expect(continuationVoice?.voiceKind == .embeddingOnly)
        #expect(continuationVoice?.speakerEmbedding != payload)
    }

    @Test("Stable narrator anchor schedule replays the full clone prompt on the first segment and then follows the active cadence")
    func stableNarratorAnchorSchedule() {
        #expect(MLXModelHandle.qwenShouldUseStableAnchorSegment(0))
        #expect(!MLXModelHandle.qwenShouldUseStableAnchorSegment(1))
        #expect(!MLXModelHandle.qwenShouldUseStableAnchorSegment(2))
        #expect(!MLXModelHandle.qwenShouldUseStableAnchorSegment(3))
        #expect(MLXModelHandle.qwenShouldUseStableAnchorSegment(4))
    }

    @Test("Long stable narrator runs widen the anchor cadence while shorter runs keep the tighter cadence")
    func stableNarratorAnchorCadenceScalesWithSegmentCount() {
        #expect(MLXModelHandle.qwenStableNarratorAnchorEverySegments(segmentCount: 3) == 4)
        #expect(MLXModelHandle.qwenStableNarratorAnchorEverySegments(segmentCount: 12) == 6)
        #expect(MLXModelHandle.qwenStableNarratorAnchorEverySegments(segmentCount: 16) == 8)
        #expect(MLXModelHandle.qwenStableNarratorAnchorEverySegments(segmentCount: 20) == 8)
    }
}

@Suite("Qwen Generation Parameters")
struct QwenGenerationParameterTests {

    @Test("Qwen reading-time budget keeps very short expressive prompts lean")
    func shortExpressivePromptUsesConservativeBudget() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3 VoiceDesign",
            capabilities: [.speechSynthesis, .voiceDesign]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "System check complete."
        )

        let params = MLXModelHandle.resolvedGenerationParameters(
            from: request,
            defaults: GenerateParameters(maxTokens: 8_192),
            descriptor: descriptor,
            behavior: .expressive,
            text: request.text
        )

        #expect(params.maxTokens == 24)
    }

    @Test("Qwen stable narrator budget keeps very short prompts lean")
    func shortStableNarratorPromptUsesLowerFloor() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "System check complete."
        )

        let params = MLXModelHandle.resolvedGenerationParameters(
            from: request,
            defaults: GenerateParameters(maxTokens: 8_192),
            descriptor: descriptor,
            behavior: .stableNarrator,
            text: request.text
        )

        #expect(params.maxTokens == 24)
    }

    @Test("Qwen stable narrator budget grows with narration length but stays below ceiling")
    func stableNarratorBudgetScalesWithTextLength() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let longText = Array(repeating: "The bridge stayed quiet while the machines kept watch.", count: 80)
            .joined(separator: " ")
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: longText
        )

        let params = MLXModelHandle.resolvedGenerationParameters(
            from: request,
            defaults: GenerateParameters(maxTokens: 8_192),
            descriptor: descriptor,
            behavior: .stableNarrator,
            text: request.text
        )

        let maxTokens = params.maxTokens ?? 0
        #expect(maxTokens > 64)
        #expect(maxTokens <= 2_048)
    }

    @Test("Explicit Qwen max token override still wins over recommendation")
    func explicitMaxTokenOverrideStillWins() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "A long-form narration request that would otherwise get a larger automatic budget.",
            maxTokens: 321
        )

        let params = MLXModelHandle.resolvedGenerationParameters(
            from: request,
            defaults: GenerateParameters(maxTokens: 8_192),
            descriptor: descriptor,
            behavior: .stableNarrator,
            text: request.text
        )

        #expect(params.maxTokens == 321)
    }

    @Test("Short stable narrator requests stay one-shot with the short stable ceiling")
    func shortStableNarratorExecutionPlan() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "Keep the systems online for the next report.",
            voiceBehavior: .stableNarrator
        )

        let plan = MLXModelHandle.plannedQwenExecution(for: request, descriptor: descriptor)

        #expect(plan.behavior == .stableNarrator)
        #expect(plan.mode == .oneShot)
        #expect(plan.segmentCount == 1)
        #expect(plan.automaticMaxTokenCeiling == 1_536)
    }

    @Test("Medium stable narrator requests switch directly to segmented continuation")
    func mediumStableNarratorExecutionPlan() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let text = Array(
            repeating: "The bridge stayed quiet while the machines kept watch over the lower deck, and the crew listened for the next instruction from the command deck.",
            count: 12
        ).joined(separator: " ")
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: text,
            voiceBehavior: .stableNarrator
        )

        let plan = MLXModelHandle.plannedQwenExecution(for: request, descriptor: descriptor)

        #expect(plan.behavior == .stableNarrator)
        #expect(plan.mode == .segmentedContinuation)
        #expect(plan.segmentCount > 1)
        #expect(plan.automaticMaxTokenCeiling == 2_048)
    }

    @Test("Medium expressive requests switch to paragraph continuation with the expressive ceiling")
    func mediumExpressiveExecutionPlan() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3 VoiceDesign",
            capabilities: [.speechSynthesis, .voiceDesign]
        )
        let paragraph = "The city listened as the announcer moved from one district bulletin to the next, with enough detail to force a long-form expressive continuation path."
        let text = Array(repeating: paragraph, count: 18).joined(separator: "\n\n")
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: text,
            voiceBehavior: .expressive
        )

        let plan = MLXModelHandle.plannedQwenExecution(for: request, descriptor: descriptor)

        #expect(plan.behavior == .expressive)
        #expect(plan.mode == .segmentedContinuation)
        #expect(plan.segmentCount > 1)
        #expect(plan.automaticMaxTokenCeiling == 3_072)
    }
}

@Suite("Qwen Orchestration Planning")
struct QwenOrchestrationPlanningTests {

    @Test("Short stable narrator requests stay one-shot")
    func shortStableNarratorUsesOneShotPlan() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            speakerEmbedding: reusableClonePromptPayload(),
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: "System check complete.",
            voice: voice
        )

        let plan = QwenSpeechOrchestrator.plan(for: MLXModelHandle(descriptor: descriptor), request: request)

        #expect(plan.behavior == .stableNarrator)
        #expect(plan.mode == .oneShot)
        #expect(plan.chunkPlan.chunks.count == 1)
        #expect(plan.automaticMaxTokenCeiling == 1_536)
    }

    @Test("Medium stable narrator requests switch directly to segmented continuation")
    func mediumStableNarratorUsesSegmentedContinuation() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            displayName: "Qwen3 Base",
            capabilities: [.speechSynthesis, .voiceCloning]
        )
        let voice = VoiceProfile(
            label: "The Architect Dark Stable",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            speakerEmbedding: reusableClonePromptPayload(),
            conditioningFormat: VoiceProfile.qwenClonePromptConditioningFormat,
            voiceKind: .clonePrompt
        )
        let text = Array(
            repeating: "The command deck remained still while the narrator outlined the next phase of the operation in careful, even detail for the entire crew.",
            count: 18
        ).joined(separator: " ")
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: text,
            voice: voice
        )

        let plan = QwenSpeechOrchestrator.plan(for: MLXModelHandle(descriptor: descriptor), request: request)

        #expect(plan.behavior == .stableNarrator)
        #expect(plan.mode == .segmentedContinuation)
        #expect(plan.chunkPlan.chunks.count > 1)
        #expect(plan.automaticMaxTokenCeiling == 2_048)
    }

    @Test("Medium expressive legacy voices segment with the expressive policy")
    func mediumExpressiveUsesExpressiveSegmentation() {
        let descriptor = qwenDescriptor(
            id: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            displayName: "Qwen3 VoiceDesign",
            capabilities: [.speechSynthesis, .voiceDesign]
        )
        let voice = VoiceProfile(
            label: "The Architect Dark",
            sourceModel: descriptor.id,
            runtimeModel: descriptor.id,
            voiceKind: .legacyPrompt,
            isLegacyExpressive: true
        )
        let paragraph = "The narrator moves through the station with a controlled, cinematic cadence, pausing long enough for each line to land before the next image arrives."
        let text = Array(repeating: paragraph, count: 14).joined(separator: "\n\n")
        let request = SpeechSynthesisRequest(
            model: descriptor.id,
            text: text,
            voice: voice
        )

        let plan = QwenSpeechOrchestrator.plan(for: MLXModelHandle(descriptor: descriptor), request: request)

        #expect(plan.behavior == .expressive)
        #expect(plan.mode == .segmentedContinuation)
        #expect(plan.chunkPlan.chunks.count > 1)
        #expect(plan.automaticMaxTokenCeiling == 3_072)
    }
}

private func qwenDescriptor(
    id: String,
    displayName: String,
    capabilities: [ModelCapability]
) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(id),
        familyID: .qwen3TTS,
        displayName: displayName,
        domain: .tts,
        capabilities: Set(capabilities)
    )
}

private func float32Data(_ values: [Float]) -> Data {
    values.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            return Data()
        }
        return Data(bytes: baseAddress, count: buffer.count * MemoryLayout<Float>.size)
    }
}

private func reusableClonePromptPayload() -> Data {
    struct Payload: Encodable {
        let version: Int
        let refSpeakerEmbedding: Data?
        let refCode: Data?
        let numCodeGroups: Int?
        let frameCount: Int?
        let xVectorOnlyMode: Bool
        let iclMode: Bool
    }

    let speakerEmbedding = float32Data([0.25, -0.5, 1.0, 0.75])
    let refCodes: [Int32] = [11, 12, 21, 22, 31, 32]
    let refCodeData = refCodes.withUnsafeBufferPointer { Data(buffer: $0) }
    let payload = Payload(
        version: 1,
        refSpeakerEmbedding: speakerEmbedding,
        refCode: refCodeData,
        numCodeGroups: 2,
        frameCount: 3,
        xVectorOnlyMode: false,
        iclMode: true
    )
    return try! JSONEncoder().encode(payload)
}
