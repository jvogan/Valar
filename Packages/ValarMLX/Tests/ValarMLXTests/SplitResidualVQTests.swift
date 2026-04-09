import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
@testable import ValarMLX

@Suite("SplitResidualVQ")
struct SplitResidualVQTests {

    @Test("EuclideanCodebook decode performs correct embedding lookup")
    func euclideanCodebookDecode() throws {
        let codebook = EuclideanCodebook(dim: 3, codebookSize: 4)
        // Set known embedding weights via update.
        let weight = MLXArray(
            [Float(1), 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
        ).reshaped(4, 3)
        try codebook.update(
            parameters: ModuleParameters.unflattened([("embed.weight", weight)]),
            verify: .noUnusedKeys
        )

        let codes = MLXArray([Int32(0), 2, 3])
        let result = codebook.decode(codes)
        MLX.eval(result)

        #expect(result.shape == [3, 3])
        let flat = result.reshaped(-1).asArray(Float.self)
        // code 0 → [1, 0, 0]
        #expect(flat[0] == 1.0)
        #expect(flat[1] == 0.0)
        #expect(flat[2] == 0.0)
        // code 2 → [0, 0, 1]
        #expect(flat[3] == 0.0)
        #expect(flat[4] == 0.0)
        #expect(flat[5] == 1.0)
        // code 3 → [1, 1, 1]
        #expect(flat[6] == 1.0)
        #expect(flat[7] == 1.0)
        #expect(flat[8] == 1.0)
    }

    @Test("VectorQuantization decode returns NCL format")
    func vectorQuantizationDecode() throws {
        let vq = VectorQuantization(dim: 2, codebookSize: 3)
        let weight = MLXArray(
            [Float(1), 2, 3, 4, 5, 6]
        ).reshaped(3, 2)
        try vq.update(
            parameters: ModuleParameters.unflattened([("codebook.embed.weight", weight)]),
            verify: .noUnusedKeys
        )

        // codes: [batch=1, time=3]
        let codes = MLXArray([Int32(0), 1, 2]).reshaped(1, 3)
        let result = vq.decode(codes)
        MLX.eval(result)

        // Output: [batch=1, dim=2, time=3] (NCL)
        #expect(result.shape == [1, 2, 3])

        // Embedding gives NLC [[1,2],[3,4],[5,6]], transposed to NCL:
        // dim0: [1, 3, 5], dim1: [2, 4, 6]
        let flat = result.reshaped(-1).asArray(Float.self)
        #expect(flat[0] == 1.0)
        #expect(flat[1] == 3.0)
        #expect(flat[2] == 5.0)
        #expect(flat[3] == 2.0)
        #expect(flat[4] == 4.0)
        #expect(flat[5] == 6.0)
    }

    @Test("ResidualVectorQuantization sums multiple codebook outputs")
    func residualVQSumsOutputs() throws {
        let rvq = ResidualVectorQuantization(numQuantizers: 2, dim: 2, codebookSize: 4)

        // Layer 0: code 0→[1,0], code 1→[0,1]
        let w0 = MLXArray([Float(1), 0, 0, 1, 0, 0, 0, 0]).reshaped(4, 2)
        // Layer 1: code 2→[10,0], code 3→[0,20]
        let w1 = MLXArray([Float(0), 0, 0, 0, 10, 0, 0, 20]).reshaped(4, 2)
        try rvq.update(
            parameters: ModuleParameters.unflattened([
                ("layers.0.codebook.embed.weight", w0),
                ("layers.1.codebook.embed.weight", w1),
            ]),
            verify: .noUnusedKeys
        )

        // codes: [numQuantizers=2, batch=1, time=2]
        let codes = MLXArray([Int32(0), 1, 2, 3]).reshaped(2, 1, 2)
        let result = rvq.decode(codes)
        MLX.eval(result)

        #expect(result.shape == [1, 2, 2])

        // Layer 0 NCL: dim0=[1,0], dim1=[0,1]
        // Layer 1 NCL: dim0=[10,0], dim1=[0,20]
        // Sum:         dim0=[11,0], dim1=[0,21]
        let flat = result.reshaped(-1).asArray(Float.self)
        #expect(flat[0] == 11.0)
        #expect(flat[1] == 0.0)
        #expect(flat[2] == 0.0)
        #expect(flat[3] == 21.0)
    }

    @Test("ResidualVectorQuantization returns zeros when codes exceed available layers")
    func residualVQReturnsZerosForOutOfBoundsCodes() throws {
        let rvq = ResidualVectorQuantization(numQuantizers: 1, dim: 2, codebookSize: 4)

        let weight = MLXArray([Float(1), 0, 0, 1, 0, 0, 0, 0]).reshaped(4, 2)
        try rvq.update(
            parameters: ModuleParameters.unflattened([
                ("layers.0.codebook.embed.weight", weight),
            ]),
            verify: .noUnusedKeys
        )

        // codes: [numQuantizers=2, batch=1, time=2] while only one layer exists.
        let codes = MLXArray([Int32(0), 1, 2, 3]).reshaped(2, 1, 2)
        let result = rvq.decode(codes)
        MLX.eval(result)

        #expect(result.shape == [1, 2, 2])
        let flat = result.reshaped(-1).asArray(Float.self)
        #expect(flat == [0.0, 0.0, 0.0, 0.0])
    }

    @Test("SplitResidualVectorQuantizer decode produces correct output shape")
    func splitRVQOutputShape() {
        let quantizer = SplitResidualVectorQuantizer(
            nQ: 3, nQSemantic: 1,
            dimension: 4, inputDimension: 8,
            outputDimension: 8, bins: 16
        )

        // codes: [batch=1, numQuantizers=3, time=5]
        let codes = MLXArray(Array(repeating: Int32(0), count: 15)).reshaped(1, 3, 5)
        let result = quantizer.decode(codes)
        MLX.eval(result)

        // Output: [batch=1, outputDimension=8, time=5]
        #expect(result.shape == [1, 8, 5])
    }

    @Test("SplitResidualVectorQuantizer splits semantic and acoustic correctly")
    func splitRVQSemanticAcousticSplit() throws {
        // dimension=2, inputDimension=2, outputDimension=2 with identity projections.
        let quantizer = SplitResidualVectorQuantizer(
            nQ: 2, nQSemantic: 1,
            dimension: 2, inputDimension: 2,
            outputDimension: 2, bins: 4
        )

        // Identity Conv1d weight: [outC, kW=1, inC] = [[1,0],[0,1]] reshaped to [2,1,2]
        let identity = MLXArray([Float(1), 0, 0, 1]).reshaped(2, 1, 2)
        // Semantic codebook: code 0→[1,0], code 1→[0,1]
        let wSemantic = MLXArray([Float(1), 0, 0, 1, 0, 0, 0, 0]).reshaped(4, 2)
        // Acoustic codebook: code 2→[10,0], code 3→[0,20]
        let wAcoustic = MLXArray([Float(0), 0, 0, 0, 10, 0, 0, 20]).reshaped(4, 2)

        let sanitized = SplitResidualVectorQuantizer.sanitize(weights: [
            "rvq_first.output_proj.weight": identity,
            "rvq_first.vq.layers.0.codebook.embed.weight": wSemantic,
            "rvq_rest.output_proj.weight": identity,
            "rvq_rest.vq.layers.0.codebook.embed.weight": wAcoustic,
        ])

        try quantizer.update(
            parameters: ModuleParameters.unflattened(Array(sanitized)),
            verify: .noUnusedKeys
        )

        // codes: [batch=1, numQuantizers=2, time=2]
        // Semantic codes: [0, 1], Acoustic codes: [2, 3]
        let codes = MLXArray([Int32(0), 1, 2, 3]).reshaped(1, 2, 2)
        let result = quantizer.decode(codes)
        MLX.eval(result)

        #expect(result.shape == [1, 2, 2])

        // Semantic: code 0→[1,0], code 1→[0,1] → NCL: [[1,0],[0,1]]
        // Acoustic: code 2→[10,0], code 3→[0,20] → NCL: [[10,0],[0,20]]
        // Sum: [[11,0],[0,21]]
        let flat = result.reshaped(-1).asArray(Float.self)
        #expect(flat[0] == 11.0)
        #expect(flat[1] == 0.0)
        #expect(flat[2] == 0.0)
        #expect(flat[3] == 21.0)
    }

    @Test("Sanitize computes embeddings from cluster_usage and embedding_sum")
    func sanitizeComputesEmbeddings() {
        let clusterUsage = MLXArray([Float(2), 4, 1, 0.5])
        let embeddingSum = MLXArray(
            [Float(2), 4, 8, 12, 3, 6, 0.5, 1]
        ).reshaped(4, 2)

        let input: [String: MLXArray] = [
            "rvq_first.vq.layers.0._codebook.cluster_usage": clusterUsage,
            "rvq_first.vq.layers.0._codebook.embedding_sum": embeddingSum,
        ]

        let sanitized = SplitResidualVectorQuantizer.sanitize(weights: input)

        let embedKey = "rvqFirst.vq.layers.0.codebook.embed.weight"
        #expect(sanitized.keys.contains(embedKey))

        let embedding = sanitized[embedKey]!
        MLX.eval(embedding)
        #expect(embedding.shape == [4, 2])

        let flat = embedding.reshaped(-1).asArray(Float.self)
        // centroid 0: [2/2, 4/2] = [1, 2]
        #expect(flat[0] == 1.0)
        #expect(flat[1] == 2.0)
        // centroid 1: [8/4, 12/4] = [2, 3]
        #expect(flat[2] == 2.0)
        #expect(flat[3] == 3.0)
        // centroid 2: [3/1, 6/1] = [3, 6]
        #expect(flat[4] == 3.0)
        #expect(flat[5] == 6.0)
        // centroid 3: [0.5/0.5, 1/0.5] = [1, 2]
        #expect(flat[6] == 1.0)
        #expect(flat[7] == 2.0)
    }

    @Test("Sanitize transposes Conv1d weights from PyTorch to MLX layout")
    func sanitizeTransposesConvWeights() {
        // PyTorch Conv1d weight: [out=4, in=8, kernel=1]
        let pytorchWeight = MLXArray.ones([4, 8, 1])
        let input: [String: MLXArray] = [
            "rvq_first.input_proj.weight": pytorchWeight,
        ]

        let sanitized = SplitResidualVectorQuantizer.sanitize(weights: input)
        let weight = sanitized["rvqFirst.inputProj.weight"]!
        MLX.eval(weight)

        // After transpose: [out=4, kernel=1, in=8]
        #expect(weight.shape == [4, 1, 8])
    }

    @Test("Sanitize passes through already-MLX-format Conv1d weights")
    func sanitizePassesThroughMLXWeights() {
        // MLX Conv1d weight: [out=4, kernel=1, in=128] — dim(2)>64, already MLX format
        let mlxWeight = MLXArray.ones([4, 1, 128])
        let input: [String: MLXArray] = [
            "rvq_first.output_proj.weight": mlxWeight,
        ]

        let sanitized = SplitResidualVectorQuantizer.sanitize(weights: input)
        let weight = sanitized["rvqFirst.outputProj.weight"]!
        MLX.eval(weight)

        // Should remain [4, 1, 128] (no transpose)
        #expect(weight.shape == [4, 1, 128])
    }
}
