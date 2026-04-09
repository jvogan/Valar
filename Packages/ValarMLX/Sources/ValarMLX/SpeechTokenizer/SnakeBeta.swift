import Foundation
@preconcurrency import MLX
import MLXNN

/// SnakeBeta activation function with learnable per-channel parameters.
///
/// Computes `x + (1 / beta) * sin(alpha * x)^2` where alpha and beta are
/// stored in log-space and exponentiated before use. This ensures alpha and
/// beta are always positive at runtime.
///
/// Parameters are initialized to zero in log-space (i.e. `exp(0) = 1.0`).
/// When loading from safetensors, the stored values replace the defaults
/// and are exponentiated on each forward pass.
final class SnakeBeta: Module {
    let channels: Int
    var alpha: MLXArray
    var beta: MLXArray

    init(channels: Int) {
        self.channels = channels
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let a = exp(alpha).reshaped(1, -1, 1)
        let b = exp(beta).reshaped(1, -1, 1)
        let s = sin(x * a)
        return x + (1.0 / (b + 1e-9)) * s * s
    }
}
