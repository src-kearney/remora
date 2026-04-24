// test_non_swiglu_chain.mlir
//
// Regression fixture for ExpertOutliningPass error handling.
//
// The SwiGLU activation chain for expert slot 0 injects %external — a block
// argument — as an operand instead of %up.  This produces an operand in the
// SwiGLU walk that is neither in the gate/up SSA mapping nor a ConstantOp.
//
// Expected behaviour:
//   moe-expert-outlining exits with a non-zero status and emits:
//     "moe-expert-outlining: unexpected non-constant operand in SwiGLU subgraph"
//
// Run manually:
//   remora mlir/stablehlo/test_non_swiglu_chain.mlir \
//       --pass-pipeline=moe-expert-outlining --no-execute
//   # → exit 1, diagnostic on the bad multiply op

module @test_non_swiglu_chain {

  func.func public @main(
      %dispatched : tensor<8x512x4096xf32>,  // [E, T, D]
      %w_gate     : tensor<8x4096x14336xf32>,
      %w_up       : tensor<8x4096x14336xf32>,
      %w_down     : tensor<8x14336x4096xf32>,
      %external   : tensor<8x512x14336xf32>  // injected block arg — not SwiGLU
  ) -> tensor<8x512x4096xf32> {

    %cst_one = stablehlo.constant dense<1.000000e+00> : tensor<f32>

    // Standard gate and up projections (lhs.shape[0]=8 → matched by pass).
    %gate = stablehlo.dot_general %dispatched, %w_gate,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>

    %up = stablehlo.dot_general %dispatched, %w_up,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>

    // Partial SwiGLU: silu(%gate) computed correctly ...
    %ng = stablehlo.negate %gate : tensor<8x512x14336xf32>
    %ex = stablehlo.exponential %ng : tensor<8x512x14336xf32>
    %bc = stablehlo.broadcast_in_dim %cst_one, dims = []
          : (tensor<f32>) -> tensor<8x512x14336xf32>
    %dn = stablehlo.add %bc, %ex : tensor<8x512x14336xf32>
    %sg = stablehlo.divide %bc, %dn : tensor<8x512x14336xf32>
    %sl = stablehlo.multiply %gate, %sg : tensor<8x512x14336xf32>

    // ... but the final multiply uses %external (block arg) instead of %up.
    // cloneStripped will find %external unmapped and non-constant → failure.
    %act = stablehlo.multiply %sl, %external : tensor<8x512x14336xf32>

    %out = stablehlo.dot_general %act, %w_down,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<8x512x14336xf32>, tensor<8x14336x4096xf32>) -> tensor<8x512x4096xf32>

    return %out : tensor<8x512x4096xf32>
  }

}
