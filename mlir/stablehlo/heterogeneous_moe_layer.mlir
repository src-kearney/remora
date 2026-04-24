// heterogeneous_moe_layer.mlir
//
// StableHLO representation of a heterogeneous MoE layer where experts have
// different intermediate (FFN) dimensions.
//
// Expert layout (Remora paper §3 motivating example):
//
//   E0, E1  —  hidden=4096, intermediate=14336  (large)
//   E2, E3  —  hidden=4096, intermediate= 8192  (medium)
//   E4, E5  —  hidden=4096, intermediate= 4096  (small)
//   E6, E7  —  hidden=4096, intermediate= 2048  (tiny)
//
//
// WHY A SINGLE BATCHED OP IS IMPOSSIBLE
// ──────────────────────────────────────
// The standard uniform MoE representation (mixtral_moe_layer.mlir) expresses
// all expert FFNs as one batched dot_general:
//
//   dot_general %dispatched, %w_gate,
//       batching_dims = [0] x [0], contracting_dims = [2] x [1]
//   : (tensor<8x512x4096xf32>, tensor<8x4096x14336xf32>) -> tensor<8x512x14336xf32>
//
// This requires %w_gate to have a uniform trailing shape [4096, F] for all
// 8 batch elements.  When F differs per expert — 14336 for E0/E1, 8192 for
// E2/E3, 4096 for E4/E5, 2048 for E6/E7 — no static tensor type can express
// this.  The type system would require:
//
//   tensor<8x4096x[14336,14336,8192,8192,4096,4096,2048,2048]xf32>
//
// which is not a valid StableHLO (or any XLA-family) tensor type.
//
//
// THE FORCED DECOMPOSITION — ONE OP PER EXPERT
// ────────────────────────────────────────────
// The finest decomposition StableHLO can express is one batched dot_general
// per expert, each with batch_size=1.  Weight tensors are shaped [1, 4096, F_e]
// for gate/up and [1, F_e, 4096] for down, where F_e is expert e's intermediate
// dim.  This exposes each expert's true intermediate dimension as a static type,
// enabling independent per-expert kernel specialization.
//
//
// WHAT THE OUTLINING PASS PRODUCES WITH num_experts=1
// ────────────────────────────────────────────────────
// ExpertOutliningPass matches dots where lhs.shape[0] == numExperts.
// With numExperts=1 (via --pass-pipeline='moe-expert-outlining{num_experts=1}'),
// every per-expert group is matched, producing 8 outlined functions:
//
//   @expert_slot_0  — E0, tensor<512x14336xf32>  intermediate
//   @expert_slot_1  — E1, tensor<512x14336xf32>
//   @expert_slot_2  — E2, tensor<512x 8192xf32>
//   @expert_slot_3  — E3, tensor<512x 8192xf32>
//   @expert_slot_4  — E4, tensor<512x 4096xf32>
//   @expert_slot_5  — E5, tensor<512x 4096xf32>
//   @expert_slot_6  — E6, tensor<512x 2048xf32>
//   @expert_slot_7  — E7, tensor<512x 2048xf32>
//
// Each @expert_slot_N carries its true intermediate dimension as a static type.
// Downstream kernel-selection passes can specialize independently per expert.
//
//
// Routing: top-1 (argmax), one-hot dispatch.
// Batch:   512 tokens.

module @heterogeneous_moe_layer {

  func.func public @main(
      %tokens   : tensor<512x4096xf32>,   // [T, D]  input token representations
      %router_w : tensor<4096x8xf32>,     // [D, E]  router projection weights
      // ── Expert 0: intermediate = 14336 (large) ──────────────────────────
      %wg_0 : tensor<1x4096x14336xf32>,   // [1, D, F]  gate weight
      %wu_0 : tensor<1x4096x14336xf32>,   // [1, D, F]  up weight
      %wd_0 : tensor<1x14336x4096xf32>,   // [1, F, D]  down weight
      // ── Expert 1: intermediate = 14336 (large) ──────────────────────────
      %wg_1 : tensor<1x4096x14336xf32>,
      %wu_1 : tensor<1x4096x14336xf32>,
      %wd_1 : tensor<1x14336x4096xf32>,
      // ── Expert 2: intermediate = 8192 (medium) ──────────────────────────
      %wg_2 : tensor<1x4096x8192xf32>,
      %wu_2 : tensor<1x4096x8192xf32>,
      %wd_2 : tensor<1x8192x4096xf32>,
      // ── Expert 3: intermediate = 8192 (medium) ──────────────────────────
      %wg_3 : tensor<1x4096x8192xf32>,
      %wu_3 : tensor<1x4096x8192xf32>,
      %wd_3 : tensor<1x8192x4096xf32>,
      // ── Expert 4: intermediate = 4096 (small) ───────────────────────────
      %wg_4 : tensor<1x4096x4096xf32>,
      %wu_4 : tensor<1x4096x4096xf32>,
      %wd_4 : tensor<1x4096x4096xf32>,
      // ── Expert 5: intermediate = 4096 (small) ───────────────────────────
      %wg_5 : tensor<1x4096x4096xf32>,
      %wu_5 : tensor<1x4096x4096xf32>,
      %wd_5 : tensor<1x4096x4096xf32>,
      // ── Expert 6: intermediate = 2048 (tiny) ────────────────────────────
      %wg_6 : tensor<1x4096x2048xf32>,
      %wu_6 : tensor<1x4096x2048xf32>,
      %wd_6 : tensor<1x2048x4096xf32>,
      // ── Expert 7: intermediate = 2048 (tiny) ────────────────────────────
      %wg_7 : tensor<1x4096x2048xf32>,
      %wu_7 : tensor<1x4096x2048xf32>,
      %wd_7 : tensor<1x2048x4096xf32>
  ) -> tensor<512x4096xf32> {

    %cst_one = stablehlo.constant dense<1.000000e+00> : tensor<f32>

    // ── 1. ROUTING ────────────────────────────────────────────────────────
    // tokens[512, 4096] × router_w[4096, 8]  →  logits[512, 8]
    %logits = stablehlo.dot_general %tokens, %router_w,
        contracting_dims = [1] x [0]
        : (tensor<512x4096xf32>, tensor<4096x8xf32>) -> tensor<512x8xf32>

    // Top-1 argmax via topk(k=1): selects the highest-scoring expert per token.
    %topk:2 = stablehlo.custom_call @mhlo.topk(%logits)
        {mhlo.attributes = {k = 1 : i64}, mhlo.version = 1 : i64}
        : (tensor<512x8xf32>) -> (tensor<512x1xf32>, tensor<512x1xi32>)

    // Squeeze [512, 1] → [512]: one integer expert index per token.
    %expert_idx = stablehlo.reshape %topk#1
        : (tensor<512x1xi32>) -> tensor<512xi32>

    // One-hot encode selected index: [512] → [512, 8].
    %one_hot = call @_one_hot(%expert_idx)
        : (tensor<512xi32>) -> tensor<512x8xf32>

    // ── 2. DISPATCH ───────────────────────────────────────────────────────
    // Scatter each token to its expert slot via one-hot gather.
    //   tokens[512, D] × one_hot[512, 8]
    //   batching_dims = [0] x [0], contracting_dims = [] x []
    //   → [512, D, 8]
    %dispatch_tde = stablehlo.dot_general %tokens, %one_hot,
        batching_dims = [0] x [0], contracting_dims = [] x []
        : (tensor<512x4096xf32>, tensor<512x8xf32>) -> tensor<512x4096x8xf32>

    // Transpose [512, D, 8] → [8, 512, D]: expert-major layout for slicing.
    %dispatched = stablehlo.transpose %dispatch_tde, dims = [2, 0, 1]
        : (tensor<512x4096x8xf32>) -> tensor<8x512x4096xf32>

    // ── 3. SLICE PER EXPERT (batch=1 each) ───────────────────────────────
    // Each expert receives its own [1, 512, 4096] slice of the dispatched
    // tensor.  This is the finest granularity StableHLO can represent when
    // expert intermediate dims differ: one dot_general per expert, each with
    // lhs.shape[0]=1, allowing ExpertOutliningPass{num_experts=1} to match
    // all 8 groups and emit 8 distinct @expert_slot_N functions.
    %e0 = stablehlo.slice %dispatched [0:1, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E0
    %e1 = stablehlo.slice %dispatched [1:2, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E1
    %e2 = stablehlo.slice %dispatched [2:3, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E2
    %e3 = stablehlo.slice %dispatched [3:4, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E3
    %e4 = stablehlo.slice %dispatched [4:5, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E4
    %e5 = stablehlo.slice %dispatched [5:6, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E5
    %e6 = stablehlo.slice %dispatched [6:7, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E6
    %e7 = stablehlo.slice %dispatched [7:8, 0:512, 0:4096]
        : (tensor<8x512x4096xf32>) -> tensor<1x512x4096xf32>   // E7

    // ── 4. FFN EXPERT 0: F = 14336 ───────────────────────────────────────
    %gate_0 = stablehlo.dot_general %e0, %wg_0,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x14336xf32>) -> tensor<1x512x14336xf32>
    %up_0   = stablehlo.dot_general %e0, %wu_0,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x14336xf32>) -> tensor<1x512x14336xf32>
    // SwiGLU: silu(gate) * up,  silu(x) = x / (1 + exp(-x))
    %ng_0  = stablehlo.negate %gate_0 : tensor<1x512x14336xf32>
    %ex_0  = stablehlo.exponential %ng_0 : tensor<1x512x14336xf32>
    %bc_0  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x14336xf32>
    %dn_0  = stablehlo.add %bc_0, %ex_0 : tensor<1x512x14336xf32>
    %sg_0  = stablehlo.divide %bc_0, %dn_0 : tensor<1x512x14336xf32>
    %sl_0  = stablehlo.multiply %gate_0, %sg_0 : tensor<1x512x14336xf32>
    %act_0 = stablehlo.multiply %sl_0, %up_0 : tensor<1x512x14336xf32>
    %out_0 = stablehlo.dot_general %act_0, %wd_0,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x14336xf32>, tensor<1x14336x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 5. FFN EXPERT 1: F = 14336 ───────────────────────────────────────
    %gate_1 = stablehlo.dot_general %e1, %wg_1,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x14336xf32>) -> tensor<1x512x14336xf32>
    %up_1   = stablehlo.dot_general %e1, %wu_1,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x14336xf32>) -> tensor<1x512x14336xf32>
    %ng_1  = stablehlo.negate %gate_1 : tensor<1x512x14336xf32>
    %ex_1  = stablehlo.exponential %ng_1 : tensor<1x512x14336xf32>
    %bc_1  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x14336xf32>
    %dn_1  = stablehlo.add %bc_1, %ex_1 : tensor<1x512x14336xf32>
    %sg_1  = stablehlo.divide %bc_1, %dn_1 : tensor<1x512x14336xf32>
    %sl_1  = stablehlo.multiply %gate_1, %sg_1 : tensor<1x512x14336xf32>
    %act_1 = stablehlo.multiply %sl_1, %up_1 : tensor<1x512x14336xf32>
    %out_1 = stablehlo.dot_general %act_1, %wd_1,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x14336xf32>, tensor<1x14336x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 6. FFN EXPERT 2: F = 8192 ────────────────────────────────────────
    %gate_2 = stablehlo.dot_general %e2, %wg_2,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x8192xf32>) -> tensor<1x512x8192xf32>
    %up_2   = stablehlo.dot_general %e2, %wu_2,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x8192xf32>) -> tensor<1x512x8192xf32>
    %ng_2  = stablehlo.negate %gate_2 : tensor<1x512x8192xf32>
    %ex_2  = stablehlo.exponential %ng_2 : tensor<1x512x8192xf32>
    %bc_2  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x8192xf32>
    %dn_2  = stablehlo.add %bc_2, %ex_2 : tensor<1x512x8192xf32>
    %sg_2  = stablehlo.divide %bc_2, %dn_2 : tensor<1x512x8192xf32>
    %sl_2  = stablehlo.multiply %gate_2, %sg_2 : tensor<1x512x8192xf32>
    %act_2 = stablehlo.multiply %sl_2, %up_2 : tensor<1x512x8192xf32>
    %out_2 = stablehlo.dot_general %act_2, %wd_2,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x8192xf32>, tensor<1x8192x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 7. FFN EXPERT 3: F = 8192 ────────────────────────────────────────
    %gate_3 = stablehlo.dot_general %e3, %wg_3,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x8192xf32>) -> tensor<1x512x8192xf32>
    %up_3   = stablehlo.dot_general %e3, %wu_3,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x8192xf32>) -> tensor<1x512x8192xf32>
    %ng_3  = stablehlo.negate %gate_3 : tensor<1x512x8192xf32>
    %ex_3  = stablehlo.exponential %ng_3 : tensor<1x512x8192xf32>
    %bc_3  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x8192xf32>
    %dn_3  = stablehlo.add %bc_3, %ex_3 : tensor<1x512x8192xf32>
    %sg_3  = stablehlo.divide %bc_3, %dn_3 : tensor<1x512x8192xf32>
    %sl_3  = stablehlo.multiply %gate_3, %sg_3 : tensor<1x512x8192xf32>
    %act_3 = stablehlo.multiply %sl_3, %up_3 : tensor<1x512x8192xf32>
    %out_3 = stablehlo.dot_general %act_3, %wd_3,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x8192xf32>, tensor<1x8192x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 8. FFN EXPERT 4: F = 4096 ────────────────────────────────────────
    %gate_4 = stablehlo.dot_general %e4, %wg_4,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>
    %up_4   = stablehlo.dot_general %e4, %wu_4,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>
    %ng_4  = stablehlo.negate %gate_4 : tensor<1x512x4096xf32>
    %ex_4  = stablehlo.exponential %ng_4 : tensor<1x512x4096xf32>
    %bc_4  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x4096xf32>
    %dn_4  = stablehlo.add %bc_4, %ex_4 : tensor<1x512x4096xf32>
    %sg_4  = stablehlo.divide %bc_4, %dn_4 : tensor<1x512x4096xf32>
    %sl_4  = stablehlo.multiply %gate_4, %sg_4 : tensor<1x512x4096xf32>
    %act_4 = stablehlo.multiply %sl_4, %up_4 : tensor<1x512x4096xf32>
    %out_4 = stablehlo.dot_general %act_4, %wd_4,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 9. FFN EXPERT 5: F = 4096 ────────────────────────────────────────
    %gate_5 = stablehlo.dot_general %e5, %wg_5,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>
    %up_5   = stablehlo.dot_general %e5, %wu_5,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>
    %ng_5  = stablehlo.negate %gate_5 : tensor<1x512x4096xf32>
    %ex_5  = stablehlo.exponential %ng_5 : tensor<1x512x4096xf32>
    %bc_5  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x4096xf32>
    %dn_5  = stablehlo.add %bc_5, %ex_5 : tensor<1x512x4096xf32>
    %sg_5  = stablehlo.divide %bc_5, %dn_5 : tensor<1x512x4096xf32>
    %sl_5  = stablehlo.multiply %gate_5, %sg_5 : tensor<1x512x4096xf32>
    %act_5 = stablehlo.multiply %sl_5, %up_5 : tensor<1x512x4096xf32>
    %out_5 = stablehlo.dot_general %act_5, %wd_5,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 10. FFN EXPERT 6: F = 2048 ───────────────────────────────────────
    %gate_6 = stablehlo.dot_general %e6, %wg_6,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x2048xf32>) -> tensor<1x512x2048xf32>
    %up_6   = stablehlo.dot_general %e6, %wu_6,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x2048xf32>) -> tensor<1x512x2048xf32>
    %ng_6  = stablehlo.negate %gate_6 : tensor<1x512x2048xf32>
    %ex_6  = stablehlo.exponential %ng_6 : tensor<1x512x2048xf32>
    %bc_6  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x2048xf32>
    %dn_6  = stablehlo.add %bc_6, %ex_6 : tensor<1x512x2048xf32>
    %sg_6  = stablehlo.divide %bc_6, %dn_6 : tensor<1x512x2048xf32>
    %sl_6  = stablehlo.multiply %gate_6, %sg_6 : tensor<1x512x2048xf32>
    %act_6 = stablehlo.multiply %sl_6, %up_6 : tensor<1x512x2048xf32>
    %out_6 = stablehlo.dot_general %act_6, %wd_6,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x2048xf32>, tensor<1x2048x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 11. FFN EXPERT 7: F = 2048 ───────────────────────────────────────
    %gate_7 = stablehlo.dot_general %e7, %wg_7,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x2048xf32>) -> tensor<1x512x2048xf32>
    %up_7   = stablehlo.dot_general %e7, %wu_7,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x4096xf32>, tensor<1x4096x2048xf32>) -> tensor<1x512x2048xf32>
    %ng_7  = stablehlo.negate %gate_7 : tensor<1x512x2048xf32>
    %ex_7  = stablehlo.exponential %ng_7 : tensor<1x512x2048xf32>
    %bc_7  = stablehlo.broadcast_in_dim %cst_one, dims = []
             : (tensor<f32>) -> tensor<1x512x2048xf32>
    %dn_7  = stablehlo.add %bc_7, %ex_7 : tensor<1x512x2048xf32>
    %sg_7  = stablehlo.divide %bc_7, %dn_7 : tensor<1x512x2048xf32>
    %sl_7  = stablehlo.multiply %gate_7, %sg_7 : tensor<1x512x2048xf32>
    %act_7 = stablehlo.multiply %sl_7, %up_7 : tensor<1x512x2048xf32>
    %out_7 = stablehlo.dot_general %act_7, %wd_7,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x512x2048xf32>, tensor<1x2048x4096xf32>) -> tensor<1x512x4096xf32>

    // ── 12. GATHER: reassemble expert outputs and weighted sum ────────────
    // Concatenate per-expert results along the expert dim → [8, 512, 4096].
    %expert_out = stablehlo.concatenate %out_0, %out_1, %out_2, %out_3,
                                         %out_4, %out_5, %out_6, %out_7, dim = 0
        : (tensor<1x512x4096xf32>, tensor<1x512x4096xf32>,
           tensor<1x512x4096xf32>, tensor<1x512x4096xf32>,
           tensor<1x512x4096xf32>, tensor<1x512x4096xf32>,
           tensor<1x512x4096xf32>, tensor<1x512x4096xf32>) -> tensor<8x512x4096xf32>

    // Select each token's result using its one-hot dispatch mask.
    //   one_hot[512, 8] × expert_out[8, 512, 4096]
    //   batching_dims = [0] x [1], contracting_dims = [1] x [0]
    //   → [512, 4096]
    %result = stablehlo.dot_general %one_hot, %expert_out,
        batching_dims = [0] x [1], contracting_dims = [1] x [0]
        : (tensor<512x8xf32>, tensor<8x512x4096xf32>) -> tensor<512x4096xf32>

    return %result : tensor<512x4096xf32>
  }

  // ── One-hot encoding ───────────────────────────────────────────────────────
  // Converts an integer expert index per token into a float mask over E=8 experts.
  // Copied verbatim from the JAX-exported mixtral_moe_layer.mlir.
  func.func private @_one_hot(%arg0: tensor<512xi32>) -> tensor<512x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0]
         : (tensor<512xi32>) -> tensor<512x1xi32>
    %1 = stablehlo.iota dim = 1 : tensor<1x8xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1]
         : (tensor<512x1xi32>) -> tensor<512x8xi32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1]
         : (tensor<1x8xi32>) -> tensor<512x8xi32>
    %4 = stablehlo.compare EQ, %2, %3, SIGNED
         : (tensor<512x8xi32>, tensor<512x8xi32>) -> tensor<512x8xi1>
    %5 = stablehlo.convert %4 : (tensor<512x8xi1>) -> tensor<512x8xf32>
    return %5 : tensor<512x8xf32>
  }

}
