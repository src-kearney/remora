#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir { class Pass; }

namespace remora::passes {

/// Walk all @expert_slot_N functions produced by moe-expert-outlining and
/// attach hardware cost attributes derived from their argument tensor shapes.
///
/// Attaches (all values computed from IR shapes, never hardcoded):
///   moe.hidden_dim           : i64  — hidden dimension D
///   moe.intermediate_dim     : i64  — intermediate dimension F (verified vs outlining attr)
///   moe.tokens_per_slot      : i64  — token count T (overrides i32 set by outlining)
///   moe.flops_estimate       : i64  — 2 * T * D * F * 3 (three matmuls, multiply-add pairs)
///   moe.bytes_estimate       : i64  — (2*T*D + 3*D*F) * 4  (f32, HBM reads + output write)
///   moe.arithmetic_intensity : f64  — flops / bytes
///   moe.cost_class           : str  — "large" | "medium" | "small" | "tiny"
///
/// Prerequisite: moe-expert-outlining must have run first.
std::unique_ptr<mlir::Pass> createExpertCostAnalysisPass();

/// Register ExpertCostAnalysisPass in the global MLIR pass registry so it is
/// usable with --pass-pipeline='...,moe-expert-cost-analysis'.
void registerExpertCostAnalysisPass();

} // namespace remora::passes
