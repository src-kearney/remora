#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir { class Pass; }

namespace remora::passes {

/// Consume cost attributes set by moe-expert-cost-analysis and emit concrete
/// per-expert compilation decisions as function-level attributes.
///
/// Reads per @expert_slot_N:
///   moe.cost_class           — required; pass fails if absent
///   moe.arithmetic_intensity — informational
///   moe.intermediate_dim     — used for BLOCK_N selection
///
/// Writes per @expert_slot_N:
///   moe.tile_class             : string — mirrors cost_class in V1
///   moe.BLOCK_M                : i64    — tile height (128/64/32/16)
///   moe.BLOCK_N                : i64    — tile width  (128/64/32)
///   moe.specialization_policy  : string — "shape_static_v1"
///
/// Prerequisite: moe-expert-cost-analysis must have run first.
std::unique_ptr<mlir::Pass> createExpertSpecializationPass();

/// Register ExpertSpecializationPass in the global MLIR pass registry so it
/// is usable with --pass-pipeline='...,moe-expert-specialization'.
void registerExpertSpecializationPass();

} // namespace remora::passes
