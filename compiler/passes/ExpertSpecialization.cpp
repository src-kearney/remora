// ExpertSpecialization.cpp
//
// Third pass in the Remora pipeline.  Consumes the cost attributes attached
// by moe-expert-cost-analysis and emits concrete compilation decisions as
// function-level attributes on each @expert_slot_N.
//
// This is the decision layer: cost analysis characterises what each expert
// is; specialization decides what it should be compiled as.
//
// In V1 the policy is "shape_static_v1": tile dimensions are selected
// directly from the static intermediate dim F, using the same class
// boundaries established by the cost analysis pass.
//
// Prerequisite: moe-expert-cost-analysis must have run.  If moe.cost_class
// is absent from any @expert_slot_N, the pass emits an MLIR error and
// signals failure rather than silently emitting wrong decisions.
//
// Attributes written per @expert_slot_N:
//   moe.tile_class            = cost_class value   : string
//   moe.BLOCK_M               = 128|64|32|16        : i64
//   moe.BLOCK_N               = 128|64|32            : i64
//   moe.specialization_policy = "shape_static_v1"   : string
//
// BLOCK_M is driven by cost_class (same mapping as the outlining pass uses
// so existing dispatch infrastructure stays consistent).
// BLOCK_N is driven independently by intermediate_dim:
//   F >= 8192 → 128,  F >= 4096 → 64,  F < 4096 → 32

#include "ExpertSpecialization.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace remora::passes {
namespace {

// ---------------------------------------------------------------------------
// Decision tables
// ---------------------------------------------------------------------------

static int64_t blockMForCostClass(StringRef costClass) {
  if (costClass == "large")  return 128;
  if (costClass == "medium") return  64;
  if (costClass == "small")  return  32;
  return 16; // "tiny"
}

/// BLOCK_N is selected from intermediate_dim independently of BLOCK_M.
/// Larger F benefits from wider tiles on the N dimension.
static int64_t blockNForIntermediateDim(int64_t F) {
  if (F >= 8192) return 128;
  if (F >= 4096) return  64;
  return 32;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ExpertSpecializationPass
    : public PassWrapper<ExpertSpecializationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertSpecializationPass)

  StringRef getArgument() const override { return "moe-expert-specialization"; }
  StringRef getDescription() const override {
    return "Consume per-expert cost attributes and emit concrete tile "
           "configuration decisions (BLOCK_M, BLOCK_N, tile_class)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    auto i64 = IntegerType::get(ctx, 64);
    bool anyFailure = false;

    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr("moe.slot_id"))
        return;

      // moe.cost_class is required — its absence means cost analysis did not run.
      auto costClassAttr = func->getAttrOfType<StringAttr>("moe.cost_class");
      if (!costClassAttr) {
        func->emitError(
            "moe-expert-specialization: @" + func.getName().str() +
            " is missing moe.cost_class — run moe-expert-cost-analysis first");
        anyFailure = true;
        return;
      }

      StringRef costClass = costClassAttr.getValue();

      // moe.intermediate_dim is needed for BLOCK_N; fall back to 0 (→ 32)
      // if somehow absent, but warn.
      int64_t intermediateDim = 0;
      if (auto attr = func->getAttrOfType<IntegerAttr>("moe.intermediate_dim"))
        intermediateDim = attr.getInt();
      else
        func->emitWarning(
            "moe-expert-specialization: @" + func.getName().str() +
            " is missing moe.intermediate_dim — BLOCK_N will default to 32");

      int64_t blockM = blockMForCostClass(costClass);
      int64_t blockN = blockNForIntermediateDim(intermediateDim);

      func->setAttr("moe.tile_class",
                    StringAttr::get(ctx, costClass));
      func->setAttr("moe.BLOCK_M",
                    IntegerAttr::get(i64, blockM));
      func->setAttr("moe.BLOCK_N",
                    IntegerAttr::get(i64, blockN));
      func->setAttr("moe.specialization_policy",
                    StringAttr::get(ctx, "shape_static_v1"));
    });

    if (anyFailure)
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> createExpertSpecializationPass() {
  return std::make_unique<ExpertSpecializationPass>();
}

void registerExpertSpecializationPass() {
  PassRegistration<ExpertSpecializationPass>();
}

} // namespace remora::passes
