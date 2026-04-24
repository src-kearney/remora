// ExpertCostAnalysis.cpp
//
// Second pass in the Remora pipeline.  Walks all @expert_slot_N functions
// produced by moe-expert-outlining, reads T / D / F from their argument
// tensor types, computes hardware cost metrics, and attaches them as
// function-level attributes.
//
// Argument layout (set by ExpertOutliningPass):
//   arg0: tensor<T x D x f32>  — dispatched tokens
//   arg1: tensor<D x F x f32>  — gate weight
//   arg2: tensor<D x F x f32>  — up weight
//   arg3: tensor<F x D x f32>  — down weight
//
// Attributes written:
//   moe.hidden_dim           = D            : i64
//   moe.intermediate_dim     = F            : i64  (verified vs existing attr)
//   moe.tokens_per_slot      = T            : i64  (upgrades i32 set by outlining)
//   moe.flops_estimate       = 2*T*D*F*3   : i64
//   moe.bytes_estimate       = (2*T*D+3*D*F)*4 : i64
//   moe.arithmetic_intensity = flops/bytes  : f64
//   moe.cost_class           = "large"|"medium"|"small"|"tiny" : string
//
// Cost class thresholds (strict >):
//   F > 8192  → large   (e.g. Mixtral full dim 14336)
//   F > 4096  → medium  (e.g. 8192)
//   F > 2048  → small   (e.g. 4096)
//   F <= 2048 → tiny    (e.g. 2048)

#include "ExpertCostAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace remora::passes {
namespace {

// ---------------------------------------------------------------------------
// Cost model helpers
// ---------------------------------------------------------------------------

/// Three matmuls in one SwiGLU FFN forward pass:
///   gate [T,D]×[D,F] → [T,F]  : 2·T·D·F FLOPs
///   up   [T,D]×[D,F] → [T,F]  : 2·T·D·F FLOPs
///   down [T,F]×[F,D] → [T,D]  : 2·T·F·D FLOPs
/// Total = 6·T·D·F, written as 2·T·D·F·3 to make the three-matmul structure
/// explicit.
static int64_t computeFlops(int64_t T, int64_t D, int64_t F) {
  return 2 * T * D * F * 3;
}

/// HBM traffic for one expert forward pass (all f32 = 4 bytes per element):
///   Reads:  tokens [T,D], gate weight [D,F], up weight [D,F], down weight [F,D]
///   Writes: output [T,D]
/// The SwiGLU intermediate [T,F] is assumed to stay on-chip.
static int64_t computeBytes(int64_t T, int64_t D, int64_t F) {
  int64_t elements = (T * D)      // tokens (read)
                   + (D * F)      // gate weight (read)
                   + (D * F)      // up weight (read)
                   + (F * D)      // down weight (read)  [= D*F]
                   + (T * D);     // output (write)
  return elements * 4;            // float32 = 4 bytes
}

/// Cost class from intermediate dim F.  Uses strict > so that boundary values
/// fall into the lower class:
///   F=14336 → large, F=8192 → medium, F=4096 → small, F=2048 → tiny.
static StringRef costClassForF(int64_t F) {
  if (F > 8192) return "large";
  if (F > 4096) return "medium";
  if (F > 2048) return "small";
  return "tiny";
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ExpertCostAnalysisPass
    : public PassWrapper<ExpertCostAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertCostAnalysisPass)

  StringRef getArgument() const override { return "moe-expert-cost-analysis"; }
  StringRef getDescription() const override {
    return "Compute per-expert hardware cost metrics from IR shapes and attach "
           "them as function attributes";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    auto i64 = IntegerType::get(ctx, 64);
    auto f64 = Float64Type::get(ctx);

    module.walk([&](func::FuncOp func) {
      // Only process functions produced by the outlining pass.
      if (!func->hasAttr("moe.slot_id"))
        return;

      // Require at least four arguments; the pass is not responsible for
      // constructing the function, just annotating it.
      if (func.getNumArguments() < 4) {
        func->emitWarning(
            "moe-expert-cost-analysis: @" + func.getName().str() +
            " has fewer than 4 arguments — skipping cost analysis");
        return;
      }

      // Read shape dimensions from argument types.
      auto tokenType      = mlir::dyn_cast<RankedTensorType>(func.getArgumentTypes()[0]);
      auto gateWeightType = mlir::dyn_cast<RankedTensorType>(func.getArgumentTypes()[1]);

      if (!tokenType || !gateWeightType ||
          tokenType.getRank() < 2 || gateWeightType.getRank() < 2) {
        func->emitWarning(
            "moe-expert-cost-analysis: @" + func.getName().str() +
            " argument types are not rank-2 tensors — skipping cost analysis");
        return;
      }

      int64_t numTokens       = tokenType.getDimSize(0);      // T
      int64_t hiddenDim       = tokenType.getDimSize(1);      // D
      int64_t intermediateDim = gateWeightType.getDimSize(1); // F

      // Verify moe.intermediate_dim set by outlining pass, if present.
      if (auto existingAttr =
              func->getAttrOfType<IntegerAttr>("moe.intermediate_dim")) {
        int64_t existingF = existingAttr.getInt();
        if (existingF != intermediateDim) {
          func->emitWarning(
              "moe-expert-cost-analysis: @" + func.getName().str() +
              " moe.intermediate_dim=" + std::to_string(existingF) +
              " does not match arg-type F=" +
              std::to_string(intermediateDim) +
              " — using arg-type value");
        }
      }

      // Compute cost metrics.
      int64_t flops = computeFlops(numTokens, hiddenDim, intermediateDim);
      int64_t bytes = computeBytes(numTokens, hiddenDim, intermediateDim);
      double  arithmeticIntensity =
          bytes > 0 ? static_cast<double>(flops) / static_cast<double>(bytes)
                    : 0.0;
      StringRef costClass = costClassForF(intermediateDim);

      // Attach attributes.  moe.intermediate_dim and moe.tokens_per_slot
      // already exist from outlining (possibly as i32); overwrite with i64
      // for type consistency across the pipeline.
      func->setAttr("moe.hidden_dim",
                    IntegerAttr::get(i64, hiddenDim));
      func->setAttr("moe.intermediate_dim",
                    IntegerAttr::get(i64, intermediateDim));
      func->setAttr("moe.tokens_per_slot",
                    IntegerAttr::get(i64, numTokens));
      func->setAttr("moe.flops_estimate",
                    IntegerAttr::get(i64, flops));
      func->setAttr("moe.bytes_estimate",
                    IntegerAttr::get(i64, bytes));
      func->setAttr("moe.arithmetic_intensity",
                    FloatAttr::get(f64, arithmeticIntensity));
      func->setAttr("moe.cost_class",
                    StringAttr::get(ctx, costClass));
    });
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> createExpertCostAnalysisPass() {
  return std::make_unique<ExpertCostAnalysisPass>();
}

void registerExpertCostAnalysisPass() {
  PassRegistration<ExpertCostAnalysisPass>();
}

} // namespace remora::passes
