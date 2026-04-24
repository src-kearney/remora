// ExpertOutlining.cpp
//
// Transforms batched MoE expert dot_generals into outlined per-expert
// functions.
//
// Input (per top-k slot, observed in mixtral_moe_layer.mlir):
//   %gate = dot_general %dispatched, %w_gate, batching=[0]x[0], contracting=[2]x[1]
//   %up   = dot_general %dispatched, %w_up,   batching=[0]x[0], contracting=[2]x[1]
//   <SwiGLU: negate / exp / broadcast / add / divide / multiply / multiply>
//   %down = dot_general %silu_out,   %w_down, batching=[0]x[0], contracting=[2]x[1]
//
// Output — @expert_slot_N (called 8× from @main, once per expert):
//   func.func @expert_slot_N(
//       %dispatched_e: tensor<TxDxf32>,
//       %w_gate_e:     tensor<DxFxf32>,
//       %w_up_e:       tensor<DxFxf32>,
//       %w_down_e:     tensor<FxDxf32>) → tensor<TxDxf32>
//   {
//     gate = dot_general(%dispatched_e, %w_gate_e, contracting=[1]x[0])
//     up   = dot_general(%dispatched_e, %w_up_e,   contracting=[1]x[0])
//     <SwiGLU at per-expert shapes [T, F]>
//     down = dot_general(%silu_out, %w_down_e,     contracting=[1]x[0])
//     return down
//   }
//
// @main replacement per slot:
//   for e in 0..7:
//     d_e   = slice(dispatched, [e,0,0]->[e+1,T,D]) reshaped → [T,D]
//     wg_e  = slice(w_gate,     [e,0,0]->[e+1,D,F]) reshaped → [D,F]
//     (similar for w_up, w_down)
//     out_e = call @expert_slot_N(d_e, wg_e, wu_e, wd_e)     → [T,D]
//     reshape → [1,T,D]
//   concat([...], dim=0) → [E,T,D]   replaces old %down result

#include "ExpertOutlining.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace remora::passes {
namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// True iff `op` is a DotGeneralOp with lhs_batching_dimensions containing 0
/// and lhs.shape[0] == numExperts.
static bool isExpertBatchDot(stablehlo::DotGeneralOp op, int64_t numExperts) {
  auto lhsTy = mlir::dyn_cast<RankedTensorType>(op.getLhs().getType());
  if (!lhsTy || lhsTy.getRank() < 1)
    return false;
  auto batch = op.getDotDimensionNumbers().getLhsBatchingDimensions();
  return llvm::is_contained(batch, (int64_t)0) &&
         lhsTy.getDimSize(0) == numExperts;
}

/// Strip the leading dimension of a ranked tensor type.
/// [8, 512, 4096] → [512, 4096]
static RankedTensorType stripLeadingDim(RankedTensorType ty) {
  return RankedTensorType::get(ty.getShape().drop_front(),
                               ty.getElementType());
}

/// Walk backwards from downOp.getLhs() through elementwise ops, stopping at
/// gateResult, upResult, constants, and block arguments.  Returns the
/// intermediate ops in topological (forward) order.
static SmallVector<Operation *>
collectSwiGLU(stablehlo::DotGeneralOp downOp, Value gateResult,
              Value upResult) {
  DenseSet<Value> boundary = {gateResult, upResult};
  SmallVector<Operation *> ops;
  DenseSet<Operation *> visited;
  SmallVector<Value> worklist = {downOp.getLhs()};

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (boundary.count(val))
      continue;
    Operation *def = val.getDefiningOp();
    if (!def || isa<stablehlo::ConstantOp>(def))
      continue;
    if (!visited.insert(def).second)
      continue;
    ops.push_back(def);
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  llvm::sort(ops, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return ops;
}

/// Clone one op into the outlined function body with the expert batch dim (0)
/// stripped from its result types.  Operands are remapped via `mapping`.
/// Constant operands not yet in the mapping are cloned fresh.
/// Returns failure() if an operand is neither in the mapping nor a constant
/// (indicating the activation pattern is not plain SwiGLU).
static LogicalResult cloneStripped(OpBuilder &b, Operation *op,
                                   IRMapping &mapping, int64_t numExperts,
                                   Location loc) {
  SmallVector<Value> operands;
  for (Value v : op->getOperands()) {
    if (mapping.contains(v)) {
      operands.push_back(mapping.lookup(v));
      continue;
    }
    // Operand not in mapping — must be a constant defined outside the chain.
    Operation *def = v.getDefiningOp();
    if (!def || !isa<stablehlo::ConstantOp>(def)) {
      op->emitError(
          "moe-expert-outlining: unexpected non-constant operand in SwiGLU "
          "subgraph — activation pattern may not be SwiGLU");
      return failure();
    }
    Operation *cloned = b.clone(*def);
    mapping.map(v, cloned->getResult(0));
    operands.push_back(cloned->getResult(0));
  }

  SmallVector<Type> resultTypes;
  for (Type ty : op->getResultTypes()) {
    if (auto rtt = mlir::dyn_cast<RankedTensorType>(ty);
        rtt && !rtt.getShape().empty() && rtt.getDimSize(0) == numExperts)
      resultTypes.push_back(stripLeadingDim(rtt));
    else
      resultTypes.push_back(ty);
  }

  OperationState state(loc, op->getName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(op->getAttrDictionary().getValue());
  Operation *cloned = b.create(state);

  for (unsigned i = 0; i < op->getNumResults(); ++i)
    mapping.map(op->getResult(i), cloned->getResult(i));

  return success();
}

// ---------------------------------------------------------------------------
// Slot descriptor
// ---------------------------------------------------------------------------

struct ExpertSlot {
  int slotId;
  Value dispatched;                    // [E, T, D]
  stablehlo::DotGeneralOp gateOp;     // [E,T,D] × [E,D,F] → [E,T,F]
  stablehlo::DotGeneralOp upOp;       // [E,T,D] × [E,D,F] → [E,T,F]
  SmallVector<Operation *> swigluOps; // elementwise chain, topological order
  stablehlo::DotGeneralOp downOp;     // [E,T,F] × [E,F,D] → [E,T,D]
  Value wGate;                         // [E, D, F] — rhs of gateOp
  Value wUp;                           // [E, D, F] — rhs of upOp
  Value wDown;                         // [E, F, D] — rhs of downOp
};

// ---------------------------------------------------------------------------
// Build @expert_slot_N
// ---------------------------------------------------------------------------

static FailureOr<func::FuncOp> buildExpertSlotFunc(ModuleOp module,
                                                    ExpertSlot &slot,
                                                    int64_t numExperts) {
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  auto f32 = Float32Type::get(ctx);

  // Batched types from the slot (include leading expert dim).
  auto batchedDispatchType  = mlir::cast<RankedTensorType>(slot.dispatched.getType());
  auto batchedGateWeightType = mlir::cast<RankedTensorType>(slot.wGate.getType());
  auto batchedUpWeightType   = mlir::cast<RankedTensorType>(slot.wUp.getType());
  auto batchedDownWeightType = mlir::cast<RankedTensorType>(slot.wDown.getType());
  auto batchedResultType     = mlir::cast<RankedTensorType>(slot.downOp.getResult().getType());

  // Per-expert types (strip leading expert dim).
  auto expertDispatchType  = stripLeadingDim(batchedDispatchType);   // [T, D]
  auto expertGateWeightType = stripLeadingDim(batchedGateWeightType); // [D, F]
  auto expertUpWeightType   = stripLeadingDim(batchedUpWeightType);   // [D, F]
  auto expertDownWeightType = stripLeadingDim(batchedDownWeightType); // [F, D]
  auto expertResultType     = stripLeadingDim(batchedResultType);     // [T, D]

  auto funcTy = FunctionType::get(
      ctx,
      {expertDispatchType, expertGateWeightType, expertUpWeightType, expertDownWeightType},
      {expertResultType});
  std::string funcName = "expert_slot_" + std::to_string(slot.slotId);

  // Insert before everything else in the module (before @main).
  OpBuilder mb(module.getBody(), module.getBody()->begin());
  auto func = mb.create<func::FuncOp>(loc, funcName, funcTy);

  // Attach MoE attributes.
  auto i32 = IntegerType::get(ctx, 32);
  func->setAttr("moe.slot_id", IntegerAttr::get(i32, slot.slotId));
  func->setAttr("moe.num_experts", IntegerAttr::get(i32, numExperts));
  func->setAttr("moe.tokens_per_slot",
                IntegerAttr::get(i32, batchedDispatchType.getDimSize(1)));

  // Build the function body.
  Block *block = func.addEntryBlock();
  OpBuilder b(block, block->end());

  Value dispArg  = block->getArgument(0); // [T, D]
  Value wGateArg = block->getArgument(1); // [D, F]
  Value wUpArg   = block->getArgument(2); // [D, F]
  Value wDownArg = block->getArgument(3); // [F, D]

  int64_t T = expertDispatchType.getDimSize(0);   // 512
  int64_t F = expertGateWeightType.getDimSize(1); // 14336 (intermediate dim)
  int64_t D = expertDispatchType.getDimSize(1);   // 4096

  // Compute and attach shape-derived compilation attributes.
  auto i64 = IntegerType::get(ctx, 64);
  // Thresholds use strict > so that each boundary value falls in the lower
  // class: F=8192 → medium (64), F=4096 → small (32), F=2048 → tiny (16).
  StringRef tileClass = F > 8192 ? "large"
                      : F > 4096 ? "medium"
                      : F > 2048 ? "small"
                      :            "tiny";
  int64_t blockM = F > 8192 ? 128
                 : F > 4096 ?  64
                 : F > 2048 ?  32
                 :             16;
  func->setAttr("moe.intermediate_dim", IntegerAttr::get(i64, F));
  func->setAttr("moe.tile_class",       StringAttr::get(ctx, tileClass));
  func->setAttr("moe.BLOCK_M",          IntegerAttr::get(i64, blockM));

  // Non-batched dimension numbers: contracting [1] × [0].
  auto noBatch = stablehlo::DotDimensionNumbersAttr::get(
      ctx, /*lhsBatch=*/{}, /*rhsBatch=*/{},
      /*lhsContract=*/{1}, /*rhsContract=*/{0});
  auto emptyPrec = ArrayAttr::get(ctx, {});

  // gate = dot(dispatched, w_gate, contracting=[1]×[0]) → [T, F]
  auto gateTy  = RankedTensorType::get({T, F}, f32);
  auto noAlg   = stablehlo::DotAlgorithmAttr{};
  auto gateVal = b.create<stablehlo::DotGeneralOp>(
      loc, gateTy, dispArg, wGateArg, noBatch, emptyPrec, noAlg);

  // up = dot(dispatched, w_up, contracting=[1]×[0]) → [T, F]
  auto upVal = b.create<stablehlo::DotGeneralOp>(
      loc, gateTy, dispArg, wUpArg, noBatch, emptyPrec, noAlg);

  // Clone SwiGLU ops with expert dim stripped from types.
  IRMapping mapping;
  mapping.map(slot.gateOp.getResult(), gateVal.getResult());
  mapping.map(slot.upOp.getResult(), upVal.getResult());
  for (Operation *op : slot.swigluOps) {
    if (failed(cloneStripped(b, op, mapping, numExperts, loc))) {
      func.erase();
      return failure();
    }
  }

  // down = dot(silu_out, w_down, contracting=[1]×[0]) → [T, D]
  Value siluOut = mapping.lookup(slot.downOp.getLhs());
  auto downTy   = RankedTensorType::get({T, D}, f32);
  auto downVal  = b.create<stablehlo::DotGeneralOp>(
      loc, downTy, siluOut, wDownArg, noBatch, emptyPrec, noAlg);

  b.create<func::ReturnOp>(loc, ValueRange{downVal.getResult()});
  return func;
}

// ---------------------------------------------------------------------------
// Replace slot in @main
// ---------------------------------------------------------------------------

static void replaceSlotInMain(func::FuncOp mainFunc, ExpertSlot &slot,
                               func::FuncOp outlinedFunc, int64_t numExperts) {
  MLIRContext *ctx = mainFunc.getContext();
  Location loc = mainFunc.getLoc();
  auto f32 = Float32Type::get(ctx);

  // Insert new ops immediately before the gate op.
  OpBuilder b(slot.gateOp);

  auto batchedDispatchType  = mlir::cast<RankedTensorType>(slot.dispatched.getType());
  auto batchedGateWeightType = mlir::cast<RankedTensorType>(slot.wGate.getType());
  auto batchedDownWeightType = mlir::cast<RankedTensorType>(slot.wDown.getType());

  int64_t numTokens        = batchedDispatchType.getDimSize(1);   // 512   — token dim
  int64_t hiddenDim        = batchedDispatchType.getDimSize(2);   // 4096  — hidden dim
  int64_t gateWeightInDim  = batchedGateWeightType.getDimSize(1); // 4096  — gate/up rhs dim 1
  int64_t intermediateDim  = batchedGateWeightType.getDimSize(2); // 14336 — intermediate dim
  int64_t downWeightInDim  = batchedDownWeightType.getDimSize(1); // 14336 — down rhs dim 1
  int64_t downWeightOutDim = batchedDownWeightType.getDimSize(2); // 4096  — down rhs dim 2

  SmallVector<Value> perExpertOuts;
  perExpertOuts.reserve(numExperts);

  for (int64_t e = 0; e < numExperts; ++e) {
    // slice dispatched[e:e+1, :, :] → reshape → [numTokens, hiddenDim]
    auto dispatchSlice = b.create<stablehlo::SliceOp>(
        loc, slot.dispatched,
        DenseI64ArrayAttr::get(ctx, {e, 0, 0}),
        DenseI64ArrayAttr::get(ctx, {e + 1, numTokens, hiddenDim}),
        DenseI64ArrayAttr::get(ctx, {1, 1, 1}));
    auto dispatchReshaped = b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({numTokens, hiddenDim}, f32), dispatchSlice);

    // slice w_gate[e:e+1, :, :] → reshape → [gateWeightInDim, intermediateDim]
    auto gateWeightSlice = b.create<stablehlo::SliceOp>(
        loc, slot.wGate,
        DenseI64ArrayAttr::get(ctx, {e, 0, 0}),
        DenseI64ArrayAttr::get(ctx, {e + 1, gateWeightInDim, intermediateDim}),
        DenseI64ArrayAttr::get(ctx, {1, 1, 1}));
    auto gateWeightReshaped = b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({gateWeightInDim, intermediateDim}, f32), gateWeightSlice);

    // slice w_up[e:e+1, :, :] → reshape → [gateWeightInDim, intermediateDim]
    auto upWeightSlice = b.create<stablehlo::SliceOp>(
        loc, slot.wUp,
        DenseI64ArrayAttr::get(ctx, {e, 0, 0}),
        DenseI64ArrayAttr::get(ctx, {e + 1, gateWeightInDim, intermediateDim}),
        DenseI64ArrayAttr::get(ctx, {1, 1, 1}));
    auto upWeightReshaped = b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({gateWeightInDim, intermediateDim}, f32), upWeightSlice);

    // slice w_down[e:e+1, :, :] → reshape → [downWeightInDim, downWeightOutDim]
    auto downWeightSlice = b.create<stablehlo::SliceOp>(
        loc, slot.wDown,
        DenseI64ArrayAttr::get(ctx, {e, 0, 0}),
        DenseI64ArrayAttr::get(ctx, {e + 1, downWeightInDim, downWeightOutDim}),
        DenseI64ArrayAttr::get(ctx, {1, 1, 1}));
    auto downWeightReshaped = b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({downWeightInDim, downWeightOutDim}, f32), downWeightSlice);

    // call @expert_slot_N → [numTokens, hiddenDim]
    auto callOp = b.create<func::CallOp>(
        loc, outlinedFunc,
        ValueRange{dispatchReshaped, gateWeightReshaped, upWeightReshaped, downWeightReshaped});

    // reshape [numTokens, hiddenDim] → [1, numTokens, hiddenDim] for concatenation
    auto batchedResult = b.create<stablehlo::ReshapeOp>(
        loc, RankedTensorType::get({1, numTokens, hiddenDim}, f32), callOp.getResult(0));
    perExpertOuts.push_back(batchedResult.getResult());
  }

  // concatenate [1, numTokens, hiddenDim] × numExperts → [numExperts, numTokens, hiddenDim]
  auto concatTy = RankedTensorType::get({numExperts, numTokens, hiddenDim}, f32);
  auto concat = b.create<stablehlo::ConcatenateOp>(
      loc, concatTy, ValueRange(perExpertOuts), /*dimension=*/0);

  // Replace all uses of the old down result with the concat result.
  slot.downOp.getResult().replaceAllUsesWith(concat.getResult());

  // Erase now-dead ops in reverse topological order.
  slot.downOp.erase();
  for (auto it = slot.swigluOps.rbegin(); it != slot.swigluOps.rend(); ++it)
    (*it)->erase();
  slot.upOp.erase();
  slot.gateOp.erase();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ExpertOutliningPass
    : public PassWrapper<ExpertOutliningPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertOutliningPass)

  Option<int64_t> numExpertsOpt{
      *this, "num_experts",
      llvm::cl::desc("Number of experts per slot (default: 8)"),
      llvm::cl::init(8)
  };

  // User-defined copy constructor suppresses the implicit default constructor,
  // so we must restore it explicitly.
  ExpertOutliningPass() = default;

  // Option<T> inherits from llvm::cl::opt<T> which deletes its copy
  // constructor.  PassWrapper::clonePass() uses copy construction, so we
  // must provide our own copy constructor that default-constructs the base
  // (giving a fresh pass with options registered) and then copies the value.
  ExpertOutliningPass(const ExpertOutliningPass &src) {
    numExpertsOpt = static_cast<int64_t>(src.numExpertsOpt);
  }

  StringRef getArgument() const override { return "moe-expert-outlining"; }
  StringRef getDescription() const override {
    return "Outline MoE expert FFN subgraphs into per-expert @expert_slot_N functions";
  }

  void runOnOperation() override {
    int64_t numExperts = numExpertsOpt;
    ModuleOp module = getOperation();

    // Find @main.
    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp f) {
      if (f.getName() == "main") {
        mainFunc = f;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!mainFunc) {
      module.emitError("moe-expert-outlining: no @main function found");
      signalPassFailure();
      return;
    }

    // Collect all expert-batch DotGeneralOps in @main.
    SmallVector<stablehlo::DotGeneralOp> expertDots;
    mainFunc.walk([&](stablehlo::DotGeneralOp op) {
      if (isExpertBatchDot(op, numExperts))
        expertDots.push_back(op);
    });
    if (expertDots.empty())
      return;

    // Group by lhs operand:
    //   gate + up share the same dispatched_tokens → group size 2
    //   down has a unique lhs (SwiGLU output)      → group size 1
    DenseMap<Value, SmallVector<stablehlo::DotGeneralOp>> byLhs;
    for (auto op : expertDots)
      byLhs[op.getLhs()].push_back(op);

    SmallVector<SmallVector<stablehlo::DotGeneralOp>> gateUpGroups;
    SmallVector<stablehlo::DotGeneralOp> downOps;
    for (auto &[lhs, ops] : byLhs) {
      if (ops.size() == 2)
        gateUpGroups.push_back(ops);
      else if (ops.size() == 1)
        downOps.push_back(ops[0]);
      else
        module.emitWarning(
            "moe-expert-outlining: unexpected group size; skipping slot");
    }

    if (gateUpGroups.size() != downOps.size()) {
      module.emitError(
          "moe-expert-outlining: gate/up group count (" +
          llvm::Twine(gateUpGroups.size()) + ") != down op count (" +
          llvm::Twine(downOps.size()) + ")");
      signalPassFailure();
      return;
    }

    // Sort groups and down ops by IR position so slot IDs match block order.
    llvm::sort(gateUpGroups, [](const auto &a, const auto &b) {
      return a.front()->isBeforeInBlock(b.front());
    });
    llvm::sort(downOps, [](auto a, auto b) {
      return a->isBeforeInBlock(b);
    });

    // Build slot descriptors.
    SmallVector<ExpertSlot> slots;
    for (size_t i = 0; i < gateUpGroups.size(); ++i) {
      auto &gateUp = gateUpGroups[i];
      // Ensure gate comes before up in SSA order.
      llvm::sort(gateUp, [](auto a, auto b) {
        return a->isBeforeInBlock(b);
      });
      stablehlo::DotGeneralOp gateOp = gateUp[0];
      stablehlo::DotGeneralOp upOp   = gateUp[1];

      // Find the down op whose SwiGLU chain traces back to this gate/up pair.
      DenseSet<Value> boundary = {gateOp.getResult(), upOp.getResult()};
      stablehlo::DotGeneralOp matchedDown;
      for (auto downOp : downOps) {
        SmallVector<Value> wl = {downOp.getLhs()};
        DenseSet<Value> vis;
        bool found = false;
        while (!wl.empty() && !found) {
          Value v = wl.pop_back_val();
          if (!vis.insert(v).second)
            continue;
          if (boundary.count(v)) {
            found = true;
            break;
          }
          if (auto *def = v.getDefiningOp())
            if (!isa<stablehlo::ConstantOp>(def))
              for (Value op : def->getOperands())
                wl.push_back(op);
        }
        if (found) {
          matchedDown = downOp;
          break;
        }
      }

      if (!matchedDown) {
        module.emitError("moe-expert-outlining: no down op found for slot " +
                         llvm::Twine(i));
        signalPassFailure();
        return;
      }

      ExpertSlot slot;
      slot.slotId     = static_cast<int>(i);
      slot.dispatched  = gateOp.getLhs();
      slot.gateOp     = gateOp;
      slot.upOp       = upOp;
      slot.downOp     = matchedDown;
      slot.wGate      = gateOp.getRhs();
      slot.wUp        = upOp.getRhs();
      slot.wDown      = matchedDown.getRhs();
      slot.swigluOps  = collectSwiGLU(matchedDown, gateOp.getResult(),
                                       upOp.getResult());
      slots.push_back(std::move(slot));
    }

    // Build all outlined functions first (adds them to the module), then
    // rewrite @main.  Building first avoids iterator invalidation during
    // the walk that found the slots.
    SmallVector<func::FuncOp> outlinedFuncs;
    for (auto &slot : slots) {
      auto result = buildExpertSlotFunc(module, slot, numExperts);
      if (failed(result)) {
        signalPassFailure();
        return;
      }
      outlinedFuncs.push_back(*result);
    }

    for (int i = 0; i < static_cast<int>(slots.size()); ++i)
      replaceSlotInMain(mainFunc, slots[i], outlinedFuncs[i], numExperts);
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> createExpertOutliningPass(int64_t /*numExperts*/) {
  // numExperts is now configured via the pipeline option string:
  //   --pass-pipeline='moe-expert-outlining{num_experts=N}'
  return std::make_unique<ExpertOutliningPass>();
}

void registerExpertOutliningPass() {
  PassRegistration<ExpertOutliningPass>();
}

} // namespace remora::passes
