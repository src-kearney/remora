#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

int runGpu(mlir::ModuleOp module, bool debug, bool launchOnGpu,
           llvm::StringRef test);
