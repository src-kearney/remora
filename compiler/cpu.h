#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

int runCpu(mlir::ModuleOp module, llvm::StringRef kernel, bool debug);
