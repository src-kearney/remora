#include "cpu.h"
#include "gpu.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "stablehlo/dialect/Register.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: remora <input.mlir> "
                    "--kernel=<elementwise|projection> "
                    "[--emit-ptx] [--run-gpu] "
                    "[--mlir-print-ir-after-all]\n";
    return 1;
  }

  llvm::StringRef kernel;
  bool debug = false;
  bool emitPtxMode = false;
  bool runGpuMode = false;
  for (int i = 2; i < argc; i++) {
    llvm::StringRef arg(argv[i]);
    if (arg.starts_with("--kernel="))
      kernel = arg.drop_front(9);
    else if (arg == "--mlir-print-ir-after-all")
      debug = true;
    else if (arg == "--emit-ptx")
      emitPtxMode = true;
    else if (arg == "--run-gpu")
      runGpuMode = true;
  }

  if (!emitPtxMode && !runGpuMode &&
      kernel != "elementwise" && kernel != "projection") {
    llvm::errs() << "Unknown kernel '" << kernel
                 << "'. Use --kernel=elementwise or --kernel=projection\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);

  mlir::MLIRContext ctx(registry);
  if (debug) ctx.disableMultithreading();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse: " << argv[1] << "\n";
    return 1;
  }

  if (emitPtxMode || runGpuMode)
    return runGpu(*module, debug, runGpuMode, kernel);
  return runCpu(*module, kernel, debug);
}
