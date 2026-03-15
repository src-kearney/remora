#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "stablehlo/dialect/Register.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: remora-compiler <input.mlir>\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  mlir::MLIRContext ctx(registry);

  mlir::registerAllPasses();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse: " << argv[1] << "\n";
    return 1;
  }

  module->print(llvm::outs());
  return 0;
}
