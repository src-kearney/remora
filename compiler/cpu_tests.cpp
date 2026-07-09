#include "cpu_tests.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

using namespace mlir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static llvm::Expected<void (*)(void *, void *, void *)>
lookupTriple(ExecutionEngine &engine) {
  auto sym = engine.lookup("_mlir_ciface_main");
  if (!sym) return sym.takeError();
  return reinterpret_cast<void (*)(void *, void *, void *)>(*sym);
}

// ---------------------------------------------------------------------------
// Elementwise: relu(x + bias)
// Inputs: x = 1.0, bias = -0.5  →  expected 0.5 everywhere
// ---------------------------------------------------------------------------

static int runElementwise(ExecutionEngine &engine) {
  const int64_t N = 1, T = 512, D = 768;
  std::vector<float> x_data(N * T * D, 1.0f);
  std::vector<float> bias_data(D, -0.5f);

  StridedMemRefType<float, 3> x_desc;
  x_desc.basePtr = x_desc.data = x_data.data();
  x_desc.offset = 0;
  x_desc.sizes[0] = N; x_desc.sizes[1] = T; x_desc.sizes[2] = D;
  x_desc.strides[0] = T * D; x_desc.strides[1] = D; x_desc.strides[2] = 1;

  StridedMemRefType<float, 1> bias_desc;
  bias_desc.basePtr = bias_desc.data = bias_data.data();
  bias_desc.offset = 0;
  bias_desc.sizes[0] = D;
  bias_desc.strides[0] = 1;

  StridedMemRefType<float, 3> result;

  auto fn = lookupTriple(engine);
  if (!fn) {
    llvm::handleAllErrors(fn.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  (*fn)(&result, &x_desc, &bias_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 0.5)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 0.5)\n";
  free(result.basePtr);
  return 0;
}

// ---------------------------------------------------------------------------
// Projection: matmul(x, w)
// Inputs: x = 1.0, w = 1/768  →  expected 1.0 everywhere
// ---------------------------------------------------------------------------

static int runProjection(ExecutionEngine &engine) {
  const int64_t N = 1, T = 512, D = 768;
  std::vector<float> x_data(N * T * D, 1.0f);
  std::vector<float> w_data(D * D, 1.0f / D);

  StridedMemRefType<float, 3> x_desc;
  x_desc.basePtr = x_desc.data = x_data.data();
  x_desc.offset = 0;
  x_desc.sizes[0] = N; x_desc.sizes[1] = T; x_desc.sizes[2] = D;
  x_desc.strides[0] = T * D; x_desc.strides[1] = D; x_desc.strides[2] = 1;

  StridedMemRefType<float, 2> w_desc;
  w_desc.basePtr = w_desc.data = w_data.data();
  w_desc.offset = 0;
  w_desc.sizes[0] = D; w_desc.sizes[1] = D;
  w_desc.strides[0] = D; w_desc.strides[1] = 1;

  StridedMemRefType<float, 3> result;

  auto fn = lookupTriple(engine);
  if (!fn) {
    llvm::handleAllErrors(fn.takeError(), [](const llvm::ErrorInfoBase &e) {
      llvm::errs() << "Symbol lookup failed: " << e.message() << "\n";
    });
    return 1;
  }
  (*fn)(&result, &x_desc, &w_desc);

  llvm::outs() << "result[0][0][0] = " << result.data[0] << " (expected 1.0)\n";
  llvm::outs() << "result[0][0][1] = " << result.data[1] << " (expected 1.0)\n";
  free(result.basePtr);
  return 0;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

int runCpuTest(ExecutionEngine &engine, llvm::StringRef test) {
  if (test == "projection") return runProjection(engine);
  return runElementwise(engine);
}
