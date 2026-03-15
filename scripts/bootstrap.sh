#!/bin/bash
# Bootstrap: builds MLIR and stablehlo-opt from source.
# Tested on macOS (Apple Silicon). Should work on Linux with standard build tools.
#
# Usage:
#   scripts/bootstrap.sh [--build-dir <path>]
#
# Environment variables:
#   BUILD_DIR   Directory for cloned repos and build artifacts (default: <repo-root>/build-deps)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Parse arguments
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-deps}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Check prerequisites
for cmd in git cmake ninja python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' is required but not found in PATH."; exit 1; }
done

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone stablehlo if not already present
if [ ! -d stablehlo ]; then
  git clone https://github.com/openxla/stablehlo.git
else
  echo "stablehlo already cloned, skipping."
fi

# Clone llvm-project if not already present
if [ ! -d llvm-project ]; then
  git clone https://github.com/llvm/llvm-project.git
else
  echo "llvm-project already cloned, skipping."
fi

LLVM_COMMIT=$(cat stablehlo/build_tools/llvm_version.txt)
echo "Checking out LLVM at $LLVM_COMMIT"
cd llvm-project && git checkout "$LLVM_COMMIT" && cd ..

sh stablehlo/build_tools/build_mlir.sh "$BUILD_DIR/llvm-project" "$BUILD_DIR/llvm-build"

cmake -GNinja -B stablehlo/build \
  -S stablehlo \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$BUILD_DIR/llvm-build/lib/cmake/mlir" \
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build stablehlo/build --target stablehlo-opt

echo ""
echo "Build complete. stablehlo-opt is at:"
echo "  $BUILD_DIR/stablehlo/build/bin/stablehlo-opt"
echo ""
echo "Set this in your environment:"
echo "  export STABLEHLO_OPT=$BUILD_DIR/stablehlo/build/bin/stablehlo-opt"
