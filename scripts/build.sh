#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/compiler/build"

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo "Error: .env not found at $ENV_FILE"
  echo "Copy .env.example to .env and fill in your paths"
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$REPO_ROOT/compiler" \
  -GNinja \
  -DMLIR_DIR="$MLIR_DIR" \
  -DSTABLEHLO_ROOT="$STABLEHLO_ROOT" \
  -DSTABLEHLO_BUILD="$STABLEHLO_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build .