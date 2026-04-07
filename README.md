# remora

[Mystic Remora](https://scryfall.com/card/ice/87/mystic-remora)

```bash
# Lower a StableHLO MoE to PTX and print all kernels (no GPU required)
compiler/build/remora mlir/stablehlo/simple_moe.mlir --emit-ptx

# JIT-compile and run on CPU
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --test=elementwise
compiler/build/remora mlir/stablehlo/simple_attention_projection.mlir  --test=projection

# Compile and run on GPU, validate output (requires CUDA)
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --test=elementwise --run-gpu

# Dump IR after every lowering pass
compiler/build/remora mlir/stablehlo/simple_attention_projection.mlir --test=projection --mlir-print-ir-after-all
```

## Prerequisites

- `git`, `cmake` >= 3.20, `ninja`, `python3`
- ~30 GB disk space (LLVM build)
- NVIDIA GPU + CUDA toolkit required for `--run-gpu` (not required for `--emit-ptx`)

## Setup

### 1. Build MLIR and stablehlo-opt from source

```bash
scripts/bootstrap.sh
```

Add `--nvptx` if you want `--emit-ptx` / `--run-gpu` support (adds NVPTX target and build tools; takes longer):

```bash
scripts/bootstrap.sh --nvptx
```

To use a custom build directory:

```bash
scripts/bootstrap.sh --build-dir /path/to/build-deps
```

Bootstrap prints the exact `.env` values you need for the next step.

### 2. Configure .env

```bash
cp .env.example .env
```

Fill in the paths printed by bootstrap:

```bash
MLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
STABLEHLO_ROOT=/path/to/stablehlo
STABLEHLO_BUILD=/path/to/stablehlo/build
LLVM_NVPTX_LIB_DIR=/path/to/llvm-project/build/lib  # needed for --emit-ptx
```

### 3. Build remora

```bash
sh scripts/build.sh
```

Produces `compiler/build/remora`.

---

## Usage: MoE end-to-end

### 1. Export StableHLO from JAX

```bash
cd scripts/export
pip install -r requirements.txt
python simple_moe.py
```

Writes `mlir/stablehlo/simple_moe.mlir` — a 2-expert SwiGLU MoE (T=8 tokens, D=32 hidden, E=2 experts, F=64 FFN dim).

### 2. CPU path

Lower through StableHLO → Linalg → LLVM dialect, JIT-compile, and execute on CPU:

```bash
compiler/build/remora mlir/stablehlo/simple_moe.mlir --test=elementwise
```

### 3. GPU path: emit PTX

Lower through the GPU pipeline and emit PTX for each kernel. Runs on CPU (no GPU required):

```bash
compiler/build/remora mlir/stablehlo/simple_moe.mlir --emit-ptx
```

Emits one PTX blob per `gpu.module`. For the MoE this produces 17 kernels.

### 4. GPU path: run on device

Requires a GPU and a build with CUDA toolkit present (`REMORA_CUDA` defined):

```bash
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir \
  --test=elementwise --run-gpu
```

Launches the kernel via the CUDA Driver API and validates output against the expected value (max abs error < 1e-5).

---

## Other kernels

The elementwise (`relu(x + bias)`) and projection (`matmul(x, w)`) kernels are simpler scaffolding for validating the pipeline:

```bash
# CPU JIT
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --test=elementwise
compiler/build/remora mlir/stablehlo/simple_attention_projection.mlir  --test=projection

# Wrapper scripts
sh scripts/run/run_elementwise.sh
sh scripts/run/run_projection.sh

# Verify against JAX reference
diff <(sh scripts/run/run_elementwise.sh) <(python scripts/verify/verify_elementwise.py)
diff <(sh scripts/run/run_projection.sh)  <(python scripts/verify/verify_projection.py)
```

---

## Debugging

Dump IR after each lowering pass:

```bash
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir \
  --test=elementwise --mlir-print-ir-after-all
```

Lower StableHLO → Linalg manually and inspect:

```bash
scripts/explore/lower_elementwise_to_linalg.sh
scripts/explore/lower_projection_to_linalg.sh

diff mlir/stablehlo/simple_attention_elementwise.mlir \
     mlir/linalg/attention_elementwise_lowered_to_linalg.mlir
```

Step through progressive lowering passes:

```bash
scripts/explore/elementwise-explore.sh
```
