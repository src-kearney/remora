# remora

[Remora](https://scryfall.com/card/ice/87/mystic-remora)

## Directory structure

```
remora/
├── .env.example          # Template for local build paths (copy to .env)
├── jax/                  # JAX model definitions and verification scripts
├── mlir/
│   ├── stablehlo/        # StableHLO IR exported from JAX
│   └── linalg/           # Linalg IR lowered from StableHLO
├── compiler/
│   ├── main.cpp          # Pass pipeline, JIT harness, kernel invocation
│   ├── CMakeLists.txt    # Build config (reads MLIR_DIR, STABLEHLO_ROOT/BUILD from .env)
│   └── build/            # CMake build output (gitignored)
├── md/                   # Design docs and quiz
└── scripts/
    ├── bootstrap.sh                              # Build MLIR + stablehlo-opt from source
    ├── build.sh                                  # Build remora (requires .env)
    ├── run_elementwise.sh                        # JIT-run elementwise kernel, print output
    ├── run_projection.sh                         # JIT-run projection kernel, print output
    ├── attention_elementwise_lower_to_linalg.sh  # Lower elementwise StableHLO → Linalg
    ├── attention_projection_lowered_to_linalg.sh # Lower projection StableHLO → Linalg
    └── elementwise-explore.sh                    # Step through lowering passes interactively
```

## Prerequisites

- `git`
- `cmake` >= 3.20
- `ninja`
- `python3`
- ~30 GB disk space (LLVM build is large)

## Setup

### 1. Build MLIR and stablehlo-opt from source

Run the bootstrap script from anywhere — it resolves all paths relative to the repo root and clones dependencies into `build-deps/` by default.

```bash
scripts/bootstrap.sh
```

To use a custom build directory:

```bash
scripts/bootstrap.sh --build-dir /path/to/your/build-deps
# or
BUILD_DIR=/path/to/your/build-deps scripts/bootstrap.sh
```

When the build finishes, the script prints the paths you'll need for the next step.

### 2. Configure your .env

The build scripts read local paths from a `.env` file (gitignored). Copy the example and fill in the paths printed by `bootstrap.sh`:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
MLIR_DIR=/path/to/build-deps/llvm-build/lib/cmake/mlir
STABLEHLO_ROOT=/path/to/build-deps/stablehlo
STABLEHLO_BUILD=/path/to/build-deps/stablehlo/build
```

If you used the default `build-deps/` location these will be `<repo-root>/build-deps/...`.

### 3. Build remora

```bash
sh scripts/build.sh
```

This sources `.env`, runs CMake, and produces `compiler/build/remora`.

## Usage

### Run the JIT kernels

`remora` lowers a StableHLO file through the full pass pipeline to LLVM dialect, JIT-compiles it via LLVM ORC, and executes it on CPU. Use `--kernel` to select the test harness matching the input file.

```bash
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --kernel=elementwise
compiler/build/remora mlir/stablehlo/simple_attention_projection.mlir  --kernel=projection
```

Or via the wrapper scripts:

```bash
sh scripts/run_elementwise.sh   # relu(x + bias), expects 0.5
sh scripts/run_projection.sh    # matmul(x, w),   expects ~1.0
```

Pass `--mlir-print-ir-after-all` to dump IR after each lowering pass:

```bash
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --kernel=elementwise --mlir-print-ir-after-all
```

### Validate against JAX

The `jax/verify_*.py` scripts run the same kernels in JAX with identical inputs and print output in the same format. Both should produce bit-identical results.

```bash
diff <(sh scripts/run_elementwise.sh) <(jax/.venv/bin/python jax/verify_elementwise.py)
diff <(sh scripts/run_projection.sh)  <(jax/.venv/bin/python jax/verify_projection.py)
```

### Lower StableHLO → Linalg

These scripts use `stablehlo-opt` to lower to Linalg and write the result to `mlir/linalg/`:

```bash
scripts/attention_elementwise_lower_to_linalg.sh
scripts/attention_projection_lowered_to_linalg.sh
```

To see what the fusion pass does — StableHLO broadcast/add/relu collapsed into a single fused `linalg.generic` — diff the input against the output:

```bash
diff mlir/stablehlo/simple_attention_elementwise.mlir \
     mlir/linalg/attention_elementwise_lowered_to_linalg.mlir
```

Set `STABLEHLO_OPT` if `stablehlo-opt` is not on your PATH:

```bash
export STABLEHLO_OPT=/path/to/build-deps/stablehlo/build/bin/stablehlo-opt
```

### Explore lowering passes interactively

```bash
scripts/elementwise-explore.sh
```

Runs the elementwise file through several progressive lowering steps and prints each result to stdout.

### (Optional) Export fresh StableHLO from JAX

```bash
cd jax
pip install -r requirements.txt
python simple_attention_elementwise.py > ../mlir/stablehlo/simple_attention_elementwise.mlir
python simple_attention_projection.py > ../mlir/stablehlo/simple_attention_projection.mlir
```

Both scripts print the exported StableHLO module to stdout.
