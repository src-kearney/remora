# [remora](https://scryfall.com/card/ice/87/mystic-remora)

A StableHLO → MLIR → PTX compiler, with a CPU/LLVM reference path.

`remora` lowers StableHLO through a sequence of MLIR passes to NVIDIA PTX (via NVVM) or to native code on CPU (via LLVM). The goal of `remora` is to study **heterogeneous GPU lowering**, and to provide infrastructure to measure how the same compiler-generated code performs across different GPU vendors, compared against each vendor's peak.

## Status

| Component | Status |
|---|---|
| NVIDIA (PTX) lowering | done |
| Benchmark harness (GEMM, FlashAttention, % of peak) | in progress |
| AMD (ROCDL) lowering | planned |
| Cross-vendor perf harness | planned |

## Build

```bash
scripts/bootstrap.sh          # builds LLVM + StableHLO into build-deps/ (~30 GB)
scripts/bootstrap.sh --nvptx  # add the NVPTX backend for --emit-ptx / --run-gpu

cp .env.example .env          # paste the paths bootstrap prints at the end
sh scripts/build.sh           # produces compiler/build/remora
```

## Usage

```bash
# Lower StableHLO and emit PTX
compiler/build/remora mlir/stablehlo/simple_moe.mlir --emit-ptx

# Run on GPU (requires building with a CUDA toolkit present)
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --run-gpu --test=elementwise

# Run on CPU via the LLVM ExecutionEngine
compiler/build/remora mlir/stablehlo/simple_attention_elementwise.mlir --test=elementwise

# Inspect the lowering: dump IR after every pass into a directory
compiler/build/remora mlir/stablehlo/simple_moe.mlir --emit-ptx --no-execute \
    --dump-compilation-phases-to=out/
```

## How it works

Remora ingests StableHLO and runs a lowering pipeline (StableHLO → linalg → … → NVVM/LLVM), emitting PTX for NVIDIA or native code on CPU. The `--dump-compilation-phases-to` flag writes a numbered MLIR snapshot after each pass, so you can watch the IR transform step by step.
