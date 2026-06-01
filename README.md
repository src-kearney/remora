# remora

[Mystic Remora](https://scryfall.com/card/ice/87/mystic-remora)

`remora` is my attempt to solve the most important problem in the world. That particular problem: figuring out what the most important problem is.

An MLIR/StableHLO compiler prototype. Still tinkering.

## Setup

```bash
scripts/bootstrap.sh          # builds LLVM + StableHLO (~30 GB, takes a while)
scripts/bootstrap.sh --nvptx  # add NVPTX for --emit-ptx / --run-gpu
```

Bootstrap clones `llvm-project` and `stablehlo` into `build-deps/` and prints
the exact `.env` values at the end:

```
MLIR_DIR=build-deps/llvm-build/lib/cmake/mlir
STABLEHLO_ROOT=build-deps/stablehlo
STABLEHLO_BUILD=build-deps/stablehlo/build
```

```bash
cp .env.example .env          # paste the paths above into .env
sh scripts/build.sh           # produces compiler/build/remora
```

**GPU benchmarks** (sweep scripts, correctness check) require a CUDA environment:
```bash
pip install -r benchmarks/requirements.txt
# vLLM separately if running comparison benchmarks — see requirements.txt
```

## Run the pipeline

```bash
# Full three-pass pipeline on the heterogeneous MoE IR
python3 scripts/emit_compiler_decisions.py \
    --input mlir/stablehlo/heterogeneous_moe_layer.mlir \
    --num-experts 1 \
    --output compiler_decisions.json

# End-to-end wire test
bash scripts/test_wire.sh
```

## Verify

```bash
python3 scripts/verify/verify_heterogeneous_outlining.py  # 40 checks
python3 scripts/verify/verify_cost_analysis.py            # 58 checks
python3 scripts/verify/verify_specialization.py           # 45 checks
```

## Inspect IR directly

```bash
compiler/build/remora mlir/stablehlo/heterogeneous_moe_layer.mlir \
    --pass-pipeline='moe-expert-outlining{num_experts=1},moe-expert-cost-analysis,moe-expert-specialization' \
    --no-execute \
    --dump-compilation-phases-to=out/
```
