# remora

[Mystic Remora](https://scryfall.com/card/ice/87/mystic-remora)

An MLIR/StableHLO compiler prototype that recovers per-expert program structure from batched MoE IR and uses it to make expert-specific compilation decisions.

## What it does

Batched MoE compilation erases expert identity — all experts become one `dot_general` with a batch dim. Remora runs three passes over the IR to recover it:

1. **moe-expert-outlining** — extracts each expert into its own `@expert_slot_N` function with static shapes
2. **moe-expert-cost-analysis** — computes FLOPs, bytes, arithmetic intensity per expert from tensor shapes
3. **moe-expert-specialization** — selects tile configs (BLOCK_M, BLOCK_N) per expert based on cost class

The compiler writes `compiler_decisions.json`; the Python dispatch reads it and fires the right Triton kernel per expert.

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
