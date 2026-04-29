#!/bin/bash
# scripts/reproduce_all.sh
#
# Reproduce every paper result end-to-end.
#
# Stages:
#   1. Compiler pipeline   — emit compiler_decisions.json
#   2. IR verification     — verify outlining, cost analysis, specialization (143 checks)
#   3. Wire test           — end-to-end JSON wire test
#   4. GPU correctness     — heterogeneous kernel correctness check (requires CUDA GPU)
#   5. GPU benchmarks      — routing skew, expert count, heterogeneous specialization
#
# Stages 1–3 run without a GPU (CPU-only, requires compiler/build/remora).
# Stages 4–5 require a CUDA GPU and Python packages:
#   pip install -r benchmarks/requirements.txt
#
# Usage:
#   bash scripts/reproduce_all.sh              # all stages
#   bash scripts/reproduce_all.sh --no-gpu     # stages 1–3 only
#   bash scripts/reproduce_all.sh --gpu-only   # stages 4–5 only
#
# Output files (written to benchmarks/results/):
#   heterogeneous_correctness.json   -- §5.4 correctness claim
#   sweep_skew.json / .csv           -- §5.2 Table 1
#   sweep_expert_count.json / .csv   -- §5.3 Table 2
#   sweep_heterogeneous.json / .csv  -- §5.4 Table 3
#
# Paper claim ↔ result file mapping is documented in docs/reproduce.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO_ROOT/benchmarks/results"
COMPILER="$REPO_ROOT/compiler/build/remora"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
RUN_CPU=1
RUN_GPU=1

for arg in "$@"; do
    case "$arg" in
        --no-gpu)  RUN_GPU=0 ;;
        --gpu-only) RUN_CPU=0 ;;
        -h|--help)
            sed -n '2,35p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ok()   { echo "  [OK]  $*"; }
fail() { echo "  [FAIL] $*" >&2; exit 1; }
hdr()  { echo; echo "=== $* ==="; }

mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Stage 1: Compiler pipeline
# ---------------------------------------------------------------------------
if [ "$RUN_CPU" -eq 1 ]; then
    hdr "Stage 1 — Compiler pipeline"

    [ -x "$COMPILER" ] || fail "compiler not found at $COMPILER — run: sh scripts/build.sh"

    DECISIONS="$REPO_ROOT/compiler_decisions.json"
    python3 "$REPO_ROOT/scripts/emit_compiler_decisions.py" \
        --input "$REPO_ROOT/mlir/stablehlo/heterogeneous_moe_layer.mlir" \
        --num-experts 1 \
        --output "$DECISIONS"
    ok "compiler_decisions.json written to $DECISIONS"

    # ---------------------------------------------------------------------------
    # Stage 2: IR verification (CPU)
    # ---------------------------------------------------------------------------
    hdr "Stage 2 — IR verification (CPU)"

    python3 "$REPO_ROOT/scripts/verify/verify_heterogeneous_outlining.py"
    ok "verify_heterogeneous_outlining.py — passed"

    python3 "$REPO_ROOT/scripts/verify/verify_cost_analysis.py"
    ok "verify_cost_analysis.py — passed"

    python3 "$REPO_ROOT/scripts/verify/verify_specialization.py"
    ok "verify_specialization.py — passed"

    # ---------------------------------------------------------------------------
    # Stage 3: Wire test
    # ---------------------------------------------------------------------------
    hdr "Stage 3 — Wire test"

    bash "$REPO_ROOT/scripts/test_wire.sh"
    ok "test_wire.sh — Wire test PASSED"
fi

# ---------------------------------------------------------------------------
# Stage 4: GPU correctness
# ---------------------------------------------------------------------------
if [ "$RUN_GPU" -eq 1 ]; then
    hdr "Stage 4 — GPU correctness (requires CUDA GPU)"

    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'" \
        || fail "No CUDA GPU available — re-run with --no-gpu to skip GPU stages"

    python3 "$REPO_ROOT/benchmarks/verify_heterogeneous_correctness.py"
    ok "verify_heterogeneous_correctness.py — all experts PASS"
    ok "Results → $RESULTS/heterogeneous_correctness.json"

    # ---------------------------------------------------------------------------
    # Stage 5: GPU benchmarks
    # ---------------------------------------------------------------------------
    hdr "Stage 5 — GPU benchmarks (requires CUDA GPU)"

    echo "  [5a] Routing skew sweep (§5.2 Table 1)..."
    python3 "$REPO_ROOT/benchmarks/sweep_skew.py"
    ok "sweep_skew.py → $RESULTS/sweep_skew.json"

    echo "  [5b] Expert count sweep (§5.3 Table 2)..."
    python3 "$REPO_ROOT/benchmarks/sweep_expert_count.py"
    ok "sweep_expert_count.py → $RESULTS/sweep_expert_count.json"

    echo "  [5c] Heterogeneous specialization benchmark (§5.4 Table 3)..."
    python3 "$REPO_ROOT/benchmarks/sweep_heterogeneous.py"
    ok "sweep_heterogeneous.py → $RESULTS/sweep_heterogeneous.json"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  reproduce_all.sh COMPLETE"
echo "  Results written to: $RESULTS/"
if [ "$RUN_GPU" -eq 1 ]; then
    echo
    echo "  Paper result ↔ file mapping:"
    echo "    §5.2 Table 1  →  sweep_skew.json"
    echo "    §5.3 Table 2  →  sweep_expert_count.json"
    echo "    §5.4 Table 3  →  sweep_heterogeneous.json"
    echo "    §5.4 correct. →  heterogeneous_correctness.json"
fi
echo "  See docs/reproduce.md for full claim-to-command mapping."
echo "============================================================"
