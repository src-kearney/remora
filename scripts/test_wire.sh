#!/bin/bash
# scripts/test_wire.sh
#
# End-to-end wire test: compiler produces compiler_decisions.json, Python
# assertions verify the per-expert configs are correct.
#
# Definition of done: prints "Wire test PASSED" with exit 0.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/compiler_decisions.json"

# ---------------------------------------------------------------------------
# Step 1 — Run the full three-pass pipeline, emit decisions JSON
# ---------------------------------------------------------------------------
python3 "$REPO_ROOT/scripts/emit_compiler_decisions.py" \
    --input "$REPO_ROOT/mlir/stablehlo/heterogeneous_moe_layer.mlir" \
    --num-experts 1 \
    --output "$OUT"

# ---------------------------------------------------------------------------
# Step 2 — Verify the JSON has correct per-expert configs
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'EOF'
import json, sys

d = json.load(open(sys.argv[1]))

failures = []

def check(name, condition, got=None):
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f" (got {got!r})" if got is not None else "")
        print(msg)
        failures.append(name)

# E0, E1: F=14336 → large → BLOCK_M=128, BLOCK_N=128
check("expert_slot_0 BLOCK_M=128",
      d["expert_slot_0"]["BLOCK_M"] == 128, d["expert_slot_0"]["BLOCK_M"])
check("expert_slot_0 BLOCK_N=128",
      d["expert_slot_0"]["BLOCK_N"] == 128, d["expert_slot_0"]["BLOCK_N"])
check("expert_slot_0 intermediate_dim=14336",
      d["expert_slot_0"]["intermediate_dim"] == 14336, d["expert_slot_0"]["intermediate_dim"])
check("expert_slot_0 tile_class=large",
      d["expert_slot_0"]["tile_class"] == "large", d["expert_slot_0"]["tile_class"])
check("expert_slot_0 specialization_policy=shape_static_v1",
      d["expert_slot_0"]["specialization_policy"] == "shape_static_v1",
      d["expert_slot_0"].get("specialization_policy"))

check("expert_slot_1 BLOCK_M=128",
      d["expert_slot_1"]["BLOCK_M"] == 128, d["expert_slot_1"]["BLOCK_M"])

# E2, E3: F=8192 → medium → BLOCK_M=64, BLOCK_N=128
check("expert_slot_2 BLOCK_M=64",
      d["expert_slot_2"]["BLOCK_M"] == 64, d["expert_slot_2"]["BLOCK_M"])
check("expert_slot_2 BLOCK_N=128",
      d["expert_slot_2"]["BLOCK_N"] == 128, d["expert_slot_2"]["BLOCK_N"])
check("expert_slot_2 intermediate_dim=8192",
      d["expert_slot_2"]["intermediate_dim"] == 8192, d["expert_slot_2"]["intermediate_dim"])
check("expert_slot_2 tile_class=medium",
      d["expert_slot_2"]["tile_class"] == "medium", d["expert_slot_2"]["tile_class"])

check("expert_slot_3 BLOCK_M=64",
      d["expert_slot_3"]["BLOCK_M"] == 64, d["expert_slot_3"]["BLOCK_M"])

# E4, E5: F=4096 → small → BLOCK_M=32, BLOCK_N=64
check("expert_slot_4 BLOCK_M=32",
      d["expert_slot_4"]["BLOCK_M"] == 32, d["expert_slot_4"]["BLOCK_M"])
check("expert_slot_4 BLOCK_N=64",
      d["expert_slot_4"]["BLOCK_N"] == 64, d["expert_slot_4"]["BLOCK_N"])
check("expert_slot_4 intermediate_dim=4096",
      d["expert_slot_4"]["intermediate_dim"] == 4096, d["expert_slot_4"]["intermediate_dim"])
check("expert_slot_4 tile_class=small",
      d["expert_slot_4"]["tile_class"] == "small", d["expert_slot_4"]["tile_class"])

check("expert_slot_5 BLOCK_M=32",
      d["expert_slot_5"]["BLOCK_M"] == 32, d["expert_slot_5"]["BLOCK_M"])

# E6, E7: F=2048 → tiny → BLOCK_M=16, BLOCK_N=32
check("expert_slot_6 BLOCK_M=16",
      d["expert_slot_6"]["BLOCK_M"] == 16, d["expert_slot_6"]["BLOCK_M"])
check("expert_slot_6 BLOCK_N=32",
      d["expert_slot_6"]["BLOCK_N"] == 32, d["expert_slot_6"]["BLOCK_N"])
check("expert_slot_6 intermediate_dim=2048",
      d["expert_slot_6"]["intermediate_dim"] == 2048, d["expert_slot_6"]["intermediate_dim"])
check("expert_slot_6 tile_class=tiny",
      d["expert_slot_6"]["tile_class"] == "tiny", d["expert_slot_6"]["tile_class"])
check("expert_slot_6 specialization_policy=shape_static_v1",
      d["expert_slot_6"]["specialization_policy"] == "shape_static_v1",
      d["expert_slot_6"].get("specialization_policy"))

check("expert_slot_7 BLOCK_M=16",
      d["expert_slot_7"]["BLOCK_M"] == 16, d["expert_slot_7"]["BLOCK_M"])

if failures:
    print(f"\nFAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)

print()
print("Wire test PASSED — compiler decisions drive correct per-expert dispatch")
EOF

# ---------------------------------------------------------------------------
# V2 wire test — token_pgo_v2 policy makes different decisions than V1
# ---------------------------------------------------------------------------
echo ""
echo "=== V2 wire test: token_pgo_v2 ==="
echo ""

V2_STATS="$REPO_ROOT/scratch/routing_stats_proportional.json"
V2_OUT="$REPO_ROOT/scratch/compiler_decisions_v2.json"
mkdir -p "$REPO_ROOT/scratch"

# Step 1 — generate routing stats for proportional distribution
python3 "$REPO_ROOT/scripts/generate_routing_stats.py" \
    --distribution proportional \
    --num-experts 8 \
    --total-tokens 512 \
    --num-samples 1000 \
    --output "$V2_STATS"

# Step 2 — emit V2 decisions using token-based thresholds
python3 "$REPO_ROOT/scripts/emit_compiler_decisions.py" \
    --input "$REPO_ROOT/mlir/stablehlo/heterogeneous_moe_layer.mlir" \
    --num-experts 1 \
    --routing-stats "$V2_STATS" \
    --output "$V2_OUT"

# Step 3 — assert V2-specific values
python3 - "$V2_OUT" <<'EOF'
import json, sys

d = json.load(open(sys.argv[1]))

failures = []

def check(name, condition, got=None):
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f" (got {got!r})" if got is not None else "")
        print(msg)
        failures.append(name)

# Proportional distribution: large experts get T≈128 → block_m_from_tokens(128)=64
# V1 would give BLOCK_M=128 for F=14336; V2 must give 64.
check("expert_slot_0 V2 BLOCK_M=64  (T≈128, not V1's 128)",
      d["expert_slot_0"]["BLOCK_M"] == 64, d["expert_slot_0"]["BLOCK_M"])
check("expert_slot_1 V2 BLOCK_M=64",
      d["expert_slot_1"]["BLOCK_M"] == 64, d["expert_slot_1"]["BLOCK_M"])

# Medium experts: T≈73 → block_m_from_tokens(73)=32; V1 gives 64
check("expert_slot_2 V2 BLOCK_M=32  (T≈73, not V1's 64)",
      d["expert_slot_2"]["BLOCK_M"] == 32, d["expert_slot_2"]["BLOCK_M"])
check("expert_slot_3 V2 BLOCK_M=32",
      d["expert_slot_3"]["BLOCK_M"] == 32, d["expert_slot_3"]["BLOCK_M"])

# Tiny experts: T≈18 → block_m_from_tokens(18)=16; V1 also gives 16 (coincidence)
check("expert_slot_6 V2 BLOCK_M=16  (T≈18)",
      d["expert_slot_6"]["BLOCK_M"] == 16, d["expert_slot_6"]["BLOCK_M"])
check("expert_slot_7 V2 BLOCK_M=16",
      d["expert_slot_7"]["BLOCK_M"] == 16, d["expert_slot_7"]["BLOCK_M"])

# Policy tag must be token_pgo_v2 (not shape_static_v1)
check("expert_slot_0 specialization_policy=token_pgo_v2",
      d["expert_slot_0"]["specialization_policy"] == "token_pgo_v2",
      d["expert_slot_0"].get("specialization_policy"))
check("expert_slot_4 specialization_policy=token_pgo_v2",
      d["expert_slot_4"]["specialization_policy"] == "token_pgo_v2",
      d["expert_slot_4"].get("specialization_policy"))

# expected_tokens field present
check("expert_slot_0 has expected_tokens",
      "expected_tokens" in d["expert_slot_0"])
check("expert_slot_0 expected_tokens ≈ 128  (proportional p50 for F=14336)",
      120 <= d["expert_slot_0"].get("expected_tokens", 0) <= 135,
      d["expert_slot_0"].get("expected_tokens"))

# Verify V2 ≠ V1 for large experts (core assertion: different policies, different decisions)
check("expert_slot_0 V2 BLOCK_M differs from V1 (64 ≠ 128)",
      d["expert_slot_0"]["BLOCK_M"] != 128, d["expert_slot_0"]["BLOCK_M"])

if failures:
    print(f"\nFAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)

print()
print("V2 wire test PASSED — token_pgo_v2 makes different decisions than shape_static_v1")
EOF

# Clean up scratch
rm -rf "$REPO_ROOT/scratch"
