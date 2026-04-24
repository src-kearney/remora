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
