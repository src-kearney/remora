#!/usr/bin/env python3
"""
scripts/verify/verify_specialization.py

Runs the full three-pass pipeline on heterogeneous_moe_layer.mlir and
verifies that moe-expert-specialization emits correct per-expert tile
configuration decisions.

Checks performed:
  1. All 8 @expert_slot_N functions are present after the full pipeline.
  2. Each function carries moe.tile_class, moe.BLOCK_M, moe.BLOCK_N,
     moe.specialization_policy.
  3. Decisions are correct per expert class.
  4. moe.specialization_policy = "shape_static_v1" on every expert.
  5. Decisions differ across expert classes.

Usage:
    python3 scripts/verify/verify_specialization.py
"""

import os
import re
import subprocess
import shutil
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
BINARY    = os.path.join(REPO_ROOT, "compiler", "build", "remora")
HETERO_IR = os.path.join(REPO_ROOT, "mlir", "stablehlo", "heterogeneous_moe_layer.mlir")

EXPERT_F = [14336, 14336, 8192, 8192, 4096, 4096, 2048, 2048]

PIPELINE = ("moe-expert-outlining{num_experts=1},"
            "moe-expert-cost-analysis,"
            "moe-expert-specialization")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
        print(msg)
        failures.append(name)


# ---------------------------------------------------------------------------
# Expected decisions (Python reference, independent of C++)
# ---------------------------------------------------------------------------

def expected_block_m(cost_class: str) -> int:
    return {"large": 128, "medium": 64, "small": 32, "tiny": 16}[cost_class]

def expected_block_n(F: int) -> int:
    if F >= 8192: return 128
    if F >= 4096: return  64
    return 32

def expected_cost_class(F: int) -> str:
    if F > 8192: return "large"
    if F > 4096: return "medium"
    if F > 2048: return "small"
    return "tiny"


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

def run_pipeline(scratch: str) -> str:
    cmd = [
        BINARY, HETERO_IR,
        f"--pass-pipeline={PIPELINE}",
        "--no-execute",
        f"--dump-compilation-phases-to={scratch}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL — compiler exited {result.returncode}:\n{result.stderr}")
        sys.exit(1)

    # Bare pipeline → single phase dump named after first pass.
    ir_path = os.path.join(scratch, "01-moe-expert-outlining", "module.mlir")
    if not os.path.exists(ir_path):
        print(f"FAIL — phase dump not found at {ir_path}")
        sys.exit(1)

    with open(ir_path) as f:
        return f.read()


def extract_func(ir_text: str, func_name: str) -> str:
    m = re.search(rf'func\.func @{re.escape(func_name)}\b', ir_text)
    if not m:
        return ""
    start = m.start()
    depth = 0
    in_func = False
    for i in range(start, len(ir_text)):
        if ir_text[i] == '{':
            depth += 1
            in_func = True
        elif ir_text[i] == '}':
            depth -= 1
            if in_func and depth == 0:
                return ir_text[start : i + 1]
    return ir_text[start:]


def parse_int_attr(text: str, attr: str) -> int | None:
    m = re.search(rf'{re.escape(attr)}\s*=\s*(-?\d+)\s*:', text)
    return int(m.group(1)) if m else None

def parse_str_attr(text: str, attr: str) -> str | None:
    m = re.search(rf'{re.escape(attr)}\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not os.path.exists(BINARY):
    print(f"Compiler binary not found: {BINARY}")
    sys.exit(1)

scratch = os.path.join(REPO_ROOT, "scratch", "verify_specialization")
os.makedirs(scratch, exist_ok=True)

try:
    print(f"Running full three-pass pipeline:")
    print(f"  moe-expert-outlining → moe-expert-cost-analysis → moe-expert-specialization")
    print()
    ir_text = run_pipeline(scratch)

    # -----------------------------------------------------------------------
    # Check 1: all 8 functions present
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Check 1: all 8 @expert_slot_N functions present")
    print("=" * 70)
    for slot_id in range(8):
        check(f"@expert_slot_{slot_id} defined",
              f"func.func @expert_slot_{slot_id}" in ir_text)

    # -----------------------------------------------------------------------
    # Check 2 & 3: correct decisions per expert
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Check 2 & 3: specialization decisions correct per expert class")
    print("=" * 70)

    block_m_values: list[int] = []

    for slot_id in range(8):
        F = EXPERT_F[slot_id]
        func_text = extract_func(ir_text, f"expert_slot_{slot_id}")
        prefix = f"@expert_slot_{slot_id} (F={F})"

        check(f"{prefix} function present", bool(func_text))
        if not func_text:
            continue

        actual_tile_class = parse_str_attr(func_text, "moe.tile_class")
        actual_block_m    = parse_int_attr(func_text, "moe.BLOCK_M")
        actual_block_n    = parse_int_attr(func_text, "moe.BLOCK_N")
        actual_policy     = parse_str_attr(func_text, "moe.specialization_policy")

        exp_cost_class = expected_cost_class(F)
        exp_block_m    = expected_block_m(exp_cost_class)
        exp_block_n    = expected_block_n(F)

        check(f"{prefix} moe.tile_class={exp_cost_class!r}",
              actual_tile_class == exp_cost_class, f"got {actual_tile_class!r}")
        check(f"{prefix} moe.BLOCK_M={exp_block_m}",
              actual_block_m == exp_block_m, f"got {actual_block_m}")
        check(f"{prefix} moe.BLOCK_N={exp_block_n}",
              actual_block_n == exp_block_n, f"got {actual_block_n}")
        check(f"{prefix} moe.specialization_policy='shape_static_v1'",
              actual_policy == "shape_static_v1", f"got {actual_policy!r}")

        if actual_block_m is not None:
            block_m_values.append(actual_block_m)

    # -----------------------------------------------------------------------
    # Check 4: specialization_policy present on all experts
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Check 4: moe.specialization_policy = 'shape_static_v1' on all experts")
    print("=" * 70)
    policy_count = ir_text.count('moe.specialization_policy = "shape_static_v1"')
    check(
        f"policy attribute present on all 8 experts (found {policy_count})",
        policy_count == 8,
        f"expected 8, got {policy_count}",
    )

    # -----------------------------------------------------------------------
    # Check 5: decisions differ across expert classes
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Check 5: decisions differ across expert classes")
    print("=" * 70)
    unique_block_m = set(block_m_values)
    check(
        f"BLOCK_M takes multiple distinct values (found {sorted(unique_block_m)})",
        len(unique_block_m) > 1,
        f"all experts got the same BLOCK_M",
    )

    # -----------------------------------------------------------------------
    # Decision table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Full pass pipeline: moe-expert-outlining → moe-expert-cost-analysis"
          " → moe-expert-specialization")
    print()
    print(f"  {'slot':<14}  {'F':>6}  {'cost_class':<8}  "
          f"{'BLOCK_M':>7}  {'BLOCK_N':>7}  {'policy'}")
    print("  " + "-" * 66)
    for slot_id in range(8):
        F = EXPERT_F[slot_id]
        func_text = extract_func(ir_text, f"expert_slot_{slot_id}")
        if not func_text:
            continue
        tile_cls = parse_str_attr(func_text, "moe.tile_class") or "?"
        block_m  = parse_int_attr(func_text, "moe.BLOCK_M")
        block_n  = parse_int_attr(func_text, "moe.BLOCK_N")
        policy   = parse_str_attr(func_text, "moe.specialization_policy") or "?"
        bm_str   = str(block_m) if block_m is not None else "?"
        bn_str   = str(block_n) if block_n is not None else "?"
        print(f"  expert_slot_{slot_id:<2}  {F:>6}  {tile_cls:<8}  "
              f"{bm_str:>7}  {bn_str:>7}  {policy}")

    print()
    print(f"  All 8 experts specialized. Policy: shape_static_v1.")
    print(f"  Decisions differ across expert classes: "
          + ("✓" if len(unique_block_m) > 1 else "✗"))

finally:
    shutil.rmtree(scratch, ignore_errors=True)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
print()
print("=" * 70)
if failures:
    print(f"FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("All checks PASSED")
    print()
    print("  • moe.tile_class: mirrors cost_class (shape_static_v1 policy)")
    print("  • moe.BLOCK_M: 128/64/32/16 per cost class")
    print("  • moe.BLOCK_N: 128/64/32 per intermediate dim")
    print("  • moe.specialization_policy: 'shape_static_v1' on all 8 experts")
    print("  • Decisions differ across classes: confirmed")
    print("=" * 70)
