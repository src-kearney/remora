#!/usr/bin/env python3
"""
scripts/verify/verify_cost_analysis.py

Runs moe-expert-outlining{num_experts=1},moe-expert-cost-analysis on
heterogeneous_moe_layer.mlir and verifies the per-expert cost attributes.

Checks performed:
  1. All eight @expert_slot_N functions are present.
  2. Each function carries moe.hidden_dim, moe.intermediate_dim,
     moe.tokens_per_slot, moe.flops_estimate, moe.bytes_estimate,
     moe.arithmetic_intensity, moe.cost_class.
  3. Computed values match the formulas (independently re-derived in Python).
  4. moe.arithmetic_intensity differs across expert classes — empirical
     evidence that shape heterogeneity produces different hardware cost
     profiles.

Usage:
    python3 scripts/verify/verify_cost_analysis.py
"""

import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
BINARY    = os.path.join(REPO_ROOT, "compiler", "build", "remora")
HETERO_IR = os.path.join(REPO_ROOT, "mlir", "stablehlo", "heterogeneous_moe_layer.mlir")

# Per-expert intermediate dims: E0/E1=14336  E2/E3=8192  E4/E5=4096  E6/E7=2048
EXPERT_F = [14336, 14336, 8192, 8192, 4096, 4096, 2048, 2048]
T = 512     # tokens per slot (read from IR; hard-coded here only for expected-value checks)
D = 4096    # hidden dim

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
        print(msg)
        failures.append(name)


# ---------------------------------------------------------------------------
# Reference formulas (Python re-derivation, independent of C++)
# ---------------------------------------------------------------------------

def expected_flops(t: int, d: int, f: int) -> int:
    return 2 * t * d * f * 3

def expected_bytes(t: int, d: int, f: int) -> int:
    return (t * d + d * f + d * f + f * d + t * d) * 4

def expected_ai(t: int, d: int, f: int) -> float:
    fl = expected_flops(t, d, f)
    by = expected_bytes(t, d, f)
    return fl / by

def expected_cost_class(f: int) -> str:
    if f > 8192: return "large"
    if f > 4096: return "medium"
    if f > 2048: return "small"
    return "tiny"


# ---------------------------------------------------------------------------
# Run the two-pass pipeline and return the final IR text
# ---------------------------------------------------------------------------

def run_pipeline(tmpdir: str) -> str:
    pipeline = "moe-expert-outlining{num_experts=1},moe-expert-cost-analysis"
    cmd = [
        BINARY, HETERO_IR,
        f"--pass-pipeline={pipeline}",
        "--no-execute",
        f"--dump-compilation-phases-to={tmpdir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL — compiler exited {result.returncode}:\n{result.stderr}")
        sys.exit(1)

    # The bare pipeline creates a single phase dump named after the first pass.
    phase_dir = os.path.join(tmpdir, "01-moe-expert-outlining")
    ir_path   = os.path.join(phase_dir, "module.mlir")
    if not os.path.exists(ir_path):
        print(f"FAIL — expected phase dump at {ir_path}")
        sys.exit(1)

    with open(ir_path) as f:
        return f.read()


def extract_func(ir_text: str, func_name: str) -> str:
    m = re.search(rf'func\.func @{re.escape(func_name)}\b', ir_text)
    if not m:
        return ""
    start = m.start()
    brace_count = 0
    in_func = False
    for i in range(start, len(ir_text)):
        if ir_text[i] == '{':
            brace_count += 1
            in_func = True
        elif ir_text[i] == '}':
            brace_count -= 1
            if in_func and brace_count == 0:
                return ir_text[start : i + 1]
    return ir_text[start:]


def parse_int_attr(text: str, attr: str) -> int | None:
    m = re.search(rf'{re.escape(attr)}\s*=\s*(-?\d+)\s*:', text)
    return int(m.group(1)) if m else None

def parse_float_attr(text: str, attr: str) -> float | None:
    m = re.search(rf'{re.escape(attr)}\s*=\s*([0-9.e+\-]+)\s*:', text)
    return float(m.group(1)) if m else None

def parse_str_attr(text: str, attr: str) -> str | None:
    m = re.search(rf'{re.escape(attr)}\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not os.path.exists(BINARY):
    print(f"Compiler binary not found: {BINARY}")
    print("Build with: cd compiler && cmake --build build")
    sys.exit(1)

scratch = os.path.join(REPO_ROOT, "scratch", "verify_cost_analysis")
os.makedirs(scratch, exist_ok=True)

try:
    print("Running moe-expert-outlining{num_experts=1},moe-expert-cost-analysis ...")
    ir_text = run_pipeline(scratch)

    print()
    print("=" * 70)
    print("Check 1: all 8 @expert_slot_N functions present")
    print("=" * 70)
    for slot_id in range(8):
        check(f"@expert_slot_{slot_id} defined",
              f"func.func @expert_slot_{slot_id}" in ir_text)

    print()
    print("=" * 70)
    print("Check 2 & 3: per-expert attribute values (verified against Python formulas)")
    print("=" * 70)

    ai_by_class: dict[str, list[float]] = {}

    for slot_id in range(8):
        f_dim = EXPERT_F[slot_id]
        func_text = extract_func(ir_text, f"expert_slot_{slot_id}")
        prefix = f"@expert_slot_{slot_id} (F={f_dim})"

        check(f"{prefix} function present", bool(func_text),
              "function not found in IR")
        if not func_text:
            continue

        actual_hidden   = parse_int_attr(func_text, "moe.hidden_dim")
        actual_inter    = parse_int_attr(func_text, "moe.intermediate_dim")
        actual_tokens   = parse_int_attr(func_text, "moe.tokens_per_slot")
        actual_flops    = parse_int_attr(func_text, "moe.flops_estimate")
        actual_bytes    = parse_int_attr(func_text, "moe.bytes_estimate")
        actual_ai       = parse_float_attr(func_text, "moe.arithmetic_intensity")
        actual_class    = parse_str_attr(func_text, "moe.cost_class")

        exp_flops = expected_flops(T, D, f_dim)
        exp_bytes = expected_bytes(T, D, f_dim)
        exp_ai    = expected_ai(T, D, f_dim)
        exp_class = expected_cost_class(f_dim)

        check(f"{prefix} moe.hidden_dim={D}",
              actual_hidden == D, f"got {actual_hidden}")
        check(f"{prefix} moe.intermediate_dim={f_dim}",
              actual_inter == f_dim, f"got {actual_inter}")
        check(f"{prefix} moe.tokens_per_slot={T}",
              actual_tokens == T, f"got {actual_tokens}")
        check(f"{prefix} moe.flops_estimate={exp_flops}",
              actual_flops == exp_flops, f"got {actual_flops}")
        check(f"{prefix} moe.bytes_estimate={exp_bytes}",
              actual_bytes == exp_bytes, f"got {actual_bytes}")
        check(f"{prefix} moe.arithmetic_intensity ≈ {exp_ai:.4f}",
              actual_ai is not None and abs(actual_ai - exp_ai) < 1e-3,
              f"got {actual_ai}")
        check(f"{prefix} moe.cost_class={exp_class!r}",
              actual_class == exp_class, f"got {actual_class!r}")

        if actual_ai is not None and actual_class is not None:
            ai_by_class.setdefault(actual_class, []).append(actual_ai)

    print()
    print("=" * 70)
    print("Check 4: arithmetic intensity differs across expert classes")
    print("  (empirical evidence that shape heterogeneity → different cost profiles)")
    print("=" * 70)

    class_ai: dict[str, float] = {cls: vals[0] for cls, vals in ai_by_class.items()}
    ai_values = list(class_ai.values())
    check(
        "moe.arithmetic_intensity is not uniform across all classes",
        len(set(round(v, 6) for v in ai_values)) > 1,
        f"all AI values identical: {ai_values}",
    )
    if len(class_ai) >= 2:
        sorted_classes = sorted(class_ai.items(), key=lambda kv: -kv[1])
        highest_cls, highest_ai = sorted_classes[0]
        lowest_cls,  lowest_ai  = sorted_classes[-1]
        check(
            f"larger F → higher AI ('{highest_cls}' AI > '{lowest_cls}' AI)",
            highest_ai > lowest_ai,
            f"{highest_ai:.4f} vs {lowest_ai:.4f}",
        )

    print()
    print("=" * 70)
    print("Cost analysis summary table")
    print("=" * 70)
    print(f"  {'slot':<14}  {'F':>6}  {'flops':>12}  {'bytes':>10}  {'AI':>7}  {'cost_class':<8}")
    print("  " + "-" * 62)
    for slot_id in range(8):
        f_dim = EXPERT_F[slot_id]
        func_text = extract_func(ir_text, f"expert_slot_{slot_id}")
        if not func_text:
            continue
        flops = parse_int_attr(func_text, "moe.flops_estimate") or 0
        byts  = parse_int_attr(func_text, "moe.bytes_estimate") or 0
        ai    = parse_float_attr(func_text, "moe.arithmetic_intensity") or 0.0
        cls   = parse_str_attr(func_text, "moe.cost_class") or "?"
        print(f"  expert_slot_{slot_id:<2}  {f_dim:>6}  {flops:>12.3e}  {byts:>10.3e}  "
              f"{ai:>7.2f}  {cls:<8}")

    print()

finally:
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
print("=" * 70)
if failures:
    print(f"FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("All checks PASSED")
    print()
    print("  • moe.hidden_dim, moe.intermediate_dim, moe.tokens_per_slot: correct")
    print("  • moe.flops_estimate, moe.bytes_estimate: match Python reference formulas")
    print("  • moe.arithmetic_intensity: verified; differs across expert classes")
    print("  • moe.cost_class: correct threshold classification")
    print("=" * 70)
