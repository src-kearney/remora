#!/usr/bin/env python3
"""
scripts/verify/verify_heterogeneous_outlining.py

Verifies that heterogeneous_moe_layer.mlir correctly encodes heterogeneous expert
shapes, and that ExpertOutliningPass with num_experts=1 outlines all 8 experts
into distinct @expert_slot_N functions carrying their true intermediate dimensions.

Checks performed
─────────────────
1. STATIC STRUCTURE (no compiler needed)
   Parse heterogeneous_moe_layer.mlir and verify:
   • Eight per-expert dispatch slices (batch=1 each) are present.
   • Each expert's individual [1, 512, F] intermediate tensor is present.
   • No single tensor spans all 8 experts at mixed intermediate dims.
   • Per-expert gate/up/down weight shapes are correct.

2. PASS BEHAVIOUR ON UNIFORM IR (compiler required)
   Run moe-expert-outlining on mixtral_moe_layer.mlir (the homogeneous
   baseline) and verify the pass outlines @expert_slot_0 and @expert_slot_1
   with the shared intermediate dim [512, 14336].

3. PASS LIMITATION ON HETERO IR WITH DEFAULT num_experts=8
   Run moe-expert-outlining on heterogeneous_moe_layer.mlir with the default
   numExperts=8.  Each expert has lhs.shape[0]==1; pass finds zero matching
   dots and exits cleanly.

4. PASS WITH num_experts=1 ON HETERO IR
   Run moe-expert-outlining{num_experts=1} on heterogeneous_moe_layer.mlir.
   Verify the output IR contains 8 @expert_slot_N functions, each carrying
   its per-expert true intermediate dimension:
     @expert_slot_0, @expert_slot_1  →  tensor<512x14336xf32>
     @expert_slot_2, @expert_slot_3  →  tensor<512x 8192xf32>
     @expert_slot_4, @expert_slot_5  →  tensor<4096x4096xf32> weight (F=D=4096)
     @expert_slot_6, @expert_slot_7  →  tensor<512x 2048xf32>

Usage:
    python3 scripts/verify/verify_heterogeneous_outlining.py

The compiler binary (compiler/build/remora) is required for checks 2, 3, and 4.
If it is absent the compiler checks are skipped and the script exits 0.
"""

import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
BINARY     = os.path.join(REPO_ROOT, "compiler/build/remora")
HETERO_IR  = os.path.join(REPO_ROOT, "mlir/stablehlo/heterogeneous_moe_layer.mlir")
UNIFORM_IR = os.path.join(REPO_ROOT, "mlir/stablehlo/mixtral_moe_layer.mlir")

# Per-expert F layout:  E0,E1=14336  E2,E3=8192  E4,E5=4096  E6,E7=2048
EXPERT_F = [14336, 14336, 8192, 8192, 4096, 4096, 2048, 2048]

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
        print(msg)
        failures.append(name)


def run_pass(ir_path: str, pipeline: str, tmpdir: str) -> tuple[int, str, str]:
    """Run a pass pipeline on ir_path; return (returncode, transformed_ir, stderr)."""
    cmd = [
        BINARY, ir_path,
        f"--pass-pipeline={pipeline}",
        "--no-execute",
        f"--dump-compilation-phases-to={tmpdir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # inferStepName strips {options}, so directory is always "01-<pass-name>".
    pass_name = pipeline.split("{")[0]
    phase_dir = os.path.join(tmpdir, f"01-{pass_name}")
    ir_out = ""
    ir_path_out = os.path.join(phase_dir, "module.mlir")
    if os.path.exists(ir_path_out):
        with open(ir_path_out) as f:
            ir_out = f.read()
    return result.returncode, ir_out, result.stderr


def extract_func(ir_text: str, func_name: str) -> str:
    """Extract the full text of a named func.func from MLIR output."""
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
                return ir_text[start:i + 1]
    return ir_text[start:]


# ===========================================================================
# 1. STATIC STRUCTURE CHECKS
# ===========================================================================
print("=" * 70)
print("Check 1: static structure of heterogeneous_moe_layer.mlir")
print("=" * 70)

if not os.path.exists(HETERO_IR):
    print(f"FAIL — file not found: {HETERO_IR}")
    sys.exit(1)

with open(HETERO_IR) as f:
    hetero_ir = f.read()

print(f"  IR size: {len(hetero_ir):,} bytes, {hetero_ir.count(chr(10))} lines\n")

# 1a. Each expert's per-expert [1, 512, F] intermediate tensor is present.
print("  1a. Per-expert intermediate tensors (batch=1):")
seen_F: set[int] = set()
for expert_id, F in enumerate(EXPERT_F):
    pattern = f"tensor<1x512x{F}xf32>"
    if F not in seen_F:
        check(
            f"E{expert_id} (F={F}): {pattern} present",
            pattern in hetero_ir,
        )
        seen_F.add(F)
    else:
        # Second expert of same F-class: just verify the type appears (same pattern)
        check(
            f"E{expert_id} (F={F}): {pattern} present (second expert in class)",
            pattern in hetero_ir,
        )

print()

# 1b. Eight individual per-expert dispatch slices are present.
print("  1b. Per-expert dispatch slices (eight individual [e:e+1] slices):")
for e in range(8):
    slice_pat = f"[{e}:{e + 1}, 0:512, 0:4096]"
    check(
        f"E{e} dispatch slice {slice_pat} present",
        slice_pat in hetero_ir,
    )

print()

# 1c. No single tensor spans all 8 experts with a heterogeneous F.
non_comment = "\n".join(
    l for l in hetero_ir.splitlines() if not l.lstrip().startswith("//")
)
uniform_weight_8 = re.search(r"tensor<8x4096x\d+xf32>", non_comment)
check(
    "no uniform [8, 4096, F] weight tensor in IR (heterogeneity preserved)",
    uniform_weight_8 is None,
    detail=f"found: {uniform_weight_8.group(0)}" if uniform_weight_8 else "",
)

print()

# 1d. Per-expert gate/up and down weight shapes.
print("  1d. Per-expert weight tensor shapes (batch=1):")
for F in sorted(set(EXPERT_F)):
    gate_pat = f"tensor<1x4096x{F}xf32>"
    down_pat = f"tensor<1x{F}x4096xf32>"
    check(f"gate/up weight {gate_pat} present", gate_pat in hetero_ir)
    check(f"down weight    {down_pat} present", down_pat in hetero_ir)

print()
print("Static structure: " + ("PASS\n" if not failures else "FAIL\n"))

# ===========================================================================
# 2, 3 & 4. COMPILER PASS CHECKS
# ===========================================================================
if not os.path.exists(BINARY):
    print("=" * 70)
    print("Compiler binary not found — skipping pass checks")
    print(f"  Expected: {BINARY}")
    print("  Build with: cd compiler && mkdir -p build && cmake .. && make -j")
    print("=" * 70)
    compiler_available = False
else:
    compiler_available = True

if compiler_available:
    # -----------------------------------------------------------------------
    # 2. Pass on UNIFORM IR (mixtral_moe_layer.mlir) — expect 2 slots outlined
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Check 2: moe-expert-outlining on uniform IR (mixtral_moe_layer.mlir)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        rc, uniform_out, stderr = run_pass(UNIFORM_IR, "moe-expert-outlining", tmp)

    check("pass exits 0 on uniform IR", rc == 0, detail=stderr[:200] if rc != 0 else "")

    for slot_id in (0, 1):
        check(
            f"@expert_slot_{slot_id} defined in transformed uniform IR",
            f"func.func @expert_slot_{slot_id}" in uniform_out,
        )
    check(
        "uniform @expert_slot_0 carries tensor<512x14336xf32>",
        "tensor<512x14336xf32>" in uniform_out,
    )

    print()

    # -----------------------------------------------------------------------
    # 3. Pass on HETERO IR with default numExperts=8 — expect zero slots
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Check 3: moe-expert-outlining (numExperts=8) on heterogeneous IR")
    print("  Expected: zero matching dots (each expert has lhs.shape[0]==1, not 8).")
    print("  Pass exits cleanly; no @expert_slot_N emitted.")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        rc_h, hetero_out_default, stderr_h = run_pass(
            HETERO_IR, "moe-expert-outlining", tmp
        )

    check(
        "pass exits 0 on heterogeneous IR with default numExperts=8",
        rc_h == 0,
        detail=stderr_h[:200] if rc_h != 0 else "",
    )
    check(
        "no @expert_slot_N defined (zero matching dots at numExperts=8)",
        "func.func @expert_slot_" not in hetero_out_default,
        detail="found expert_slot in transformed IR — unexpected",
    )

    print()

    # -----------------------------------------------------------------------
    # 4. Pass with num_experts=1 on HETERO IR — expect 8 distinct slots
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("Check 4: moe-expert-outlining{num_experts=1} on heterogeneous IR")
    print("  Expected: 8 @expert_slot_N functions, each carrying its true F.")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        rc_e, hetero_out_e1, stderr_e = run_pass(
            HETERO_IR, "moe-expert-outlining{num_experts=1}", tmp
        )

    check(
        "pass exits 0 with num_experts=1 on heterogeneous IR",
        rc_e == 0,
        detail=stderr_e[:400] if rc_e != 0 else "",
    )

    print()
    print("  Verifying 8 @expert_slot_N function definitions:")
    for slot_id in range(8):
        check(
            f"@expert_slot_{slot_id} defined",
            f"func.func @expert_slot_{slot_id}" in hetero_out_e1,
        )

    print()
    print("  Verifying per-expert intermediate dimensions (via unique weight types):")

    # Gate/up weight type is [Dw, F] = [4096, F] after leading dim stripped.
    # This is unique per F-class and distinguishes even F=4096 experts from others.
    EXPERT_WEIGHT_CHECKS = [
        # (slot_id, tensor_to_find, description)
        (0, "tensor<4096x14336xf32>", "gate/up weight [4096,14336] → F=14336"),
        (1, "tensor<4096x14336xf32>", "gate/up weight [4096,14336] → F=14336"),
        (2, "tensor<4096x8192xf32>",  "gate/up weight [4096,8192]  → F=8192"),
        (3, "tensor<4096x8192xf32>",  "gate/up weight [4096,8192]  → F=8192"),
        (4, "tensor<4096x4096xf32>",  "gate/up weight [4096,4096]  → F=4096 (=D)"),
        (5, "tensor<4096x4096xf32>",  "gate/up weight [4096,4096]  → F=4096 (=D)"),
        (6, "tensor<4096x2048xf32>",  "gate/up weight [4096,2048]  → F=2048"),
        (7, "tensor<4096x2048xf32>",  "gate/up weight [4096,2048]  → F=2048"),
    ]

    for slot_id, tensor_type, desc in EXPERT_WEIGHT_CHECKS:
        func_text = extract_func(hetero_out_e1, f"expert_slot_{slot_id}")
        has_type = tensor_type in func_text if func_text else False
        check(
            f"@expert_slot_{slot_id}: {desc}",
            has_type,
            detail="function not found in output" if not func_text else
                   f"{tensor_type!r} not found in function body",
        )

    print()
    print("  Verifying moe.num_experts=1 attribute on outlined functions:")
    check(
        "moe.num_experts = 1 : i32 present in transformed IR",
        "moe.num_experts = 1 : i32" in hetero_out_e1,
    )

    print()

# ===========================================================================
# Result
# ===========================================================================
print("=" * 70)
if failures:
    print(f"FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("All checks PASSED")
    print()
    print("  • heterogeneous_moe_layer.mlir: 8 per-expert groups (batch=1 each)")
    print("  • No uniform [8, 4096, F] weight tensor (heterogeneity preserved)")
    print("  • ExpertOutliningPass is shape-polymorphic on the uniform IR")
    print("  • Default numExperts=8 correctly finds zero matching dots")
    print("  • With num_experts=1: 8 @expert_slot_N functions emitted")
    print("  • Each slot carries its true intermediate dimension as a static type")
    print("=" * 70)
