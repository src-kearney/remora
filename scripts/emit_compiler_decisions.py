#!/usr/bin/env python3
"""
scripts/emit_compiler_decisions.py

Runs the full Remora pass pipeline (outlining + cost-analysis + specialization)
on a StableHLO/MLIR input, parses per-expert attributes from the phase-dump IR,
and writes compiler_decisions.json.

Two specialization policies are supported:

  shape_static_v1  (default)
    BLOCK_M selected by intermediate dim F.  Matches the C++ specialization pass.
    F > 8192 → 128, F > 4096 → 64, F > 2048 → 32, else 16.

  token_pgo_v2  (--routing-stats path/to/routing_stats.json)
    BLOCK_M selected by expected per-expert token count T (p50 from routing stats).
    T ≥ 256 → 128, T ≥ 128 → 64, T ≥ 64 → 32, else 16.
    Oracle sweep (sweep_heterogeneous.py --extended) shows T is the dominant signal.

Usage:
    # V1 (shape-static, default)
    python3 scripts/emit_compiler_decisions.py \\
        --input  mlir/stablehlo/heterogeneous_moe_layer.mlir \\
        --num-experts 1 \\
        --output compiler_decisions.json

    # V2 (PGO, token-based)
    python3 scripts/emit_compiler_decisions.py \\
        --input  mlir/stablehlo/heterogeneous_moe_layer.mlir \\
        --num-experts 1 \\
        --routing-stats routing_stats_proportional.json \\
        --output compiler_decisions_v2.json
"""

import argparse
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
BINARY    = os.path.join(REPO_ROOT, "compiler", "build", "remora")


# ---------------------------------------------------------------------------
# Run the pass pipeline and return the final IR text
# ---------------------------------------------------------------------------

def run_pipeline(input_mlir: str, num_experts: int) -> str:
    """
    Run moe-expert-outlining + moe-expert-cost-analysis on `input_mlir`.

    The two passes are given as a bare comma-separated pipeline.
    splitTopLevelPipeline treats this as a single step (no enclosing parens),
    running both passes through one PassManager and dumping one phase snapshot
    named after the first pass: 01-moe-expert-outlining/module.mlir.

    Returns the final IR string (contains attributes from both passes).
    Raises SystemExit on compiler error.
    """
    if not os.path.exists(BINARY):
        sys.exit(f"error: compiler binary not found at {BINARY}\n"
                 f"       Build with: cd compiler && cmake --build build")

    pipeline = (f"moe-expert-outlining{{num_experts={num_experts}}},"
                f"moe-expert-cost-analysis,"
                f"moe-expert-specialization")

    scratch = os.path.join(REPO_ROOT, "scratch", "emit_decisions")
    os.makedirs(scratch, exist_ok=True)

    try:
        cmd = [
            BINARY, input_mlir,
            f"--pass-pipeline={pipeline}",
            "--no-execute",
            f"--dump-compilation-phases-to={scratch}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            sys.exit(
                f"error: compiler exited with status {result.returncode}\n"
                f"{result.stderr}"
            )

        # Bare pipeline → single phase dir named after the first pass.
        phase_dir = os.path.join(scratch, "01-moe-expert-outlining")
        ir_path   = os.path.join(phase_dir, "module.mlir")

        if not os.path.exists(ir_path):
            sys.exit(
                f"error: expected phase dump at {ir_path} but file not found\n"
                f"       stderr: {result.stderr}"
            )

        with open(ir_path) as f:
            return f.read()

    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Attribute regex patterns
# ---------------------------------------------------------------------------
# Matches a full func.func @expert_slot_N declaration including its
# attributes block (single-line in MLIR printed form).
_RE_FUNC_DECL = re.compile(
    r'func\.func\s+@(expert_slot_\d+)\b[^{]*attributes\s*\{([^}]+)\}'
)

_RE_SLOT_ID    = re.compile(r'moe\.slot_id\s*=\s*(\d+)\s*:')
_RE_INTER_DIM  = re.compile(r'moe\.intermediate_dim\s*=\s*(\d+)\s*:')
_RE_TILE_CLASS = re.compile(r'moe\.tile_class\s*=\s*"([^"]+)"')
_RE_BLOCK_M    = re.compile(r'moe\.BLOCK_M\s*=\s*(\d+)\s*:')
_RE_BLOCK_N    = re.compile(r'moe\.BLOCK_N\s*=\s*(\d+)\s*:')
_RE_FLOPS      = re.compile(r'moe\.flops_estimate\s*=\s*(\d+)\s*:')
_RE_BYTES      = re.compile(r'moe\.bytes_estimate\s*=\s*(\d+)\s*:')
_RE_AI         = re.compile(r'moe\.arithmetic_intensity\s*=\s*([0-9.e+\-]+)\s*:')
_RE_COST_CLASS = re.compile(r'moe\.cost_class\s*=\s*"([^"]+)"')
_RE_POLICY     = re.compile(r'moe\.specialization_policy\s*=\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# V2 policy: token-count-based BLOCK_M selection
# ---------------------------------------------------------------------------

def block_m_from_tokens(t: int) -> int:
    """Select BLOCK_M from per-expert expected token count.

    Thresholds derived from oracle sweep (sweep_heterogeneous.py --extended):
    optimal BLOCK_M tracks token count T, not intermediate dim F.
    """
    if t >= 256: return 128
    if t >= 128: return 64
    if t >= 64:  return 32
    return 16


def apply_pgo_overrides(
    decisions: dict[str, dict],
    routing_stats: dict,
) -> dict[str, dict]:
    """Override BLOCK_M in ``decisions`` using per-expert token counts.

    routing_stats must contain:
      ``expert_token_counts``: dict mapping "expert_N" → {mean, p50, p95}.

    Uses p50 (median) as the representative T.  Sets
    ``specialization_policy = "token_pgo_v2"`` and adds ``expected_tokens``
    to each entry.  BLOCK_N is fixed at 64 (matches oracle BN).
    Entries with no corresponding routing stat are left unchanged (V1 fallback).
    """
    counts = routing_stats.get("expert_token_counts", {})
    for slot_key, cfg in decisions.items():
        slot_id  = int(slot_key.split("_")[-1])   # "expert_slot_N" → N
        stat_key = f"expert_{slot_id}"
        if stat_key not in counts:
            print(f"warning: no routing stats for {stat_key}, "
                  f"keeping V1 decision for {slot_key}", file=sys.stderr)
            continue
        p50 = int(counts[stat_key]["p50"])
        cfg["BLOCK_M"]               = block_m_from_tokens(p50)
        cfg["BLOCK_N"]               = 64   # fixed; matches oracle BN=64
        cfg["expected_tokens"]       = p50
        cfg["specialization_policy"] = "token_pgo_v2"
    return decisions


def parse_decisions(ir_text: str) -> dict[str, dict]:
    """
    Parse all @expert_slot_N function declarations from the IR text.
    Returns a dict keyed by "expert_slot_N" with per-expert decision dicts.
    """
    decisions: dict[str, dict] = {}

    for m in _RE_FUNC_DECL.finditer(ir_text):
        func_name = m.group(1)
        attrs     = m.group(2)

        slot_m      = _RE_SLOT_ID.search(attrs)
        inter_m     = _RE_INTER_DIM.search(attrs)
        tile_m      = _RE_TILE_CLASS.search(attrs)
        block_m_m   = _RE_BLOCK_M.search(attrs)
        block_n_m   = _RE_BLOCK_N.search(attrs)
        flops_m     = _RE_FLOPS.search(attrs)
        bytes_m     = _RE_BYTES.search(attrs)
        ai_m        = _RE_AI.search(attrs)
        cost_cls_m  = _RE_COST_CLASS.search(attrs)
        policy_m    = _RE_POLICY.search(attrs)

        required = [
            ("moe.slot_id",          slot_m),
            ("moe.intermediate_dim", inter_m),
            ("moe.tile_class",       tile_m),
            ("moe.BLOCK_M",          block_m_m),
        ]
        missing = [name for name, match in required if match is None]
        if missing:
            print(f"warning: @{func_name} is missing required attributes: {missing}",
                  file=sys.stderr)
            continue

        entry: dict = {
            "tile_class":       tile_m.group(1),
            "BLOCK_M":          int(block_m_m.group(1)),
            "intermediate_dim": int(inter_m.group(1)),
        }

        # Specialization attributes (present when moe-expert-specialization ran).
        if block_n_m:
            entry["BLOCK_N"] = int(block_n_m.group(1))
        if policy_m:
            entry["specialization_policy"] = policy_m.group(1)

        # Cost analysis attributes (present when moe-expert-cost-analysis ran).
        if flops_m:
            entry["flops_estimate"] = int(flops_m.group(1))
        if bytes_m:
            entry["bytes_estimate"] = int(bytes_m.group(1))
        if ai_m:
            entry["arithmetic_intensity"] = float(ai_m.group(1))
        if cost_cls_m:
            entry["cost_class"] = cost_cls_m.group(1)

        decisions[f"expert_slot_{slot_m.group(1)}"] = entry

    return decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit compiler_decisions.json from the remora pass pipeline."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input MLIR file",
    )
    parser.add_argument(
        "--num-experts", type=int, default=1,
        help="num_experts passed to moe-expert-outlining (default: 1)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for compiler_decisions.json",
    )
    parser.add_argument(
        "--routing-stats", default=None, metavar="PATH",
        help=(
            "Path to routing_stats.json (output of generate_routing_stats.py). "
            "When provided, uses V2 token-based BLOCK_M policy "
            "(specialization_policy=token_pgo_v2) instead of V1 shape-static policy."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"error: input file not found: {args.input}")

    routing_stats = None
    if args.routing_stats:
        if not os.path.exists(args.routing_stats):
            sys.exit(f"error: routing stats file not found: {args.routing_stats}")
        with open(args.routing_stats) as f:
            routing_stats = json.load(f)
        policy_label = "token_pgo_v2"
    else:
        policy_label = "shape_static_v1"

    print(f"Running outlining + cost-analysis + specialization "
          f"(num_experts={args.num_experts}) on {args.input} ...")
    print(f"Specialization policy: {policy_label}")
    ir_text = run_pipeline(args.input, args.num_experts)

    decisions = parse_decisions(ir_text)

    if not decisions:
        sys.exit(
            "error: no @expert_slot_N functions found in the outlined IR.\n"
            "       Check that the input IR matches the expected format and "
            "that --num-experts matches the batch dimension."
        )

    if routing_stats is not None:
        decisions = apply_pgo_overrides(decisions, routing_stats)

    ordered = dict(sorted(decisions.items(),
                          key=lambda kv: int(kv[0].split("_")[-1])))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(ordered)} expert decisions to {args.output}")
    for slot, cfg in ordered.items():
        ai_str  = (f"  AI={cfg['arithmetic_intensity']:.2f}"
                   if "arithmetic_intensity" in cfg else "")
        cls_str = (f"  cost_class={cfg['cost_class']!r}"
                   if "cost_class" in cfg else "")
        tok_str = (f"  expected_tokens={cfg['expected_tokens']}"
                   if "expected_tokens" in cfg else "")
        print(f"  {slot}: tile_class={cfg['tile_class']!r:8s}  "
              f"BLOCK_M={cfg['BLOCK_M']:3d}  "
              f"intermediate_dim={cfg['intermediate_dim']}"
              f"{tok_str}{ai_str}{cls_str}")


if __name__ == "__main__":
    main()
