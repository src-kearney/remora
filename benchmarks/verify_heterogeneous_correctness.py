#!/usr/bin/env python3
"""
benchmarks/verify_heterogeneous_correctness.py

Correctness validation for Remora per-expert dispatch on the heterogeneous
MoE workload: 8 experts with four distinct intermediate dimensions
(F = 14336, 8192, 4096, 2048), each compiled with a different BLOCK_M
selected by the Remora specialization pass.

What this validates:
  - Each expert's Triton kernel produces numerically correct output when
    run with its compiler-assigned tile configuration (BLOCK_M / BLOCK_N).
  - Experts with different intermediate dims are handled independently.
  - The compiler_decisions.json wire (BLOCK_M per slot) is honoured.

Comparison: Triton output vs PyTorch fp32 reference, per expert.
Pass criterion: max |error| < atol = 1e-3 (fp16 accumulation noise).

Usage:
    python3 benchmarks/verify_heterogeneous_correctness.py
    python3 benchmarks/verify_heterogeneous_correctness.py \\
        --decisions compiler_decisions.json
"""

import argparse
import json
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DECISIONS = os.path.join(REPO_ROOT, "compiler_decisions.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

# ---------------------------------------------------------------------------
# Heterogeneous expert layout (mirrors heterogeneous_moe_layer.mlir)
# ---------------------------------------------------------------------------
HIDDEN_DIM = 4096
TOKENS_PER_EXPERT = 512   # T — matches the IR batch size

EXPERT_LAYOUT = [
    {"slot": 0, "F": 14336, "cost_class": "large"},
    {"slot": 1, "F": 14336, "cost_class": "large"},
    {"slot": 2, "F":  8192, "cost_class": "medium"},
    {"slot": 3, "F":  8192, "cost_class": "medium"},
    {"slot": 4, "F":  4096, "cost_class": "small"},
    {"slot": 5, "F":  4096, "cost_class": "small"},
    {"slot": 6, "F":  2048, "cost_class": "tiny"},
    {"slot": 7, "F":  2048, "cost_class": "tiny"},
]

# ---------------------------------------------------------------------------
# Import Triton kernel from triton_expert_kernel.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from triton_expert_kernel import (
    _expert_ffn_with_cfg,
    _pytorch_expert_ffn_ref,
    _BLOCK_M_TO_NUM_WARPS,
)

import triton


# ---------------------------------------------------------------------------
# Warmup one distinct (BLOCK_M, BLOCK_N, F) combination
# ---------------------------------------------------------------------------

def warmup_config(
    hidden_dim: int,
    inter_dim: int,
    block_m: int,
    block_n: int,
    block_k: int,
    device: str,
) -> None:
    dtype = torch.float16
    T = max(block_m, 1)
    x      = torch.zeros(T, hidden_dim, dtype=dtype, device=device)
    w_gate = torch.zeros(hidden_dim, inter_dim, dtype=dtype, device=device)
    w_up   = torch.zeros(hidden_dim, inter_dim, dtype=dtype, device=device)
    w_down = torch.zeros(inter_dim,  hidden_dim, dtype=dtype, device=device)
    nw = _BLOCK_M_TO_NUM_WARPS.get(block_m, 4)
    _expert_ffn_with_cfg(x, w_gate, w_up, w_down,
                         block_m, block_n, block_k, nw)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Load compiler decisions
# ---------------------------------------------------------------------------

def load_decisions(path: str) -> dict[int, dict]:
    """
    Returns a dict mapping slot_id → {BLOCK_M, BLOCK_N, tile_class, ...}.
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key, cfg in raw.items():
        slot_id = int(key.split("_")[-1])
        out[slot_id] = cfg
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heterogeneous correctness check for Remora per-expert dispatch."
    )
    parser.add_argument(
        "--decisions", default=DEFAULT_DECISIONS,
        help=f"Path to compiler_decisions.json (default: {DEFAULT_DECISIONS})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for weight initialisation (default: 42)",
    )
    parser.add_argument(
        "--tokens", type=int, default=TOKENS_PER_EXPERT,
        help=f"Tokens per expert (default: {TOKENS_PER_EXPERT})",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-3,
        help="Absolute tolerance for correctness check (default: 1e-3)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device found — this script requires a GPU.")
        sys.exit(1)

    if not os.path.exists(args.decisions):
        sys.exit(
            f"error: compiler_decisions.json not found at {args.decisions}\n"
            f"       Run: python3 scripts/emit_compiler_decisions.py "
            f"--input mlir/stablehlo/heterogeneous_moe_layer.mlir "
            f"--num-experts 1 --output {args.decisions}"
        )

    device = "cuda"
    dtype  = torch.float16
    D      = HIDDEN_DIM
    T      = args.tokens

    print("=" * 72)
    print("Remora  heterogeneous expert correctness check")
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"Layout:   {len(EXPERT_LAYOUT)} experts, "
          f"F in {{14336, 8192, 4096, 2048}}, T={T}, D={D}")
    print(f"Decisions: {args.decisions}")
    print("=" * 72)
    print()

    # Load compiler decisions
    decisions = load_decisions(args.decisions)
    print("Compiler decisions loaded:")
    for slot, cfg in sorted(decisions.items()):
        print(f"  expert_slot_{slot}: tile_class={cfg['tile_class']!r:8s}  "
              f"BLOCK_M={cfg['BLOCK_M']:3d}  BLOCK_N={cfg['BLOCK_N']:3d}  "
              f"F={cfg['intermediate_dim']}")
    print()

    # Warmup: one JIT compile per distinct (BLOCK_M, BLOCK_N, F) triple
    print("Warming up Triton kernels (one JIT per distinct config)...")
    seen_configs: set[tuple] = set()
    for expert in EXPERT_LAYOUT:
        slot = expert["slot"]
        F    = expert["F"]
        cfg  = decisions.get(slot, {})
        bm   = cfg.get("BLOCK_M", 64)
        bn   = cfg.get("BLOCK_N", 64)
        bk   = 32
        key  = (bm, bn, F)
        if key not in seen_configs:
            seen_configs.add(key)
            warmup_config(D, F, bm, bn, bk, device)
            print(f"  BLOCK_M={bm:3d}  BLOCK_N={bn:3d}  F={F:5d}  compiled ✓")
    torch.cuda.synchronize()
    print()

    # Generate weights and tokens
    torch.manual_seed(args.seed)
    scale = 0.02

    expert_weights = []
    for expert in EXPERT_LAYOUT:
        F = expert["F"]
        expert_weights.append({
            "w_gate": (torch.randn(D, F, dtype=dtype, device=device) * scale),
            "w_up":   (torch.randn(D, F, dtype=dtype, device=device) * scale),
            "w_down": (torch.randn(F, D, dtype=dtype, device=device) * scale),
        })

    expert_tokens = [
        torch.randn(T, D, dtype=dtype, device=device) * 0.1
        for _ in EXPERT_LAYOUT
    ]

    # Run Triton kernels with compiler-assigned BLOCK_M per expert
    torch.cuda.synchronize()
    triton_outs = []
    for expert in EXPERT_LAYOUT:
        slot = expert["slot"]
        F    = expert["F"]
        cfg  = decisions.get(slot, {})
        bm   = cfg.get("BLOCK_M", 64)
        bn   = cfg.get("BLOCK_N", 64)
        bk   = 32
        nw   = _BLOCK_M_TO_NUM_WARPS.get(bm, 4)
        x      = expert_tokens[slot]
        w      = expert_weights[slot]
        out = _expert_ffn_with_cfg(
            x, w["w_gate"], w["w_up"], w["w_down"],
            bm, bn, bk, nw,
        )
        triton_outs.append(out)
    torch.cuda.synchronize()

    # PyTorch fp32 reference per expert
    ref_outs = [
        _pytorch_expert_ffn_ref(
            expert_tokens[e],
            expert_weights[e]["w_gate"],
            expert_weights[e]["w_up"],
            expert_weights[e]["w_down"],
        )
        for e in range(len(EXPERT_LAYOUT))
    ]

    # Report
    print(f"{'slot':>4}  {'F':>6}  {'tile_class':>10}  {'BLOCK_M':>7}  "
          f"{'max |err|':>10}  {'mean |err|':>10}  status")
    print("-" * 72)

    failures = []
    rows = []

    for i, expert in enumerate(EXPERT_LAYOUT):
        slot = expert["slot"]
        F    = expert["F"]
        cost = expert["cost_class"]
        cfg  = decisions.get(slot, {})
        bm   = cfg.get("BLOCK_M", "?")
        tc   = cfg.get("tile_class", "?")

        tri = triton_outs[i].float()
        ref = ref_outs[i].float()

        max_err  = (tri - ref).abs().max().item()
        mean_err = (tri - ref).abs().mean().item()
        ok       = max_err < args.atol
        status   = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(f"expert_slot_{slot}")

        print(f"  {slot:2d}  {F:6d}  {tc:>10}  {bm:>7}  "
              f"{max_err:10.2e}  {mean_err:10.2e}  {status}")

        rows.append({
            "slot": slot,
            "intermediate_dim": F,
            "cost_class": cost,
            "tile_class": tc,
            "BLOCK_M": bm,
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "pass": ok,
        })

    print()

    # Summary
    if failures:
        print(f"FAILED — {len(failures)} expert(s) exceeded atol={args.atol}: {failures}")
        result_status = "FAIL"
    else:
        print(f"ALL {len(EXPERT_LAYOUT)} EXPERTS PASS  (atol={args.atol}, fp16)")
        print()
        print("  Each expert's Triton kernel produces numerically correct output")
        print("  when run with its compiler-assigned BLOCK_M.")
        print("  Experts with F=14336 (BLOCK_M=128) through F=2048 (BLOCK_M=16)")
        print("  all pass at the same tolerance as the homogeneous check.")
        result_status = "PASS"

    # Save results JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "heterogeneous_correctness.json")
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "hidden_dim": D,
        "tokens_per_expert": T,
        "atol": args.atol,
        "seed": args.seed,
        "decisions_file": os.path.basename(args.decisions),
        "status": result_status,
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)
        f.write("\n")
    print()
    print(f"Results written to {out_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
