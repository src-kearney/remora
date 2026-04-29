#!/usr/bin/env python3
"""
benchmarks/sweep_heterogeneous.py

Heterogeneous expert specialization benchmark.

Compares two dispatch strategies on a workload with 8 experts of different
intermediate dimensions (F = 14336, 8192, 4096, 2048):

  baseline  — uniform tile config: BLOCK_M=64 for every expert regardless of shape
  remora    — compiler-specialized: BLOCK_M per expert from compiler_decisions.json
              (128 for large, 64 for medium, 32 for small, 16 for tiny)

This is the benchmark that validates Remora's core claim: that per-expert
compiler decisions produce more appropriate tile configurations than uniform
treatment, and that the compiler_decisions.json wire actually changes what
runs on the GPU.

Measurement:
  - Per-expert latency (CUDA Events, median over TIMED_ITERS)
  - Total latency summed across all 8 experts
  - Speedup: baseline / remora per expert and overall

Output:
  stdout                                          — formatted comparison table
  benchmarks/results/sweep_heterogeneous.json
  benchmarks/results/sweep_heterogeneous.csv

Usage:
    python3 benchmarks/sweep_heterogeneous.py
    python3 benchmarks/sweep_heterogeneous.py --decisions compiler_decisions.json
"""

import argparse
import csv
import json
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
RESULTS   = os.path.join(_HERE, "results")

DEFAULT_DECISIONS = os.path.join(REPO_ROOT, "compiler_decisions.json")

sys.path.insert(0, _HERE)
from triton_expert_kernel import _expert_ffn_with_cfg, _BLOCK_M_TO_NUM_WARPS

import triton

# ---------------------------------------------------------------------------
# Workload definition — mirrors heterogeneous_moe_layer.mlir
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

# Baseline: one config for all experts regardless of shape
BASELINE_BLOCK_M = 64
BASELINE_BLOCK_N = 64
BASELINE_BLOCK_K = 32

WARMUP_ITERS = 5
TIMED_ITERS  = 50

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_expert(
    x:      torch.Tensor,
    w_gate: torch.Tensor,
    w_up:   torch.Tensor,
    w_down: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    warmup: int = WARMUP_ITERS,
    timed:  int = TIMED_ITERS,
) -> float:
    """Return median latency in ms for one expert's forward pass."""
    nw = _BLOCK_M_TO_NUM_WARPS.get(block_m, 4)

    # Warmup
    for _ in range(warmup):
        _expert_ffn_with_cfg(x, w_gate, w_up, w_down,
                              block_m, block_n, block_k, nw)
    torch.cuda.synchronize()

    # Timed iters
    latencies = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _expert_ffn_with_cfg(x, w_gate, w_up, w_down,
                              block_m, block_n, block_k, nw)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies.sort()
    return latencies[len(latencies) // 2]   # median


# ---------------------------------------------------------------------------
# Load compiler decisions
# ---------------------------------------------------------------------------

def load_decisions(path: str) -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k.split("_")[-1]): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heterogeneous specialization benchmark: uniform vs Remora."
    )
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",  type=int, default=TIMED_ITERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device — requires GPU.")
        sys.exit(1)

    if not os.path.exists(args.decisions):
        sys.exit(f"error: {args.decisions} not found — run emit_compiler_decisions.py first")

    device = "cuda"
    dtype  = torch.float16
    D      = HIDDEN_DIM
    T      = TOKENS_PER_EXPERT

    decisions = load_decisions(args.decisions)

    print("=" * 76)
    print("Remora  heterogeneous specialization benchmark")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Layout:  {len(EXPERT_LAYOUT)} experts, F∈{{14336,8192,4096,2048}}, T={T}, D={D}")
    print(f"Warmup:  {args.warmup} iters   Timed: {args.iters} iters   Metric: median latency")
    print()
    print("baseline  — BLOCK_M=64 for all experts (uniform, shape-unaware)")
    print("remora    — BLOCK_M per compiler_decisions.json (128/64/32/16)")
    print("=" * 76)
    print()

    # Generate weights and tokens
    torch.manual_seed(args.seed)
    scale = 0.02

    expert_weights = []
    for e in EXPERT_LAYOUT:
        F = e["F"]
        expert_weights.append({
            "w_gate": torch.randn(D, F, dtype=dtype, device=device) * scale,
            "w_up":   torch.randn(D, F, dtype=dtype, device=device) * scale,
            "w_down": torch.randn(F, D, dtype=dtype, device=device) * scale,
        })

    expert_tokens = [
        torch.randn(T, D, dtype=dtype, device=device) * 0.1
        for _ in EXPERT_LAYOUT
    ]

    # --- Warmup JIT compilation for all distinct configs ---
    print("Compiling kernels...")
    seen: set[tuple] = set()
    configs_to_warmup = []
    # baseline config
    configs_to_warmup.append((BASELINE_BLOCK_M, BASELINE_BLOCK_N, BASELINE_BLOCK_K,
                               EXPERT_LAYOUT[0]["F"]))
    # remora configs
    for e in EXPERT_LAYOUT:
        slot = e["slot"]
        cfg  = decisions.get(slot, {})
        configs_to_warmup.append((cfg.get("BLOCK_M", 64), cfg.get("BLOCK_N", 64),
                                   32, e["F"]))

    for bm, bn, bk, F in configs_to_warmup:
        key = (bm, bn, F)
        if key in seen:
            continue
        seen.add(key)
        nw = _BLOCK_M_TO_NUM_WARPS.get(bm, 4)
        x_w  = torch.zeros(max(bm, 1), D, dtype=dtype, device=device)
        wg_w = torch.zeros(D, F, dtype=dtype, device=device)
        wu_w = torch.zeros(D, F, dtype=dtype, device=device)
        wd_w = torch.zeros(F, D, dtype=dtype, device=device)
        _expert_ffn_with_cfg(x_w, wg_w, wu_w, wd_w, bm, bn, bk, nw)
        torch.cuda.synchronize()
        print(f"  BLOCK_M={bm:3d}  BLOCK_N={bn:3d}  F={F:5d}  ✓")
    print()

    # --- Benchmark ---
    print(f"  {'slot':>4}  {'F':>6}  {'class':>6}  "
          f"{'base BM':>7}  {'base ms':>8}  "
          f"{'remo BM':>7}  {'remo ms':>8}  "
          f"{'speedup':>8}")
    print("  " + "-" * 72)

    rows = []
    total_base_ms  = 0.0
    total_remo_ms  = 0.0

    for i, e in enumerate(EXPERT_LAYOUT):
        slot = e["slot"]
        F    = e["F"]
        cost = e["cost_class"]
        w    = expert_weights[i]
        x    = expert_tokens[i]

        # Baseline timing
        base_ms = time_expert(
            x, w["w_gate"], w["w_up"], w["w_down"],
            BASELINE_BLOCK_M, BASELINE_BLOCK_N, BASELINE_BLOCK_K,
            args.warmup, args.iters,
        )

        # Remora timing
        cfg    = decisions.get(slot, {})
        remo_bm = cfg.get("BLOCK_M", 64)
        remo_bn = cfg.get("BLOCK_N", 64)
        remo_bk = 32
        remo_ms = time_expert(
            x, w["w_gate"], w["w_up"], w["w_down"],
            remo_bm, remo_bn, remo_bk,
            args.warmup, args.iters,
        )

        speedup = base_ms / remo_ms if remo_ms > 0 else float("nan")
        total_base_ms += base_ms
        total_remo_ms += remo_ms

        winner = "remora" if speedup > 1.02 else ("baseline" if speedup < 0.98 else "~tie")

        print(f"  {slot:4d}  {F:6d}  {cost:>6}  "
              f"{BASELINE_BLOCK_M:7d}  {base_ms:8.3f}  "
              f"{remo_bm:7d}  {remo_ms:8.3f}  "
              f"{speedup:7.2f}x  [{winner}]")

        rows.append({
            "slot":            slot,
            "intermediate_dim": F,
            "cost_class":      cost,
            "baseline_block_m": BASELINE_BLOCK_M,
            "baseline_ms":     base_ms,
            "remora_block_m":  remo_bm,
            "remora_ms":       remo_ms,
            "speedup":         speedup,
            "winner":          winner,
        })

    total_speedup = total_base_ms / total_remo_ms if total_remo_ms > 0 else float("nan")

    print("  " + "-" * 72)
    print(f"  {'TOTAL':>38}  {total_base_ms:8.3f}  {'':7}  {total_remo_ms:8.3f}  "
          f"{total_speedup:7.2f}x")
    print()

    # Summary
    remora_wins  = sum(1 for r in rows if r["winner"] == "remora")
    base_wins    = sum(1 for r in rows if r["winner"] == "baseline")
    ties         = sum(1 for r in rows if r["winner"] == "~tie")
    print(f"remora wins: {remora_wins}/8   baseline wins: {base_wins}/8   ties: {ties}/8")
    print(f"total speedup: {total_speedup:.2f}x  "
          f"(baseline {total_base_ms:.3f} ms → remora {total_remo_ms:.3f} ms)")

    # Save results
    os.makedirs(RESULTS, exist_ok=True)
    meta = {
        "gpu":              torch.cuda.get_device_name(0),
        "hidden_dim":       D,
        "tokens_per_expert": T,
        "warmup_iters":     args.warmup,
        "timed_iters":      args.iters,
        "baseline_block_m": BASELINE_BLOCK_M,
        "decisions_file":   os.path.basename(args.decisions),
        "total_baseline_ms": total_base_ms,
        "total_remora_ms":   total_remo_ms,
        "total_speedup":     total_speedup,
    }
    out_json = os.path.join(RESULTS, "sweep_heterogeneous.json")
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)
        f.write("\n")

    out_csv = os.path.join(RESULTS, "sweep_heterogeneous.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Results → {out_json}")
    print(f"          {out_csv}")


if __name__ == "__main__":
    main()
