#!/usr/bin/env python3
"""
benchmarks/sweep_heterogeneous_pgo.py

V1 vs V2 specialization policy benchmark.

Compares three dispatch strategies across three token distributions:

  baseline — BLOCK_M=64 uniform for all experts
  v1       — shape_static_v1: BLOCK_M from intermediate dim F
             (128/64/32/16 for large/medium/small/tiny, compiler_decisions.json)
  v2_pgo   — token_pgo_v2: BLOCK_M from expected per-expert token count T
             (T≥256→128, T≥128→64, T≥64→32, else 16)

Token distributions:
  uniform      — 512 tokens per expert (same for all)
  proportional — tokens ∝ expert F dim (large experts get more tokens)
  skewed       — Zipf 1/rank (expert 0 gets ~188, expert 7 gets ~24)

Oracle reference (from sweep_heterogeneous.py --extended):
  Empirically best BLOCK_M per expert found by sweeping {16,32,64,128}.
  Loaded from benchmarks/results/sweep_heterogeneous_proportional.json if present.

Output:
  stdout  — per-expert table + summary
  benchmarks/results/sweep_heterogeneous_pgo.json
  benchmarks/results/sweep_heterogeneous_pgo.csv

Usage:
    python3 benchmarks/sweep_heterogeneous_pgo.py
    python3 benchmarks/sweep_heterogeneous_pgo.py --decisions compiler_decisions.json
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
ORACLE_RESULTS    = os.path.join(RESULTS, "sweep_heterogeneous_proportional.json")

sys.path.insert(0, _HERE)
from triton_expert_kernel import _expert_ffn_with_cfg, _BLOCK_M_TO_NUM_WARPS

import triton

# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------
HIDDEN_DIM = 4096

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

BASELINE_BLOCK_M = 64
BASELINE_BLOCK_N = 64
BASELINE_BLOCK_K = 32

WARMUP_ITERS = 5
TIMED_ITERS  = 50

# ---------------------------------------------------------------------------
# V2 policy: token-count-based BLOCK_M selection
# Must match emit_compiler_decisions.py::block_m_from_tokens
# ---------------------------------------------------------------------------

def block_m_from_tokens(t: int) -> int:
    """Select BLOCK_M from per-expert expected token count T.

    Derived from oracle sweep results in sweep_heterogeneous_proportional.json:
    optimal BLOCK_M tracks token count T, not intermediate dim F.
    """
    if t >= 256: return 128
    if t >= 128: return 64
    if t >= 64:  return 32
    return 16

# ---------------------------------------------------------------------------
# Token distributions
# ---------------------------------------------------------------------------
TOTAL_TOKENS = 512

def _make_proportional_tokens(total: int = TOTAL_TOKENS) -> list[int]:
    F_vals = [e["F"] for e in EXPERT_LAYOUT]
    tot_F  = sum(F_vals)
    tokens = [max(16, round(total * F / tot_F)) for F in F_vals]
    tokens[-1] = max(16, tokens[-1] + (total - sum(tokens)))
    return tokens

def _make_skewed_tokens(total: int = TOTAL_TOKENS) -> list[int]:
    n       = len(EXPERT_LAYOUT)
    weights = [1.0 / (i + 1) for i in range(n)]
    tot_w   = sum(weights)
    tokens  = [max(16, round(total * w / tot_w)) for w in weights]
    tokens[-1] = max(16, tokens[-1] + (total - sum(tokens)))
    return tokens

DISTRIBUTIONS: dict[str, list[int]] = {
    "uniform":      [TOTAL_TOKENS] * len(EXPERT_LAYOUT),
    "proportional": _make_proportional_tokens(),
    "skewed":       _make_skewed_tokens(),
}

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
    for _ in range(warmup):
        _expert_ffn_with_cfg(x, w_gate, w_up, w_down, block_m, block_n, block_k, nw)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _expert_ffn_with_cfg(x, w_gate, w_up, w_down, block_m, block_n, block_k, nw)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies.sort()
    return latencies[len(latencies) // 2]

# ---------------------------------------------------------------------------
# Load compiler decisions
# ---------------------------------------------------------------------------

def load_decisions(path: str) -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k.split("_")[-1]): v for k, v in raw.items()}

# ---------------------------------------------------------------------------
# Load oracle summary from previous extended run (optional)
# ---------------------------------------------------------------------------

def load_oracle_summary() -> dict[str, dict] | None:
    """Load per-distribution oracle totals from sweep_heterogeneous_proportional.json."""
    if not os.path.exists(ORACLE_RESULTS):
        return None
    with open(ORACLE_RESULTS) as f:
        data = json.load(f)
    summary = {}
    for s in data.get("summary", []):
        summary[s["distribution"]] = {
            "oracle_total_ms":   s["oracle_total_ms"],
            "oracle_vs_baseline": s["oracle_vs_baseline"],
        }
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 vs V2 PGO specialization policy benchmark."
    )
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS,
                        help="Path to V1 compiler_decisions.json")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--warmup",  type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",   type=int, default=TIMED_ITERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device — requires GPU.")
        sys.exit(1)

    if not os.path.exists(args.decisions):
        sys.exit(f"error: {args.decisions} not found — run emit_compiler_decisions.py first")

    device = "cuda"
    dtype  = torch.float16
    D      = HIDDEN_DIM

    v1_decisions = load_decisions(args.decisions)
    oracle_data  = load_oracle_summary()

    print("=" * 80)
    print("Remora  V1 vs V2 PGO specialization benchmark")
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Layout:    {len(EXPERT_LAYOUT)} experts, F∈{{14336,8192,4096,2048}}")
    print(f"Warmup:    {args.warmup} iters   Timed: {args.iters} iters   Metric: median")
    print()
    print("Strategies:")
    print("  baseline  — BLOCK_M=64 for all experts (uniform, shape-unaware)")
    print("  v1        — shape_static_v1: BLOCK_M from F  (compiler_decisions.json)")
    print("  v2_pgo    — token_pgo_v2:    BLOCK_M from T  (T≥256→128, ≥128→64, ≥64→32, else 16)")
    if oracle_data:
        print("  oracle    — empirically best BLOCK_M per expert (from sweep_heterogeneous --extended)")
    print()
    print("Token distributions:")
    for dname, toks in DISTRIBUTIONS.items():
        print(f"  {dname:<13} {toks}")
    print("=" * 80)

    # --- Warmup all kernels ---
    print("\nCompiling kernels...")
    seen: set[tuple] = set()
    torch.manual_seed(args.seed)
    scale = 0.02

    # Pre-generate weights (constant across distributions, D and F per expert)
    expert_weights = []
    for e in EXPERT_LAYOUT:
        F = e["F"]
        expert_weights.append({
            "w_gate": torch.randn(D, F, dtype=dtype, device=device) * scale,
            "w_up":   torch.randn(D, F, dtype=dtype, device=device) * scale,
            "w_down": torch.randn(F, D, dtype=dtype, device=device) * scale,
        })

    # Collect all (BM, BN, F) combos that will be timed
    all_configs: list[tuple[int, int, int]] = []
    for e in EXPERT_LAYOUT:
        slot = e["slot"]
        F_e  = e["F"]
        all_configs.append((BASELINE_BLOCK_M, BASELINE_BLOCK_N, F_e))
        v1_cfg = v1_decisions.get(slot, {})
        all_configs.append((v1_cfg.get("BLOCK_M", 64), v1_cfg.get("BLOCK_N", 64), F_e))
        for tokens_list in DISTRIBUTIONS.values():
            T_e  = tokens_list[e["slot"]]
            v2bm = block_m_from_tokens(T_e)
            all_configs.append((v2bm, 64, F_e))   # V2 uses BN=64

    for bm, bn, F_e in all_configs:
        key = (bm, bn, F_e)
        if key in seen:
            continue
        seen.add(key)
        nw  = _BLOCK_M_TO_NUM_WARPS.get(bm, 4)
        T_w = max(bm, 1)
        x_w  = torch.zeros(T_w, D,   dtype=dtype, device=device)
        wg_w = torch.zeros(D,   F_e, dtype=dtype, device=device)
        wu_w = torch.zeros(D,   F_e, dtype=dtype, device=device)
        wd_w = torch.zeros(F_e, D,   dtype=dtype, device=device)
        _expert_ffn_with_cfg(x_w, wg_w, wu_w, wd_w, bm, bn, BASELINE_BLOCK_K, nw)
        torch.cuda.synchronize()
        print(f"  BLOCK_M={bm:3d}  BLOCK_N={bn:3d}  F={F_e:5d}  ✓")
    print()

    # --- Per-distribution benchmarks ---
    all_rows: list[dict] = []
    summaries: list[dict] = []

    HDR = (f"  {'dist':>13}  {'slot':>4}  {'F':>6}  {'T_e':>4}  "
           f"{'base_ms':>8}  {'v1_BM':>5}  {'v1_ms':>8}  "
           f"{'v2_BM':>5}  {'v2_ms':>8}")
    SEP = "  " + "-" * (len(HDR) - 2)

    for dist_name, tokens_list in DISTRIBUTIONS.items():
        # V2 BLOCK_M decisions for this distribution
        v2_bm_for = {e["slot"]: block_m_from_tokens(tokens_list[e["slot"]])
                     for e in EXPERT_LAYOUT}

        print(f"\n{'─'*80}")
        print(f"Distribution: {dist_name}  |  tokens: {tokens_list}")
        print(f"V2 BLOCK_M: {[v2_bm_for[e['slot']] for e in EXPERT_LAYOUT]}")
        print(HDR)
        print(SEP)

        dist_tokens = [
            torch.randn(T_e, D, dtype=dtype, device=device) * 0.1
            for T_e in tokens_list
        ]

        base_total = 0.0
        v1_total   = 0.0
        v2_total   = 0.0

        for i, e in enumerate(EXPERT_LAYOUT):
            slot = e["slot"]
            F_e  = e["F"]
            cost = e["cost_class"]
            w    = expert_weights[i]
            x    = dist_tokens[i]
            T_e  = tokens_list[slot]

            base_ms = time_expert(
                x, w["w_gate"], w["w_up"], w["w_down"],
                BASELINE_BLOCK_M, BASELINE_BLOCK_N, BASELINE_BLOCK_K,
                args.warmup, args.iters,
            )

            v1_cfg  = v1_decisions.get(slot, {})
            v1_bm   = v1_cfg.get("BLOCK_M", 64)
            v1_bn   = v1_cfg.get("BLOCK_N", 64)
            v1_ms   = time_expert(
                x, w["w_gate"], w["w_up"], w["w_down"],
                v1_bm, v1_bn, BASELINE_BLOCK_K,
                args.warmup, args.iters,
            )

            v2_bm   = v2_bm_for[slot]
            v2_ms   = time_expert(
                x, w["w_gate"], w["w_up"], w["w_down"],
                v2_bm, 64, BASELINE_BLOCK_K,   # V2: BN=64 fixed (matches oracle)
                args.warmup, args.iters,
            )

            base_total += base_ms
            v1_total   += v1_ms
            v2_total   += v2_ms

            print(f"  {dist_name:>13}  {slot:4d}  {F_e:6d}  {T_e:4d}  "
                  f"{base_ms:8.3f}  {v1_bm:5d}  {v1_ms:8.3f}  "
                  f"{v2_bm:5d}  {v2_ms:8.3f}")

            all_rows.append({
                "distribution":      dist_name,
                "slot":              slot,
                "intermediate_dim":  F_e,
                "cost_class":        cost,
                "tokens_per_expert": T_e,
                "baseline_block_m":  BASELINE_BLOCK_M,
                "baseline_ms":       base_ms,
                "v1_block_m":        v1_bm,
                "v1_ms":             v1_ms,
                "v2_block_m":        v2_bm,
                "v2_ms":             v2_ms,
            })

        v1_speedup = base_total / v1_total   if v1_total   > 0 else float("nan")
        v2_speedup = base_total / v2_total   if v2_total   > 0 else float("nan")

        print(SEP)
        print(f"  {'TOTAL':>13}  {'':>4}  {'':>6}  {'':>4}  "
              f"{base_total:8.3f}  {'':>5}  {v1_total:8.3f}  "
              f"{'':>5}  {v2_total:8.3f}")
        print(f"  v1_vs_baseline={v1_speedup:.2f}x  "
              f"v2_vs_baseline={v2_speedup:.2f}x", end="")

        oracle_ms      = float("nan")
        oracle_speedup = float("nan")
        if oracle_data and dist_name in oracle_data:
            oracle_ms      = oracle_data[dist_name]["oracle_total_ms"]
            oracle_speedup = oracle_data[dist_name]["oracle_vs_baseline"]
            print(f"  oracle={oracle_ms:.3f}ms ({oracle_speedup:.2f}x)", end="")
        print()

        summaries.append({
            "distribution":      dist_name,
            "tokens":            tokens_list,
            "baseline_total_ms": base_total,
            "v1_total_ms":       v1_total,
            "v1_vs_baseline":    v1_speedup,
            "v2_total_ms":       v2_total,
            "v2_vs_baseline":    v2_speedup,
            "oracle_total_ms":   oracle_ms,
            "oracle_vs_baseline": oracle_speedup,
        })

    # --- Final summary table ---
    print()
    print("=" * 80)
    print("Summary")
    have_oracle = oracle_data is not None
    hdr2  = (f"  {'distribution':>13}  {'baseline':>10}  "
             f"{'v1':>8}  {'v1/base':>7}  "
             f"{'v2':>8}  {'v2/base':>7}")
    hdr2 += f"  {'oracle':>8}  {'orc/base':>8}" if have_oracle else ""
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for s in summaries:
        row = (f"  {s['distribution']:>13}  "
               f"{s['baseline_total_ms']:10.3f}  "
               f"{s['v1_total_ms']:8.3f}  "
               f"{s['v1_vs_baseline']:6.2f}x  "
               f"{s['v2_total_ms']:8.3f}  "
               f"{s['v2_vs_baseline']:6.2f}x")
        if have_oracle and not (s["oracle_total_ms"] != s["oracle_total_ms"]):
            row += (f"  {s['oracle_total_ms']:8.3f}  "
                    f"{s['oracle_vs_baseline']:7.2f}x")
        print(row)
    print("=" * 80)

    # --- Save results ---
    os.makedirs(RESULTS, exist_ok=True)
    meta = {
        "gpu":              torch.cuda.get_device_name(0),
        "hidden_dim":       D,
        "warmup_iters":     args.warmup,
        "timed_iters":      args.iters,
        "baseline_block_m": BASELINE_BLOCK_M,
        "v1_decisions_file": os.path.basename(args.decisions),
        "v2_policy":        "token_pgo_v2",
        "v2_thresholds":    "T>=256→128, T>=128→64, T>=64→32, else 16",
        "distributions":    {k: v for k, v in DISTRIBUTIONS.items()},
    }

    out_json = os.path.join(RESULTS, "sweep_heterogeneous_pgo.json")
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "results": all_rows, "summary": summaries},
                  f, indent=2)
        f.write("\n")

    csv_fields = [
        "distribution", "slot", "intermediate_dim", "cost_class",
        "tokens_per_expert",
        "baseline_block_m", "baseline_ms",
        "v1_block_m", "v1_ms",
        "v2_block_m", "v2_ms",
    ]
    out_csv = os.path.join(RESULTS, "sweep_heterogeneous_pgo.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Results → {out_json}")
    print(f"          {out_csv}")


if __name__ == "__main__":
    main()
