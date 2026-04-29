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

Output (default mode):
  stdout                                          — formatted comparison table
  benchmarks/results/sweep_heterogeneous.json
  benchmarks/results/sweep_heterogeneous.csv

Extended mode (--extended):
  Runs three token distributions (uniform, proportional, skewed) and adds an
  oracle sweep (best BLOCK_M per expert found empirically).  Saves to:
  benchmarks/results/sweep_heterogeneous_proportional.json
  benchmarks/results/sweep_heterogeneous_proportional.csv

  uniform      — T=512 per expert (same as default)
  proportional — tokens proportional to each expert's intermediate dim F
                 (large experts get more tokens, matching realistic deployment)
  skewed       — Zipf 1/rank: expert 0 gets ~188 tokens, expert 7 gets ~24

  oracle       — for each expert, sweep BLOCK_M ∈ {16,32,64,128} with fixed
                 BLOCK_N=64 BLOCK_K=32 and keep the fastest; upper-bounds what
                 a correct BLOCK_M heuristic could achieve.

Usage:
    python3 benchmarks/sweep_heterogeneous.py
    python3 benchmarks/sweep_heterogeneous.py --decisions compiler_decisions.json
    python3 benchmarks/sweep_heterogeneous.py --extended
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
# Extended mode: oracle BLOCK_M sweep
# ---------------------------------------------------------------------------

# Candidates tried per expert when searching for the empirically best BLOCK_M.
ORACLE_BM_CANDIDATES = [16, 32, 64, 128]

# BLOCK_N and BLOCK_K held fixed across all oracle candidates so only BLOCK_M varies.
# Matches the baseline BN/BK for a clean apples-to-apples BM comparison.
ORACLE_BLOCK_N = 64
ORACLE_BLOCK_K = 32

# ---------------------------------------------------------------------------
# Token distribution helpers (extended mode)
# ---------------------------------------------------------------------------

def _make_proportional_tokens(total: int = 512) -> list[int]:
    """Distribute ``total`` tokens proportionally to each expert's F dim.

    Large experts (F=14336) receive proportionally more tokens, simulating a
    deployment where routing correlates with expert capacity.
    Sum is adjusted to equal ``total`` exactly.
    """
    F_vals = [e["F"] for e in EXPERT_LAYOUT]
    tot_F  = sum(F_vals)
    tokens = [max(16, round(total * F / tot_F)) for F in F_vals]
    # Correct rounding so the total is exact
    diff   = total - sum(tokens)
    tokens[-1] = max(16, tokens[-1] + diff)
    return tokens


def _make_skewed_tokens(total: int = 512) -> list[int]:
    """Zipf-like distribution: expert i receives tokens ∝ 1/(i+1).

    Expert 0 gets ~188 tokens, Expert 7 gets ~24 tokens.
    Sum adjusted to equal ``total`` exactly.
    """
    n       = len(EXPERT_LAYOUT)
    weights = [1.0 / (i + 1) for i in range(n)]
    tot_w   = sum(weights)
    tokens  = [max(16, round(total * w / tot_w)) for w in weights]
    diff    = total - sum(tokens)
    tokens[-1] = max(16, tokens[-1] + diff)
    return tokens

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


def time_expert_oracle(
    x:      torch.Tensor,
    w_gate: torch.Tensor,
    w_up:   torch.Tensor,
    w_down: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    timed:  int = TIMED_ITERS,
) -> tuple[int, float, dict[int, float]]:
    """Sweep ORACLE_BM_CANDIDATES and return the empirically best BLOCK_M.

    BLOCK_N=ORACLE_BLOCK_N and BLOCK_K=ORACLE_BLOCK_K are held fixed so that
    only BLOCK_M varies — this isolates BLOCK_M as the independent variable.

    Returns:
        best_bm   — the BLOCK_M that produced the lowest median latency
        best_ms   — that latency in ms
        all_ms    — {bm: ms} dict for all candidates (useful for debugging)
    """
    all_ms: dict[int, float] = {}
    for bm in ORACLE_BM_CANDIDATES:
        nw = _BLOCK_M_TO_NUM_WARPS.get(bm, 4)
        ms = time_expert(x, w_gate, w_up, w_down,
                         bm, ORACLE_BLOCK_N, ORACLE_BLOCK_K,
                         warmup, timed)
        all_ms[bm] = ms
    best_bm = min(all_ms, key=all_ms.__getitem__)
    return best_bm, all_ms[best_bm], all_ms


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
    parser.add_argument(
        "--extended", action="store_true",
        help=(
            "Also run proportional and skewed token distributions with oracle "
            "BLOCK_M sweep. Saves to sweep_heterogeneous_proportional.json/csv."
        ),
    )
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

    # =========================================================================
    # Extended mode: proportional + skewed + uniform, with oracle sweep
    # =========================================================================

    if not args.extended:
        return

    DISTRIBUTIONS: dict[str, list[int]] = {
        "uniform":      [TOKENS_PER_EXPERT] * len(EXPERT_LAYOUT),
        "proportional": _make_proportional_tokens(TOKENS_PER_EXPERT),
        "skewed":       _make_skewed_tokens(TOKENS_PER_EXPERT),
    }

    print()
    print("=" * 80)
    print("Extended mode: uniform / proportional / skewed  +  oracle BLOCK_M sweep")
    print(f"Oracle candidates: BLOCK_M ∈ {ORACLE_BM_CANDIDATES}  "
          f"(BLOCK_N={ORACLE_BLOCK_N}, BLOCK_K={ORACLE_BLOCK_K} fixed)")
    print()
    print("Token distributions:")
    for dname, toks in DISTRIBUTIONS.items():
        total = sum(toks)
        print(f"  {dname:<13} {toks}  (total={total})")
    print("=" * 80)

    # --- Warmup all oracle (BM, F) combinations ---
    print("\nCompiling oracle kernels...")
    ext_seen: set[tuple] = set()
    for bm in ORACLE_BM_CANDIDATES:
        for e in EXPERT_LAYOUT:
            F_e = e["F"]
            key = (bm, ORACLE_BLOCK_N, F_e)
            if key in ext_seen:
                continue
            ext_seen.add(key)
            nw = _BLOCK_M_TO_NUM_WARPS.get(bm, 4)
            T_w = max(bm, 1)
            x_w  = torch.zeros(T_w, D,   dtype=dtype, device=device)
            wg_w = torch.zeros(D,   F_e, dtype=dtype, device=device)
            wu_w = torch.zeros(D,   F_e, dtype=dtype, device=device)
            wd_w = torch.zeros(F_e, D,   dtype=dtype, device=device)
            _expert_ffn_with_cfg(x_w, wg_w, wu_w, wd_w,
                                 bm, ORACLE_BLOCK_N, ORACLE_BLOCK_K, nw)
            torch.cuda.synchronize()
            print(f"  BLOCK_M={bm:3d}  BLOCK_N={ORACLE_BLOCK_N:3d}  F={F_e:5d}  ✓")
    print()

    # --- Per-expert header ---
    hdr = (f"  {'dist':>13}  {'slot':>4}  {'F':>6}  {'T_e':>4}  "
           f"{'base BM':>7}  {'base ms':>8}  "
           f"{'remo BM':>7}  {'remo ms':>8}  "
           f"{'orac BM':>7}  {'orac ms':>8}")
    sep = "  " + "-" * (len(hdr) - 2)

    ext_rows: list[dict] = []
    ext_summaries: list[dict] = []

    for dist_name, tokens_list in DISTRIBUTIONS.items():
        print(f"\n--- {dist_name} (tokens: {tokens_list}) ---")
        print(hdr)
        print(sep)

        # Generate fresh token tensors for this distribution
        dist_tokens = [
            torch.randn(T_e, D, dtype=dtype, device=device) * 0.1
            for T_e in tokens_list
        ]

        dist_base_total   = 0.0
        dist_remo_total   = 0.0
        dist_oracle_total = 0.0

        for i, e in enumerate(EXPERT_LAYOUT):
            slot = e["slot"]
            F_e  = e["F"]
            cost = e["cost_class"]
            w    = expert_weights[i]
            x    = dist_tokens[i]
            T_e  = tokens_list[i]

            # Baseline
            base_ms = time_expert(
                x, w["w_gate"], w["w_up"], w["w_down"],
                BASELINE_BLOCK_M, BASELINE_BLOCK_N, BASELINE_BLOCK_K,
                args.warmup, args.iters,
            )

            # Remora (compiler-decided BM/BN)
            cfg     = decisions.get(slot, {})
            remo_bm = cfg.get("BLOCK_M", 64)
            remo_bn = cfg.get("BLOCK_N", 64)
            remo_ms = time_expert(
                x, w["w_gate"], w["w_up"], w["w_down"],
                remo_bm, remo_bn, ORACLE_BLOCK_K,
                args.warmup, args.iters,
            )

            # Oracle
            oracle_bm, oracle_ms, oracle_all = time_expert_oracle(
                x, w["w_gate"], w["w_up"], w["w_down"],
                args.warmup, args.iters,
            )

            dist_base_total   += base_ms
            dist_remo_total   += remo_ms
            dist_oracle_total += oracle_ms

            print(f"  {dist_name:>13}  {slot:4d}  {F_e:6d}  {T_e:4d}  "
                  f"{BASELINE_BLOCK_M:7d}  {base_ms:8.3f}  "
                  f"{remo_bm:7d}  {remo_ms:8.3f}  "
                  f"{oracle_bm:7d}  {oracle_ms:8.3f}")

            ext_rows.append({
                "distribution":     dist_name,
                "slot":             slot,
                "intermediate_dim": F_e,
                "cost_class":       cost,
                "tokens_per_expert": T_e,
                "baseline_block_m": BASELINE_BLOCK_M,
                "baseline_ms":      base_ms,
                "remora_block_m":   remo_bm,
                "remora_ms":        remo_ms,
                "oracle_block_m":   oracle_bm,
                "oracle_ms":        oracle_ms,
                "oracle_bm16_ms":   oracle_all.get(16, float("nan")),
                "oracle_bm32_ms":   oracle_all.get(32, float("nan")),
                "oracle_bm64_ms":   oracle_all.get(64, float("nan")),
                "oracle_bm128_ms":  oracle_all.get(128, float("nan")),
            })

        print(sep)
        remo_vs_base   = (dist_base_total / dist_remo_total
                          if dist_remo_total > 0 else float("nan"))
        oracle_vs_base = (dist_base_total / dist_oracle_total
                          if dist_oracle_total > 0 else float("nan"))
        print(f"  {'TOTAL':>13}  {'':>4}  {'':>6}  {'':>4}  "
              f"{'':>7}  {dist_base_total:8.3f}  "
              f"{'':>7}  {dist_remo_total:8.3f}  "
              f"{'':>7}  {dist_oracle_total:8.3f}")
        print(f"  remora_vs_baseline={remo_vs_base:.2f}x  "
              f"oracle_vs_baseline={oracle_vs_base:.2f}x")

        ext_summaries.append({
            "distribution":         dist_name,
            "tokens":               tokens_list,
            "baseline_total_ms":    dist_base_total,
            "remora_total_ms":      dist_remo_total,
            "oracle_total_ms":      dist_oracle_total,
            "remora_vs_baseline":   remo_vs_base,
            "oracle_vs_baseline":   oracle_vs_base,
        })

    # --- Summary table ---
    print()
    print("=" * 80)
    print("Summary")
    print(f"  {'distribution':>13}  {'baseline ms':>11}  {'remora ms':>9}  "
          f"{'oracle ms':>9}  {'remora/base':>11}  {'oracle/base':>11}")
    print("  " + "-" * 70)
    for s in ext_summaries:
        print(f"  {s['distribution']:>13}  "
              f"{s['baseline_total_ms']:11.3f}  "
              f"{s['remora_total_ms']:9.3f}  "
              f"{s['oracle_total_ms']:9.3f}  "
              f"{s['remora_vs_baseline']:10.2f}x  "
              f"{s['oracle_vs_baseline']:10.2f}x")
    print("=" * 80)

    # --- Save extended results ---
    os.makedirs(RESULTS, exist_ok=True)
    ext_meta = {
        "gpu":               torch.cuda.get_device_name(0),
        "hidden_dim":        D,
        "total_tokens":      TOKENS_PER_EXPERT,
        "warmup_iters":      args.warmup,
        "timed_iters":       args.iters,
        "baseline_block_m":  BASELINE_BLOCK_M,
        "oracle_block_n":    ORACLE_BLOCK_N,
        "oracle_block_k":    ORACLE_BLOCK_K,
        "oracle_candidates": ORACLE_BM_CANDIDATES,
        "decisions_file":    os.path.basename(args.decisions),
        "distributions":     {k: v for k, v in DISTRIBUTIONS.items()},
    }
    out_json_ext = os.path.join(RESULTS, "sweep_heterogeneous_proportional.json")
    with open(out_json_ext, "w") as f:
        json.dump(
            {"meta": ext_meta, "results": ext_rows, "summary": ext_summaries},
            f, indent=2,
        )
        f.write("\n")

    out_csv_ext = os.path.join(RESULTS, "sweep_heterogeneous_proportional.csv")
    csv_fields = [
        "distribution", "slot", "intermediate_dim", "cost_class",
        "tokens_per_expert",
        "baseline_block_m", "baseline_ms",
        "remora_block_m",   "remora_ms",
        "oracle_block_m",   "oracle_ms",
        "oracle_bm16_ms", "oracle_bm32_ms", "oracle_bm64_ms", "oracle_bm128_ms",
    ]
    with open(out_csv_ext, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ext_rows)

    print()
    print(f"Extended results → {out_json_ext}")
    print(f"                   {out_csv_ext}")


if __name__ == "__main__":
    main()
