#!/usr/bin/env python3
"""
benchmarks/sweep_concentration.py

Sweeps routing concentration — the fraction of token-slot assignments going
to the top expert — from 0.125 (uniform) to 1.0 (single expert) in
fine-grained steps, comparing vLLM fused_moe vs Remora per-expert dispatch.

Concentration model
-------------------
For concentration c, expert 0 is assigned probability c; the remaining
(1-c) is split uniformly across experts 1-7.  topk_ids are sampled via
torch.multinomial without replacement; topk_weights are the softmax-
normalized selected probabilities (rows sum to 1).

c=1.0 is handled as a special case (multinomial degenerates) using the
same fixed routing as the "single" distribution in sweep_skew.py.

Concentration levels
--------------------
Fine-grained in [0.75, 1.0] because that is where the crossover is
expected based on sweep_skew results.

Output
------
  stdout                              — table + ASCII speedup plot + crossover
  benchmarks/results/sweep_concentration.json
  benchmarks/results/sweep_concentration.csv
"""

import csv
import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Sibling-module imports — reuse remora_forward and fused_moe from sweep_skew
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# sweep_skew exposes remora_forward, fused_moe, and the shared constants at
# module level; importing it also pre-imports vLLM and triton_expert_kernel.
from sweep_skew import (
    remora_forward,
    fused_moe,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    NUM_EXPERTS,
    TOP_K,
    NUM_TOKENS,
    DTYPE,
    WARMUP_ITERS,
    TIMED_ITERS,
)
from triton_expert_kernel import warmup_all_buckets

# ---------------------------------------------------------------------------
# Concentration levels to sweep
# ---------------------------------------------------------------------------
CONCENTRATIONS: list[float] = [
    0.125,    # uniform (1/8)
    0.25,
    0.375,
    0.50,
    0.625,
    0.70,
    0.75,
    0.80,
    0.85,
    0.875,
    0.90,
    0.925,
    0.9375,
    0.95,
    0.9625,
    0.975,
    0.9875,
    1.0,      # single
]

# Winner thresholds
WIN_REMORA = 1.05
WIN_VLLM   = 0.95


# ---------------------------------------------------------------------------
# Routing distribution for a given concentration
# ---------------------------------------------------------------------------

def _make_concentration(
    c:           float,
    num_tokens:  int,
    num_experts: int,
    top_k:       int,
    dtype:       torch.dtype,
    device:      str,
):
    """
    Expert 0 has probability c; experts 1..E-1 share (1-c)/(E-1) each.
    topk_weights are softmax-normalized over the selected experts so rows
    sum to 1.

    c=1.0: multinomial would degenerate (only 1 non-zero prob); handled
    explicitly as the "single" distribution (slot 0 → expert 0 weight 1,
    slot 1 → expert 1 weight 0).
    """
    if c >= 1.0:
        ids = torch.stack([
            torch.zeros(num_tokens, dtype=torch.int32, device=device),
            torch.ones( num_tokens, dtype=torch.int32, device=device),
        ], dim=1)
        weights = torch.stack([
            torch.ones( num_tokens, dtype=dtype, device=device),
            torch.zeros(num_tokens, dtype=dtype, device=device),
        ], dim=1)
        return ids.contiguous(), weights.contiguous()

    p    = torch.full((num_experts,), (1.0 - c) / (num_experts - 1),
                      dtype=torch.float32, device=device)
    p[0] = c
    probs = p.unsqueeze(0).expand(num_tokens, -1).contiguous()

    ids  = torch.multinomial(probs, num_samples=top_k, replacement=False).to(torch.int32)
    sel  = torch.gather(probs, 1, ids.long())
    weights = (sel / sel.sum(dim=-1, keepdim=True)).to(dtype)
    return ids.contiguous(), weights.contiguous()


# ---------------------------------------------------------------------------
# Timing helper (shared pattern with sweep_skew)
# ---------------------------------------------------------------------------

def _timed(fn, warmup: int, timed: int) -> float:
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(timed):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev))

    return float(np.median(latencies))


# ---------------------------------------------------------------------------
# ASCII speedup plot
# ---------------------------------------------------------------------------

def _ascii_plot(concentrations: list[float], speedups: list[float]) -> str:
    """
    Return a multi-line ASCII chart of speedup vs concentration.
    The horizontal reference line is drawn at speedup=1.0.
    """
    WIDTH  = 62
    HEIGHT = 16
    Y_MIN  = 0.70
    Y_MAX  = 2.10

    def to_row(spd: float) -> int:
        r = int((Y_MAX - spd) / (Y_MAX - Y_MIN) * (HEIGHT - 1) + 0.5)
        return max(0, min(HEIGHT - 1, r))

    def to_col(i: int) -> int:
        c = int(i / max(len(concentrations) - 1, 1) * (WIDTH - 1) + 0.5)
        return max(0, min(WIDTH - 1, c))

    grid: list[list[str]] = [[" "] * WIDTH for _ in range(HEIGHT)]

    # Reference line at speedup = 1.0
    ref_row = to_row(1.0)
    for col in range(WIDTH):
        grid[ref_row][col] = "-"

    # Plot points (drawn after reference so stars overwrite dashes)
    for i, spd in enumerate(speedups):
        grid[to_row(spd)][to_col(i)] = "*"

    lines: list[str] = ["speedup vs concentration"]
    for r in range(HEIGHT):
        spd_at_row = Y_MAX - r * (Y_MAX - Y_MIN) / (HEIGHT - 1)
        # Label every other row
        if abs(round(spd_at_row * 4) - spd_at_row * 4) < 0.01:
            label = f"{spd_at_row:4.2f} |"
        else:
            label = "     |"
        lines.append(label + "".join(grid[r]))

    lines.append("     +" + "-" * WIDTH + ">")
    pad = WIDTH - 4
    lines.append(f"     {'0.125':<{pad}}{'1.0':>4}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    gpu    = torch.cuda.get_device_name(0)

    print("=" * 76)
    print("remora  sweep_concentration  —  vLLM vs Remora across routing concentration")
    print(f"GPU:          {gpu}")
    print(f"Mixtral dims: hidden={HIDDEN_DIM}, intermediate={INTERMEDIATE_DIM}, "
          f"experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"Batch size:   {NUM_TOKENS} tokens")
    print(f"Timing:       {WARMUP_ITERS} warmup + {TIMED_ITERS} timed iters, median")
    print("=" * 76)
    print()

    # ---- Weights -----------------------------------------------------------
    torch.manual_seed(0)
    scale = 0.02

    w1_vllm = (torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w2_vllm = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()

    w_gate  = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w_up    = (torch.randn(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()
    w_down  = (torch.randn(NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM,
                           dtype=DTYPE, device=device) * scale).contiguous()

    # ---- Pre-compile Triton buckets ----------------------------------------
    warmup_all_buckets(hidden_dim=HIDDEN_DIM, inter_dim=INTERMEDIATE_DIM, device=device)

    # ---- Header ------------------------------------------------------------
    print(
        f"{'concentration':>13}  {'top_expert':>10}  "
        f"{'vllm_ms':>8}  {'remora_ms':>9}  {'speedup':>7}  {'winner':<7}"
    )
    print("-" * 68)

    # ---- Sweep -------------------------------------------------------------
    results: dict[str, dict] = {}
    speedup_list: list[float] = []

    for c in CONCENTRATIONS:
        topk_ids, topk_weights = _make_concentration(
            c, NUM_TOKENS, NUM_EXPERTS, TOP_K, DTYPE, device
        )
        hidden_states = (
            torch.randn(NUM_TOKENS, HIDDEN_DIM, dtype=DTYPE, device=device) * 0.1
        ).contiguous()

        top_expert_tokens = int(
            (topk_ids == 0).sum().item()
        )
        zero_expert_count = int(
            (topk_ids.flatten().bincount(minlength=NUM_EXPERTS) == 0).sum().item()
        )
        tpe = topk_ids.flatten().bincount(minlength=NUM_EXPERTS).cpu().tolist()

        vllm_ms = _timed(
            lambda: fused_moe(
                hidden_states, w1_vllm, w2_vllm, topk_weights, topk_ids, inplace=False
            ),
            WARMUP_ITERS, TIMED_ITERS,
        )
        remora_ms = _timed(
            lambda: remora_forward(
                hidden_states, w_gate, w_up, w_down, topk_ids, topk_weights
            ),
            WARMUP_ITERS, TIMED_ITERS,
        )

        speedup = vllm_ms / remora_ms
        speedup_list.append(speedup)

        if speedup > WIN_REMORA:
            winner = "remora"
        elif speedup < WIN_VLLM:
            winner = "vllm"
        else:
            winner = "~tie"

        results[f"{c:.4f}"] = {
            "concentration":     c,
            "top_expert_tokens": top_expert_tokens,
            "zero_expert_count": zero_expert_count,
            "tokens_per_expert": tpe,
            "vllm_ms":           round(vllm_ms,   4),
            "remora_ms":         round(remora_ms,  4),
            "speedup":           round(speedup,    3),
            "winner":            winner,
        }

        print(
            f"  {c:11.4f}  {top_expert_tokens:>10d}  "
            f"{vllm_ms:>7.3f}ms  {remora_ms:>8.3f}ms  "
            f"{speedup:>6.2f}x  {winner:<7}"
        )

    print()

    # ---- Crossover ---------------------------------------------------------
    crossover_c    = None
    crossover_data = None
    for c, key in zip(CONCENTRATIONS, results):
        r = results[key]
        if r["winner"] == "remora":
            crossover_c    = c
            crossover_data = r
            break

    if crossover_c is not None:
        print(
            f"Crossover: Remora wins above concentration={crossover_c:.4f}\n"
            f"           (top expert receives {crossover_data['top_expert_tokens']} / "
            f"{NUM_TOKENS * TOP_K} token-slots, "
            f"{crossover_data['zero_expert_count']} of {NUM_EXPERTS} experts receive 0 tokens)"
        )
    else:
        print("Crossover: Remora never wins within the swept range.")
    print()

    # ---- ASCII plot --------------------------------------------------------
    print(_ascii_plot(CONCENTRATIONS, speedup_list))
    print()

    # ---- Save JSON ---------------------------------------------------------
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "gpu":              gpu,
        "hidden_dim":       HIDDEN_DIM,
        "intermediate_dim": INTERMEDIATE_DIM,
        "num_experts":      NUM_EXPERTS,
        "top_k":            TOP_K,
        "batch_size":       NUM_TOKENS,
        "dtype":            str(DTYPE),
        "warmup_iters":     WARMUP_ITERS,
        "timed_iters":      TIMED_ITERS,
        "win_remora_threshold": WIN_REMORA,
        "win_vllm_threshold":   WIN_VLLM,
    }
    json_path = os.path.join(out_dir, "sweep_concentration.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"JSON saved to {os.path.relpath(json_path)}")

    # ---- Save CSV ----------------------------------------------------------
    csv_path   = os.path.join(out_dir, "sweep_concentration.csv")
    fieldnames = [
        "concentration", "top_expert_tokens", "zero_expert_count",
        "vllm_ms", "remora_ms", "speedup", "winner",
    ] + [f"e{i}" for i in range(NUM_EXPERTS)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, r in results.items():
            row: dict = {
                "concentration":     r["concentration"],
                "top_expert_tokens": r["top_expert_tokens"],
                "zero_expert_count": r["zero_expert_count"],
                "vllm_ms":           r["vllm_ms"],
                "remora_ms":         r["remora_ms"],
                "speedup":           r["speedup"],
                "winner":            r["winner"],
            }
            for i, cnt in enumerate(r["tokens_per_expert"]):
                row[f"e{i}"] = cnt
            writer.writerow(row)
    print(f"CSV  saved to {os.path.relpath(csv_path)}")


if __name__ == "__main__":
    main()
