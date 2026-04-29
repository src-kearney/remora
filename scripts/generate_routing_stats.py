#!/usr/bin/env python3
"""
scripts/generate_routing_stats.py

Simulate per-expert token routing and emit a routing_stats.json file for use
with emit_compiler_decisions.py --routing-stats.

Simulates routing by drawing multinomial samples: each of `total-tokens` tokens
is assigned to an expert according to the distribution weights.  Aggregates
mean, p50, and p95 across `num-samples` draws.

Distributions:
  uniform      — equal probability for all experts
  proportional — probability ∝ expert intermediate dim F (14336, 8192, 4096, 2048)
  skewed       — Zipf 1/rank: expert i gets probability ∝ 1/(i+1)

Output format (routing_stats.json):
  {
    "distribution":   "proportional",
    "num_samples":    1000,
    "num_experts":    8,
    "total_tokens":   512,
    "expert_token_counts": {
      "expert_0": {"mean": 128.1, "p50": 128, "p95": 141},
      ...
    }
  }

Usage:
    python3 scripts/generate_routing_stats.py \\
        --distribution proportional \\
        --num-experts 8 \\
        --total-tokens 512 \\
        --num-samples 1000 \\
        --output routing_stats_proportional.json
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Expert layout must match heterogeneous_moe_layer.mlir
_EXPERT_F = [14336, 14336, 8192, 8192, 4096, 4096, 2048, 2048]


# ---------------------------------------------------------------------------
# Distribution weights
# ---------------------------------------------------------------------------

def _weights_uniform(num_experts: int) -> list[float]:
    return [1.0 / num_experts] * num_experts


def _weights_proportional(num_experts: int) -> list[float]:
    """Probability ∝ intermediate dim F.  Uses _EXPERT_F for first 8 experts."""
    F_vals = _EXPERT_F[:num_experts]
    if len(F_vals) < num_experts:
        # Extend with the last known F if requested E > 8
        F_vals += [F_vals[-1]] * (num_experts - len(F_vals))
    tot = sum(F_vals)
    return [F / tot for F in F_vals]


def _weights_skewed(num_experts: int) -> list[float]:
    """Zipf 1/rank weights: expert i gets weight 1/(i+1)."""
    weights = [1.0 / (i + 1) for i in range(num_experts)]
    tot = sum(weights)
    return [w / tot for w in weights]


_WEIGHT_FNS = {
    "uniform":      _weights_uniform,
    "proportional": _weights_proportional,
    "skewed":       _weights_skewed,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_routing(
    distribution: str,
    num_experts: int,
    total_tokens: int,
    num_samples: int,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Draw ``num_samples`` multinomial routing samples.
    Returns a dict mapping "expert_N" → {mean, p50, p95}.
    """
    if distribution not in _WEIGHT_FNS:
        sys.exit(f"error: unknown distribution '{distribution}'. "
                 f"Choose from: {list(_WEIGHT_FNS)}")

    rng     = np.random.default_rng(seed)
    weights = np.array(_WEIGHT_FNS[distribution](num_experts))

    # Each sample: multinomial draw of total_tokens tokens → num_experts bins
    counts = rng.multinomial(total_tokens, weights, size=num_samples)
    # counts shape: (num_samples, num_experts)

    result: dict[str, dict] = {}
    for e in range(num_experts):
        expert_counts = counts[:, e].astype(float)
        result[f"expert_{e}"] = {
            "mean": float(np.mean(expert_counts)),
            "p50":  int(np.percentile(expert_counts, 50)),
            "p95":  int(np.percentile(expert_counts, 95)),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-expert routing statistics from a simulated distribution."
    )
    parser.add_argument(
        "--distribution",
        required=True,
        choices=list(_WEIGHT_FNS),
        help="Token distribution to simulate.",
    )
    parser.add_argument(
        "--num-experts", type=int, default=8,
        help="Number of experts (default: 8)",
    )
    parser.add_argument(
        "--total-tokens", type=int, default=512,
        help="Total tokens per forward pass (default: 512)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of forward-pass samples to draw (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for routing_stats.json",
    )
    args = parser.parse_args()

    print(f"Simulating {args.num_samples} routing samples "
          f"({args.distribution}, E={args.num_experts}, T={args.total_tokens}) ...")

    expert_counts = simulate_routing(
        distribution=args.distribution,
        num_experts=args.num_experts,
        total_tokens=args.total_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    out = {
        "distribution":         args.distribution,
        "num_samples":          args.num_samples,
        "num_experts":          args.num_experts,
        "total_tokens":         args.total_tokens,
        "expert_token_counts":  expert_counts,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Wrote routing stats to {args.output}")
    print(f"  {'expert':>8}  {'mean':>7}  {'p50':>5}  {'p95':>5}")
    print("  " + "-" * 34)
    for key, stats in expert_counts.items():
        print(f"  {key:>8}  {stats['mean']:7.1f}  {stats['p50']:5d}  {stats['p95']:5d}")


if __name__ == "__main__":
    main()
