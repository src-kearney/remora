#!/usr/bin/env python3
"""
Dump XLA's compiled HLO for simple_moe to see what it does with per-expert boundaries.

Usage:
    python3 scripts/export/dump_xla_hlo.py

Output lands in txt/xla_dump/. Look for the *after_optimizations* file.
Search for `dot` — if it's one batched dot with no per-expert func splits, XLA
erases expert boundaries just like JAX's StableHLO export does.
"""
import os
import glob

DUMP_DIR = os.path.join(os.path.dirname(__file__), "../../txt/xla_dump")
os.makedirs(DUMP_DIR, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_to={DUMP_DIR} --xla_dump_hlo_as_text"

import jax
import jax.numpy as jnp


def silu(x):
    return x * jax.nn.sigmoid(x)


def moe_layer(tokens, router_w, w_gate, w_up, w_down):
    T, D = tokens.shape
    E = w_gate.shape[0]

    logits = tokens @ router_w
    expert_idx = jnp.argmax(logits, axis=-1)
    dispatch = jax.nn.one_hot(expert_idx, num_classes=E)

    dispatched = jnp.einsum('te,td->etd', dispatch, tokens)

    gate = jnp.einsum('etd,edf->etf', dispatched, w_gate)
    up   = jnp.einsum('etd,edf->etf', dispatched, w_up)
    hidden = silu(gate) * up
    expert_out = jnp.einsum('etf,efd->etd', hidden, w_down)

    output = jnp.einsum('te,etd->td', dispatch, expert_out)
    return output


T, D, E, F = 8, 32, 2, 64

# .lower() produces StableHLO; .compile() hands it to XLA and triggers the dump.
lowered = jax.jit(moe_layer).lower(
    jnp.zeros((T, D)),
    jnp.zeros((D, E)),
    jnp.zeros((E, D, F)),
    jnp.zeros((E, D, F)),
    jnp.zeros((E, F, D)),
)
lowered.compile()

print("XLA dump written to txt/xla_dump/")
print()

files = sorted(glob.glob(os.path.join(DUMP_DIR, "*.txt")))
if not files:
    print("No .txt files found — check txt/xla_dump/ manually.")
else:
    # Find the jit_moe_layer after_optimizations file specifically.
    target = next(
        (f for f in files if "jit_moe_layer" in f and "after_optimizations" in f
         and "buffer-assignment" not in f and "memory-usage" not in f),
        None
    )
    if target is None:
        print("Could not find jit_moe_layer after_optimizations file. Files found:")
        for f in files:
            print(f"  {f}")
    else:
        print(f"Reading: {target}")
        print("---")
        with open(target) as f:
            content = f.read()
        print(content)
