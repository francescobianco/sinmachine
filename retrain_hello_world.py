#!/usr/bin/env python3
"""Direct optimizer for hello-world with phase_feedback=0.

Without phase feedback the trajectory is a pure sum of sinusoids —
smooth, underdetermined (18 params vs 11 targets), analytically solvable.
"""

import json, math, sys
import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from sinmachine import (build_default_perm, y_center, char_to_idx,
                        _VOCAB_SIZE, MODELS_DIR, _TOKEN_DISPLAY)

PERM = build_default_perm(multi_end=True)
Q, A = "hello", "world\x03"
FULL = Q + A
N = len(FULL)   # 11
N_HARM = 8

TARGETS = np.array([y_center(char_to_idx(c, PERM)) for c in FULL])


def simulate(phi, omegas, amps, dt):
    """No phase feedback — pure sinusoidal trajectory."""
    t = np.arange(N, dtype=float) * dt
    amp_total = np.sum(amps)
    if amp_total < 1e-9:
        return np.zeros(N)
    y = np.zeros(N)
    for omega, a in zip(omegas, amps):
        y += a * np.sin(omega * t + phi)
    return y / amp_total


def unpack(x):
    phi    = x[0]
    omegas = np.abs(x[1:1 + N_HARM * 2:2])
    amps   = np.abs(x[2:2 + N_HARM * 2:2])
    dt     = abs(x[-1]) + 1e-6
    return phi, omegas, amps, dt


def loss(x):
    phi, omegas, amps, dt = unpack(x)
    y = simulate(phi, omegas, amps, dt)
    return float(np.mean((y - TARGETS) ** 2))


def decode(y_vals):
    out = []
    for y in y_vals:
        idx = max(0, min(_VOCAB_SIZE - 1, int((y + 1) / 2 * _VOCAB_SIZE)))
        c = PERM[idx]
        out.append(_TOKEN_DISPLAY.get(c, c))
    return "".join(out)


def display(text):
    return "".join(_TOKEN_DISPLAY.get(c, c) for c in text)


# ── bounds ──────────────────────────────────────────────────────────
bounds = [(0, 2 * math.pi)]           # phi
for _ in range(N_HARM):
    bounds += [(0.1, 30.0), (0.001, 10.0)]   # omega, amplitude
bounds += [(0.01, 2.0)]               # dt

SEP = "─" * 64
print(SEP)
print("  hello-world retrain  (phase_feedback=0, smooth landscape)")
print(SEP)
print(f"  Target : {display(FULL)!r}")
print(f"  Targets: {[round(v, 4) for v in TARGETS.tolist()]}")
print()

best_loss = float("inf")
best_x    = None

# ── Phase 1: differential evolution (many seeds) ────────────────────
print("  Phase 1 — differential evolution")
for seed in range(60):
    r = differential_evolution(
        loss, bounds,
        maxiter=5000, tol=1e-12, seed=seed,
        popsize=20, mutation=(0.3, 1.9), recombination=0.9,
        init="sobol", polish=True, workers=1, updating="deferred",
    )
    if r.fun < best_loss:
        best_loss = r.fun
        best_x    = r.x.copy()
        phi, omegas, amps, dt = unpack(best_x)
        got = decode(simulate(phi, omegas, amps, dt))
        want = display(FULL)
        print(f"  ★ seed={seed:2d}  loss={best_loss:.8f}  got={got!r}  want={want!r}")
    else:
        print(f"    seed={seed:2d}  loss={r.fun:.8f}")
    if best_loss < 1e-8:
        print("  EXACT MATCH — stopping early")
        break

# ── Phase 2: Nelder-Mead refinement from best DE ────────────────────
print()
print("  Phase 2 — Nelder-Mead refinement")
r = minimize(loss, best_x, method="Nelder-Mead",
             options={"maxiter": 200_000, "xatol": 1e-12, "fatol": 1e-12})
if r.fun < best_loss:
    best_loss = r.fun
    best_x    = r.x.copy()
print(f"  NM loss  : {best_loss:.10f}")

# ── Phase 3: random restart Nelder-Mead ─────────────────────────────
if best_loss > 1e-8:
    print()
    print("  Phase 3 — random-restart Nelder-Mead (10 000 trials)")
    rng = np.random.default_rng(999)
    for trial in range(10_000):
        phi0   = rng.uniform(0, 2 * math.pi)
        omegas0 = rng.uniform(0.1, 25, N_HARM)
        amps0   = rng.uniform(0.001, 5, N_HARM)
        dt0    = rng.uniform(0.02, 1.5)
        x0 = np.array([phi0] + [v for o, a in zip(omegas0, amps0) for v in (o, a)] + [dt0])
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10})
        if r.fun < best_loss:
            best_loss = r.fun
            best_x    = r.x.copy()
            phi, omegas, amps, dt = unpack(best_x)
            got = decode(simulate(phi, omegas, amps, dt))
            print(f"  ★ trial={trial:5d}  loss={best_loss:.10f}  got={got!r}")
        if best_loss < 1e-8:
            print("  EXACT MATCH!")
            break

# ── Results ─────────────────────────────────────────────────────────
phi, omegas, amps, dt = unpack(best_x)
y_vals = simulate(phi, omegas, amps, dt)
got    = decode(y_vals)
want   = display(FULL)

print()
print(SEP)
print(f"  Final loss : {best_loss:.10f}")
print(f"  Want       : {want!r}")
print(f"  Got        : {got!r}")
print(f"  phi        : {phi:.6f} rad ({math.degrees(phi):.1f}°)")
print(f"  dt         : {dt:.6f}")
n_correct = sum(a == b for a, b in zip(got, want))
print(f"  Char match : {n_correct}/{N}")

# Serialise perm
perm_serial = [_TOKEN_DISPLAY.get(c, c) for c in PERM]

out = {
    "name":           "hello-world",
    "description":    "joint-trained on 'datasets/hello-world.jsonl', phase_feedback=0",
    "harmonics":      [[round(float(o), 6), round(float(a), 6)]
                       for o, a in zip(omegas, amps)],
    "dt":             round(float(dt), 6),
    "phase_feedback": 0.0,
    "steps":          30,
    "multi_end":      True,
    "perm":           perm_serial,
    "_training_phi":  round(float(phi), 6),
}

out_path = MODELS_DIR / "hello-world.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved → {out_path}")
print(SEP)