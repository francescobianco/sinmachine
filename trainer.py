#!/usr/bin/env python3
"""SinMachine Trainer — bilevel inverse optimisation.

Given a dataset of (question, answer) pairs, optimises the harmonic model
parameters so that, for each question, the phase search finds a coherent
entry point and the trajectory continuation produces the expected answer.

Bilevel formulation:
    inner:  φ_q* = argmin_φ  d(generate(φ, |q|), q)     [phase search]
    outer:  L    = Σ d(generate_continuation(φ_q*, |a|), a)  [answer loss]

This replaces the old approach of encoding the question via SHA256, which
broke the internal coherence of the system (see docs/02-encoder-problem.md).
"""

import json
import math
import pathlib
import random
import sys

from sinmachine import char_to_y, VOCAB, _VOCAB_SIZE, MODELS_DIR

_SEP = "─" * 60


# ── low-level simulation with explicit parameters ─────────────────

def _simulate_params(t0: float, phi0: float,
                     harmonics: list, amp_total: float,
                     dt: float, phase_feedback: float,
                     steps: int) -> tuple[list[float], float, float]:
    """Run sampler with raw parameters. Returns (y_values, t_final, phi_final)."""
    phi = phi0
    t = t0
    ys = []
    for _ in range(steps):
        y = sum(a * math.sin(omega * t + phi) for omega, a in harmonics) / amp_total
        idx = int((y + 1) / 2 * _VOCAB_SIZE)
        idx = max(0, min(_VOCAB_SIZE - 1, idx))
        ys.append(y)
        phi += phase_feedback * (idx / _VOCAB_SIZE) * 2 * math.pi
        t += dt
    return ys, t, phi


# ── inner optimisation: phase search ─────────────────────────────

def _find_phase(target: str,
                harmonics: list, amp_total: float,
                dt: float, phase_feedback: float,
                resolution: int = 500) -> float:
    """Grid search for the phase from which the target string would emerge."""
    targets_y = [char_to_y(c) for c in target]
    best_phi, best_loss = 0.0, float("inf")

    for i in range(resolution):
        phi = (i / resolution) * 2 * math.pi
        ys, _, _ = _simulate_params(0.0, phi, harmonics, amp_total, dt, phase_feedback, len(target))
        loss = sum((y - ty) ** 2 for y, ty in zip(ys, targets_y)) / len(targets_y)
        if loss < best_loss:
            best_loss = loss
            best_phi = phi

    return best_phi


# ── outer loss ────────────────────────────────────────────────────

def bilevel_loss(params: list[float], n_harmonics: int,
                 dataset: list[tuple[str, str]],
                 search_resolution: int = 500) -> float:
    """Bilevel loss: search phase for each question, measure answer continuation loss.

    The phase search (inner) is approximate but sufficient for gradient-free
    outer optimisation via hill climbing or Nelder-Mead.
    """
    harmonics = [(abs(params[i*2]), abs(params[i*2+1])) for i in range(n_harmonics)]
    amp_total = sum(a for _, a in harmonics) or 1e-9
    dt = abs(params[n_harmonics*2]) + 1e-6
    phase_feedback = params[n_harmonics*2 + 1]

    total, count = 0.0, 0
    for q, a in dataset:
        phi_q = _find_phase(q, harmonics, amp_total, dt, phase_feedback, search_resolution)
        _, t_end, phi_end = _simulate_params(0.0, phi_q, harmonics, amp_total, dt, phase_feedback, len(q))
        ys_a, _, _ = _simulate_params(t_end, phi_end, harmonics, amp_total, dt, phase_feedback, len(a))
        for y_gen, ch in zip(ys_a, a):
            total += (y_gen - char_to_y(ch)) ** 2
            count += 1
    return total / max(count, 1)


# ── optimisers ────────────────────────────────────────────────────

def hill_climb(x0: list[float], n_harmonics: int,
               dataset: list[tuple[str, str]],
               iterations: int = 3000,
               search_resolution: int = 500) -> tuple[list[float], float]:
    x = list(x0)
    best_loss = bilevel_loss(x, n_harmonics, dataset, search_resolution)
    scale = 0.1

    for i in range(iterations):
        candidate = [v + random.gauss(0, scale) for v in x]
        loss = bilevel_loss(candidate, n_harmonics, dataset, search_resolution)
        if loss < best_loss:
            best_loss = loss
            x = candidate
            scale = min(scale * 1.1, 1.0)
        else:
            scale = max(scale * 0.998, 1e-5)

        if (i + 1) % 500 == 0:
            print(f"    iter {i+1:4d}  loss={best_loss:.6f}  scale={scale:.5f}")

    return x, best_loss


def scipy_optimise(x0: list[float], n_harmonics: int,
                   dataset: list[tuple[str, str]],
                   search_resolution: int = 500) -> tuple[list[float], float]:
    from scipy.optimize import minimize

    def objective(x):
        return bilevel_loss(list(x), n_harmonics, dataset, search_resolution)

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 8000, "xatol": 1e-5, "fatol": 1e-5, "disp": False})
    return list(result.x), result.fun


# ── pack / unpack parameters ──────────────────────────────────────

def pack(harmonics: list, dt: float, phase_feedback: float) -> list[float]:
    flat = []
    for omega, amp in harmonics:
        flat.extend([omega, amp])
    flat.extend([dt, phase_feedback])
    return flat


def unpack(params: list[float], n: int) -> tuple[list, float, float]:
    harmonics = [[abs(params[i*2]), abs(params[i*2+1])] for i in range(n)]
    dt = abs(params[n*2]) + 1e-6
    phase_feedback = params[n*2 + 1]
    return harmonics, dt, phase_feedback


# ── main training loop ────────────────────────────────────────────

def train(base_model_name: str, dataset_path: str, output_name: str,
          search_resolution: int = 500) -> None:
    base_path = MODELS_DIR / f"{base_model_name}.json"
    with open(base_path) as f:
        base = json.load(f)
    harmonics = [tuple(h) for h in base["harmonics"]]
    n = len(harmonics)
    steps = base["steps"]

    dataset: list[tuple[str, str]] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                dataset.append((obj["q"], obj["a"]))

    print(f"  Base model  : {base_model_name}  ({n} harmonics, {steps} steps)")
    print(f"  Dataset     : {len(dataset)} pairs  [{dataset_path}]")
    print(f"  Phase search: {search_resolution} grid points per pair per iteration")

    x0 = pack(harmonics, base["dt"], base["phase_feedback"])
    initial_loss = bilevel_loss(x0, n, dataset, search_resolution)
    print(f"  Initial loss: {initial_loss:.6f}")
    print()

    try:
        import scipy  # noqa: F401
        print("  Optimiser: scipy Nelder-Mead")
        x_opt, final_loss = scipy_optimise(x0, n, dataset, search_resolution)
    except ImportError:
        print("  Optimiser: hill climbing  (install scipy for better results)")
        x_opt, final_loss = hill_climb(x0, n, dataset, search_resolution=search_resolution)

    print(f"\n  Final loss  : {final_loss:.6f}  (Δ={final_loss - initial_loss:+.6f})")

    opt_harmonics, opt_dt, opt_feedback = unpack(x_opt, n)
    out_model = {
        "name": output_name,
        "description": f"bilevel-trained on '{dataset_path}' from base '{base_model_name}'",
        "harmonics": [[round(o, 6), round(a, 6)] for o, a in opt_harmonics],
        "dt": round(opt_dt, 6),
        "phase_feedback": round(opt_feedback, 6),
        "steps": steps,
    }
    out_path = MODELS_DIR / f"{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_model, f, indent=2)
    print(f"  Saved → {out_path}")

    print(f"\n  Optimised harmonics:")
    for i, (omega, amp) in enumerate(opt_harmonics):
        orig_omega, orig_amp = harmonics[i]
        print(f"    [{i}]  ω={omega:.4f} (was {orig_omega:.4f})  A={amp:.4f} (was {orig_amp:.4f})")
    print(f"  dt            : {opt_dt:.6f}  (was {base['dt']})")
    print(f"  phase_feedback: {opt_feedback:.6f}  (was {base['phase_feedback']})")


# ── entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SinMachine bilevel inverse trainer")
    parser.add_argument("dataset", help="path to .jsonl dataset file")
    parser.add_argument("--base", default="default", help="base model name (default: default)")
    parser.add_argument("--output", default="trained", help="output model name (default: trained)")
    parser.add_argument("--resolution", type=int, default=500,
                        help="phase search grid resolution per pair (default: 500)")
    args = parser.parse_args()

    print(f"\n{_SEP}")
    print("  SinMachine Trainer  —  bilevel inverse optimisation")
    print(_SEP)
    train(args.base, args.dataset, args.output, args.resolution)
    print(_SEP + "\n")
