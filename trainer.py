#!/usr/bin/env python3
"""SinMachine Trainer — three optimisation modes.

Bilevel:
    inner:  φ_q* = argmin_φ  d(generate(φ, |q|), q)     [phase search]
    outer:  L    = Σ d(generate_continuation(φ_q*, |a|), a)  [answer loss]

Joint (single-pair only):
    Optimise (φ, model_params) simultaneously for one pair.

Multi-joint (correct multi-pair extension of joint):
    Optimise (φ_0..φ_N, model_params) simultaneously — one φ per pair.
    No bilevel inner loop. O(N·steps) per evaluation.

Stream:
    Concatenate all QA pairs into one long sequence. Find ONE φ from which
    the entire stream emerges. Inference by scanning the generated text,
    not by phase search.
"""

import json
import math
import pathlib
import random
import sys

from sinmachine import (char_to_y, VOCAB, _VOCAB_SIZE, MODELS_DIR,
                        _END_CHAR, _MULTI_END_ZONES, _DEFAULT_END_ZONES)

_SEP = "─" * 60


def _end_pos_loss(y_gen: float, end_zones: list) -> float:
    """Training loss for an END position: distance to nearest zone center."""
    return min((y_gen - ctr) ** 2 for ctr, _ in end_zones)


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


# ── joint optimisation (single-pair shortcut) ────────────────────

def joint_loss(x: list[float], n_harmonics: int,
               dataset: list[tuple[str, str]],
               end_zones: list = None) -> float:
    """Joint loss over (φ, model_params): question reconstruction + answer continuation.

    Variables: [phi, omega_0, amp_0, ..., omega_n, amp_n, dt, phase_feedback]
    """
    if end_zones is None:
        end_zones = _DEFAULT_END_ZONES
    phi = x[0]
    harmonics = [(abs(x[1 + i*2]), abs(x[2 + i*2])) for i in range(n_harmonics)]
    amp_total = sum(a for _, a in harmonics) or 1e-9
    dt = abs(x[1 + n_harmonics*2]) + 1e-6
    phase_feedback = x[2 + n_harmonics*2]

    total, count = 0.0, 0
    for q, a in dataset:
        ys_q, t_end, phi_end = _simulate_params(0.0, phi, harmonics, amp_total, dt, phase_feedback, len(q))
        ys_a, _, _ = _simulate_params(t_end, phi_end, harmonics, amp_total, dt, phase_feedback, len(a))
        for y_gen, ch in zip(ys_q, q):
            total += (y_gen - char_to_y(ch)) ** 2
            count += 1
        for y_gen, ch in zip(ys_a, a):
            if ch == _END_CHAR:
                total += _end_pos_loss(y_gen, end_zones)
            else:
                total += (y_gen - char_to_y(ch)) ** 2
            count += 1
    return total / max(count, 1)


def joint_train(base_model_name: str, dataset_path: str, output_name: str,
                multi_end: bool = False) -> None:
    """Joint optimisation of φ and model parameters for small datasets."""
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
    print(f"  Mode        : joint (φ + model params, multi-start)")

    best_x, best_loss = None, float("inf")

    try:
        from scipy.optimize import minimize, differential_evolution

        # build bounds: phi in [0, 2pi], omegas > 0, amps > 0, dt > 0, feedback free
        bounds = [(0, 2 * math.pi)]
        for _ in range(n):
            bounds += [(0.1, 30.0), (0.001, 5.0)]
        bounds += [(0.01, 1.0), (-0.5, 0.5)]

        print("  Optimiser   : differential evolution (global) + Nelder-Mead (local)")
        print()

        end_zones = _MULTI_END_ZONES if multi_end else _DEFAULT_END_ZONES

        def objective(x):
            return joint_loss(list(x), n, dataset, end_zones)

        # global search
        best_de_x, best_de_loss = None, float("inf")
        for seed in [42, 7, 137]:
            r = differential_evolution(
                objective, bounds,
                maxiter=10000, tol=1e-8, seed=seed,
                popsize=25, mutation=(0.3, 1.9), recombination=0.85,
                disp=False, workers=1,
                updating="deferred", polish=True,
                init="sobol",
            )
            print(f"  DE seed={seed:<3}  loss={r.fun:.6f}")
            if r.fun < best_de_loss:
                best_de_loss, best_de_x = r.fun, r.x

        # local refinement from best DE result
        result_nm = minimize(objective, best_de_x, method="Nelder-Mead",
                             options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9})
        x_opt, final_loss = list(result_nm.x), result_nm.fun
        print(f"  NM loss     : {final_loss:.6f}")

    except (ImportError, TypeError):
        # fallback: multi-start hill climbing
        print("  Optimiser   : multi-start hill climbing")
        print()
        for seed in range(10):
            random.seed(seed)
            phi0_init = random.uniform(0, 2 * math.pi)
            x0 = [phi0_init] + pack(harmonics, base["dt"], base["phase_feedback"])
            x_cand, loss_cand = _joint_hill_climb(x0, n, dataset)
            if loss_cand < best_loss:
                best_loss = loss_cand
                best_x = x_cand
            print(f"    seed {seed}  loss={loss_cand:.6f}  best={best_loss:.6f}")
        x_opt, final_loss = best_x, best_loss

    # unpack and save
    phi_opt = x_opt[0]
    opt_harmonics = [[abs(x_opt[1 + i*2]), abs(x_opt[2 + i*2])] for i in range(n)]
    opt_dt = abs(x_opt[1 + n*2]) + 1e-6
    opt_feedback = x_opt[2 + n*2]
    amp_total = sum(a for _, a in opt_harmonics) or 1e-9

    print(f"\n  Final loss  : {final_loss:.6f}")
    print(f"  Best φ      : {phi_opt:.6f} rad  ({math.degrees(phi_opt):.1f}°)")

    # verify
    ys_q, t_end, phi_end = _simulate_params(0.0, phi_opt, opt_harmonics, amp_total, opt_dt, opt_feedback, len(dataset[0][0]))
    ys_a, _, _ = _simulate_params(t_end, phi_end, opt_harmonics, amp_total, opt_dt, opt_feedback, len(dataset[0][1]))
    recon_q = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_q)
    recon_a = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_a)
    print(f"  Question    : got={recon_q!r}  want={dataset[0][0]!r}")
    print(f"  Answer      : got={recon_a!r}  want={dataset[0][1]!r}")

    out_model = {
        "name": output_name,
        "description": f"joint-trained on '{dataset_path}' from base '{base_model_name}'",
        "harmonics": [[round(o, 6), round(a, 6)] for o, a in opt_harmonics],
        "dt": round(opt_dt, 6),
        "phase_feedback": round(opt_feedback, 6),
        "steps": steps,
        "_training_phi": round(phi_opt, 6),
    }
    if multi_end:
        out_model["end_zones"] = [[round(c, 4), round(h, 4)] for c, h in _MULTI_END_ZONES]
    out_path = MODELS_DIR / f"{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_model, f, indent=2)
    print(f"  Saved → {out_path}")


# ── multi-joint optimisation (correct multi-pair) ─────────────────

def multijoint_loss(x: list[float], n_harmonics: int,
                    dataset: list[tuple[str, str]],
                    end_zones: list = None) -> float:
    """Joint loss with one φ per pair — correct for multi-pair datasets.

    Variables: [phi_0, phi_1, ..., phi_N, omega_0, amp_0, ..., dt, feedback]
    No inner phase search. All phis are optimised simultaneously with model params.
    """
    if end_zones is None:
        end_zones = _DEFAULT_END_ZONES
    n = len(dataset)
    phis = x[:n]
    harmonics = [(abs(x[n + i*2]), abs(x[n + i*2+1])) for i in range(n_harmonics)]
    amp_total = sum(a for _, a in harmonics) or 1e-9
    dt = abs(x[n + n_harmonics*2]) + 1e-6
    phase_feedback = x[n + n_harmonics*2 + 1]

    total, count = 0.0, 0
    for phi, (q, a) in zip(phis, dataset):
        ys_q, t_end, phi_end = _simulate_params(
            0.0, phi, harmonics, amp_total, dt, phase_feedback, len(q))
        ys_a, _, _ = _simulate_params(
            t_end, phi_end, harmonics, amp_total, dt, phase_feedback, len(a))
        for y_gen, ch in zip(ys_q, q):
            total += (y_gen - char_to_y(ch)) ** 2
            count += 1
        for y_gen, ch in zip(ys_a, a):
            if ch == _END_CHAR:
                total += _end_pos_loss(y_gen, end_zones)
            else:
                total += (y_gen - char_to_y(ch)) ** 2
            count += 1
    return total / max(count, 1)


def multijoint_train(base_model_name: str, dataset_path: str, output_name: str,
                     multi_end: bool = False) -> None:
    """Multi-pair joint optimisation: one φ per pair, one DE run over all params."""
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

    n_pairs = len(dataset)
    print(f"  Base model  : {base_model_name}  ({n} harmonics, {steps} steps)")
    print(f"  Dataset     : {n_pairs} pairs  [{dataset_path}]")
    print(f"  Mode        : multi-joint ({n_pairs} phis + model params, no inner search)")
    print(f"  Params total: {n_pairs + n*2 + 2}")

    try:
        from scipy.optimize import differential_evolution, minimize

        bounds = [(0, 2 * math.pi)] * n_pairs
        for _ in range(n):
            bounds += [(0.1, 30.0), (0.001, 5.0)]
        bounds += [(0.01, 1.0), (-0.5, 0.5)]

        end_zones = _MULTI_END_ZONES if multi_end else _DEFAULT_END_ZONES

        def objective(x):
            return multijoint_loss(list(x), n, dataset, end_zones)

        print("  Optimiser   : differential evolution (global) + Nelder-Mead (local)")
        print()

        best_de_x, best_de_loss = None, float("inf")
        for seed in [42, 7, 137]:
            r = differential_evolution(
                objective, bounds,
                maxiter=5000, tol=1e-7, seed=seed,
                popsize=15, mutation=(0.5, 1.5), recombination=0.9,
                disp=False, workers=1,
                updating="deferred", polish=True,
                init="sobol",
            )
            print(f"  DE seed={seed:<3}  loss={r.fun:.6f}")
            if r.fun < best_de_loss:
                best_de_loss, best_de_x = r.fun, r.x

        result_nm = minimize(objective, best_de_x, method="Nelder-Mead",
                             options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9})
        x_opt = list(result_nm.x)
        final_loss = result_nm.fun
        print(f"  NM loss     : {final_loss:.6f}")

    except ImportError:
        print("  scipy not available — install it for multi-joint training")
        return

    phis_opt = x_opt[:n_pairs]
    opt_harmonics = [[abs(x_opt[n_pairs + i*2]), abs(x_opt[n_pairs + i*2+1])] for i in range(n)]
    opt_dt = abs(x_opt[n_pairs + n*2]) + 1e-6
    opt_feedback = x_opt[n_pairs + n*2 + 1]
    amp_total = sum(a for _, a in opt_harmonics) or 1e-9

    print(f"\n  Final loss  : {final_loss:.6f}")

    correct = 0
    for phi, (q, a) in zip(phis_opt, dataset):
        ys_q, t_end, phi_end = _simulate_params(
            0.0, phi, opt_harmonics, amp_total, opt_dt, opt_feedback, len(q))
        ys_a, _, _ = _simulate_params(
            t_end, phi_end, opt_harmonics, amp_total, opt_dt, opt_feedback, len(a))
        got_q = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_q)
        got_a = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_a)
        ok = "✓" if got_q == q and got_a == a else "✗"
        if got_q == q and got_a == a:
            correct += 1
        print(f"  {ok}  q={got_q!r:<8} want={q!r:<8}  a={got_a!r:<5} want={a!r}")

    print(f"\n  Correct: {correct}/{n_pairs}")

    out_model = {
        "name": output_name,
        "description": f"multijoint-trained on '{dataset_path}' from base '{base_model_name}'",
        "harmonics": [[round(o, 6), round(a, 6)] for o, a in opt_harmonics],
        "dt": round(opt_dt, 6),
        "phase_feedback": round(opt_feedback, 6),
        "steps": steps,
        "_training_phis": [round(phi, 6) for phi in phis_opt],
    }
    if multi_end:
        out_model["end_zones"] = [[round(c, 4), round(h, 4)] for c, h in _MULTI_END_ZONES]
    out_path = MODELS_DIR / f"{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_model, f, indent=2)
    print(f"  Saved → {out_path}")


# ── stream training ────────────────────────────────────────────────

def stream_train(base_model_name: str, dataset_path: str, output_name: str) -> None:
    """Stream training: find one φ from which the entire dataset stream emerges.

    The stream is the concatenation of all QA pairs. Inference works by
    scanning the generated text (not by phase search), so it is O(1) per query
    once the model is trained.

    This is the most radical departure from bilevel: no inner loop, no per-pair
    phis. The model is a compressed waveform that contains the full dataset.
    """
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

    stream = "".join(q + a for q, a in dataset)
    targets_y = [char_to_y(c) for c in stream]

    print(f"  Base model  : {base_model_name}  ({n} harmonics, {steps} steps)")
    print(f"  Dataset     : {len(dataset)} pairs  [{dataset_path}]")
    print(f"  Stream len  : {len(stream)} chars")
    print(f"  Mode        : stream (single φ, scan-based inference)")

    try:
        from scipy.optimize import differential_evolution, minimize

        def _stream_loss(x: list) -> float:
            phi = x[0]
            h = [(abs(x[1 + i*2]), abs(x[2 + i*2])) for i in range(n)]
            at = sum(a for _, a in h) or 1e-9
            dt = abs(x[1 + n*2]) + 1e-6
            pf = x[2 + n*2]
            ys, _, _ = _simulate_params(0.0, phi, h, at, dt, pf, len(stream))
            return sum((y - ty)**2 for y, ty in zip(ys, targets_y)) / len(targets_y)

        bounds = [(0, 2 * math.pi)]
        for _ in range(n):
            bounds += [(0.1, 30.0), (0.001, 5.0)]
        bounds += [(0.01, 1.0), (-0.5, 0.5)]

        print("  Optimiser   : differential evolution (global) + Nelder-Mead (local)")
        print()

        best_de_x, best_de_loss = None, float("inf")
        for seed in [42, 7, 137]:
            r = differential_evolution(
                _stream_loss, bounds,
                maxiter=5000, tol=1e-7, seed=seed,
                popsize=20, mutation=(0.5, 1.5), recombination=0.9,
                disp=False, workers=1,
                updating="deferred", polish=True,
                init="sobol",
            )
            print(f"  DE seed={seed:<3}  loss={r.fun:.6f}")
            if r.fun < best_de_loss:
                best_de_loss, best_de_x = r.fun, r.x

        result_nm = minimize(_stream_loss, best_de_x, method="Nelder-Mead",
                             options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9})
        x_opt = list(result_nm.x)
        final_loss = result_nm.fun
        print(f"  NM loss     : {final_loss:.6f}")

    except ImportError:
        print("  scipy not available")
        return

    phi_opt = x_opt[0]
    opt_harmonics = [[abs(x_opt[1 + i*2]), abs(x_opt[2 + i*2])] for i in range(n)]
    opt_dt = abs(x_opt[1 + n*2]) + 1e-6
    opt_feedback = x_opt[2 + n*2]
    amp_total = sum(a for _, a in opt_harmonics) or 1e-9

    ys_stream, _, _ = _simulate_params(0.0, phi_opt, opt_harmonics, amp_total, opt_dt, opt_feedback, len(stream))
    got_stream = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_stream)

    print(f"\n  Final loss  : {final_loss:.6f}")
    print(f"  φ           : {phi_opt:.6f} rad  ({math.degrees(phi_opt):.1f}°)")
    print(f"  Stream want : {stream!r}")
    print(f"  Stream got  : {got_stream!r}")

    correct = sum(1 for a, b in zip(got_stream, stream) if a == b)
    print(f"  Char match  : {correct}/{len(stream)}")

    out_model = {
        "name": output_name,
        "description": f"stream-trained on '{dataset_path}' from base '{base_model_name}'",
        "harmonics": [[round(o, 6), round(a, 6)] for o, a in opt_harmonics],
        "dt": round(opt_dt, 6),
        "phase_feedback": round(opt_feedback, 6),
        "steps": steps,
        "_stream_phi": round(phi_opt, 6),
        "_stream": stream,
    }
    out_path = MODELS_DIR / f"{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_model, f, indent=2)
    print(f"  Saved → {out_path}")


def _joint_hill_climb(x0, n_harmonics, dataset, iterations=2000):
    x = list(x0)
    best_loss = joint_loss(x, n_harmonics, dataset)
    scale = 0.2
    for i in range(iterations):
        candidate = [v + random.gauss(0, scale) for v in x]
        candidate[0] = candidate[0] % (2 * math.pi)  # keep phi in [0, 2pi]
        loss = joint_loss(candidate, n_harmonics, dataset)
        if loss < best_loss:
            best_loss = loss
            x = candidate
            scale = min(scale * 1.1, 1.0)
        else:
            scale = max(scale * 0.998, 1e-5)
    return x, best_loss


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
    parser.add_argument("--joint", action="store_true",
                        help="joint optimisation of phi + model params (single-pair only)")
    parser.add_argument("--multijoint", action="store_true",
                        help="multi-joint: one phi per pair, no bilevel inner search (correct for multi-pair)")
    parser.add_argument("--stream", action="store_true",
                        help="stream mode: encode full dataset as one sequence, scan-based inference")
    parser.add_argument("--multi-end", action="store_true",
                        help="use 3 END zones spread across y-space (low/mid/high) for easier END learning")
    args = parser.parse_args()
    multi_end = args.multi_end

    print(f"\n{_SEP}")
    if args.stream:
        print("  SinMachine Trainer  —  stream mode")
        print(_SEP)
        stream_train(args.base, args.dataset, args.output)
    elif args.multijoint:
        print("  SinMachine Trainer  —  multi-joint optimisation")
        print(_SEP)
        multijoint_train(args.base, args.dataset, args.output, multi_end=multi_end)
    elif args.joint:
        print("  SinMachine Trainer  —  joint optimisation  (single-pair)")
        print(_SEP)
        joint_train(args.base, args.dataset, args.output, multi_end=multi_end)
    else:
        print("  SinMachine Trainer  —  bilevel inverse optimisation")
        print(_SEP)
        train(args.base, args.dataset, args.output, args.resolution)
    print(_SEP + "\n")
