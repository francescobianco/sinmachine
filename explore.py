#!/usr/bin/env python3
"""Timeout-penalized configuration explorer.

Tries (n_harmonics × n_pairs) configurations using the cheapest optimizer
that fits the parameter count.  Any path taking more than --timeout seconds
is discarded.  Goal: find the minimum fast-converging but conceptually
significant configuration.

Optimizer selection by D (number of parameters):
  D ≤ 6   →  multi-start Nelder-Mead  (grid of random starts, no gradients)
  D ≤ 15  →  basin-hopping + NM
  D > 15  →  DE (global, slow for large D — usually discarded by timeout)

Usage:
    python3 explore.py [--timeout 15] [--threshold 0.001] [--max-pairs 5]
"""

import argparse
import json
import math
import sys
import time

from sinmachine import (char_to_y, VOCAB, _VOCAB_SIZE,
                        _END_CHAR, _MULTI_END_ZONES, _DEFAULT_END_ZONES)
from trainer import _simulate_params


# ── loss and decode ───────────────────────────────────────────────

def _loss(x, n_pairs, n_h, dataset, end_zones):
    phis = x[:n_pairs]
    h = [(abs(x[n_pairs + i*2]), abs(x[n_pairs + i*2+1])) for i in range(n_h)]
    at = sum(a for _, a in h) or 1e-9
    dt = abs(x[n_pairs + n_h*2]) + 1e-6
    pf = x[n_pairs + n_h*2 + 1]
    total, count = 0.0, 0
    for phi, (q, a) in zip(phis, dataset):
        ys_q, t_end, phi_end = _simulate_params(0.0, phi, h, at, dt, pf, len(q))
        ys_a, _, _ = _simulate_params(t_end, phi_end, h, at, dt, pf, len(a))
        for y_gen, ch in zip(ys_q, q):
            total += (y_gen - char_to_y(ch)) ** 2
            count += 1
        for y_gen, ch in zip(ys_a, a):
            if ch == _END_CHAR:
                total += min((y_gen - c)**2 for c, _ in end_zones)
            else:
                total += (y_gen - char_to_y(ch)) ** 2
            count += 1
    return total / max(count, 1)


def _decode_check(x, n_pairs, n_h, dataset):
    phis = x[:n_pairs]
    h = [(abs(x[n_pairs + i*2]), abs(x[n_pairs + i*2+1])) for i in range(n_h)]
    at = sum(a for _, a in h) or 1e-9
    dt = abs(x[n_pairs + n_h*2]) + 1e-6
    pf = x[n_pairs + n_h*2 + 1]
    correct, details = 0, []
    for phi, (q, a) in zip(phis, dataset):
        ys_q, t_end, phi_end = _simulate_params(0.0, phi, h, at, dt, pf, len(q))
        ys_a, _, _ = _simulate_params(t_end, phi_end, h, at, dt, pf, len(a))
        got_q = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_q)
        got_a = "".join(VOCAB[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_a)
        ok = (got_q == q and got_a == a)
        correct += ok
        details.append((ok, q, got_q, a, got_a))
    return correct, details


def _bounds(n_pairs, n_h):
    b = [(0, 2 * math.pi)] * n_pairs
    for _ in range(n_h):
        b += [(0.1, 12.0), (0.001, 2.0)]
    b += [(0.01, 0.4), (-0.2, 0.2)]
    return b


# ── optimisers ────────────────────────────────────────────────────

def _multistart_nm(n_pairs, n_h, dataset, end_zones, timeout, threshold, rng_seed=0):
    """Multi-start Nelder-Mead: best for small D (≤ 8)."""
    import random, math as m
    from scipy.optimize import minimize

    bounds = _bounds(n_pairs, n_h)
    D = len(bounds)
    random.seed(rng_seed)

    def objective(x):
        return _loss(list(x), n_pairs, n_h, dataset, end_zones)

    best_x, best_loss = [0.0]*D, float("inf")
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        x0 = [random.uniform(lo, hi) for lo, hi in bounds]
        remaining = deadline - time.perf_counter()
        if remaining < 0.05:
            break

        try:
            r = minimize(objective, x0, method="Nelder-Mead",
                         options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-7,
                                  "disp": False})
            if r.fun < best_loss:
                best_loss = r.fun
                best_x = list(r.x)
            if best_loss <= threshold:
                break
        except Exception:
            pass

    elapsed = time.perf_counter() - (deadline - timeout)
    return best_loss, min(elapsed, timeout), best_x


def _basin_hopping(n_pairs, n_h, dataset, end_zones, timeout, threshold, seed=42):
    """Basin-hopping: good for medium D (≤ 15)."""
    from scipy.optimize import basinhopping, Bounds

    bounds = _bounds(n_pairs, n_h)
    D = len(bounds)
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    import numpy as np
    rng = np.random.default_rng(seed)
    x0 = lo + rng.random(D) * (np.array(hi) - np.array(lo))

    deadline = time.perf_counter() + timeout
    best_x, best_loss = list(x0), float("inf")

    class _CB:
        def __call__(self, x, f, accepted):
            nonlocal best_x, best_loss
            if f < best_loss:
                best_loss = f
                best_x = list(x)
            if best_loss <= threshold or time.perf_counter() >= deadline:
                raise StopIteration

    try:
        basinhopping(
            lambda x: _loss(list(x), n_pairs, n_h, dataset, end_zones),
            x0,
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "options": {"maxiter": 300, "xatol": 1e-5, "fatol": 1e-6},
            },
            niter=10000,
            T=0.05,
            stepsize=0.3,
            seed=seed,
            callback=_CB(),
        )
    except StopIteration:
        pass
    except Exception:
        pass

    elapsed = time.perf_counter() - (deadline - timeout)
    return best_loss, min(elapsed, timeout), best_x


def _de_search(n_pairs, n_h, dataset, end_zones, timeout, threshold, seed=42):
    """Differential evolution: global but slow for large D."""
    from scipy.optimize import differential_evolution

    bounds = _bounds(n_pairs, n_h)
    D = len(bounds)
    pop = max(8, D * 8)

    deadline = time.perf_counter() + timeout
    best_x, best_loss = [0.0]*D, float("inf")

    class _CB:
        def __call__(self, xk, convergence):
            nonlocal best_x, best_loss
            loss = _loss(list(xk), n_pairs, n_h, dataset, end_zones)
            if loss < best_loss:
                best_loss = loss
                best_x = list(xk)
            return best_loss <= threshold or time.perf_counter() >= deadline

    try:
        differential_evolution(
            lambda x: _loss(list(x), n_pairs, n_h, dataset, end_zones),
            bounds,
            maxiter=50000, tol=1e-9, seed=seed,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            disp=False, workers=1, updating="deferred", polish=False,
            init="sobol", callback=_CB(),
        )
    except Exception:
        pass

    elapsed = time.perf_counter() - (deadline - timeout)
    return best_loss, min(elapsed, timeout), best_x


def run_trial(dataset, n_h, timeout, threshold, end_zones):
    n_pairs = len(dataset)
    D = n_pairs + n_h * 2 + 2

    if D <= 8:
        strategy = "NM-multi"
        loss, elapsed, x = _multistart_nm(n_pairs, n_h, dataset, end_zones, timeout, threshold)
    elif D <= 16:
        strategy = "basin-hop"
        loss, elapsed, x = _basin_hopping(n_pairs, n_h, dataset, end_zones, timeout, threshold)
    else:
        strategy = "DE"
        loss, elapsed, x = _de_search(n_pairs, n_h, dataset, end_zones, timeout, threshold)

    timed_out = elapsed >= timeout * 0.95
    hit = loss <= threshold
    correct, details = _decode_check(x, n_pairs, n_h, dataset)
    return loss, elapsed, correct, details, strategy, timed_out, hit


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="datasets/simple-sums-noend.jsonl")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--multi-end", action="store_true")
    args = parser.parse_args()

    end_zones = _MULTI_END_ZONES if args.multi_end else _DEFAULT_END_ZONES

    dataset = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                dataset.append((obj["q"], obj["a"]))

    harmonics_options = [1, 2, 4, 8]

    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  SinMachine — Timeout-Penalized Exploration")
    print(sep)
    print(f"  Dataset  : {args.dataset}  ({len(dataset)} pairs)")
    print(f"  Timeout  : {args.timeout}s  Threshold: loss<{args.threshold}")
    print(sep)
    print(f"  {'N':>3}  {'H':>3}  {'D':>4}  {'strategy':>9}  {'loss':>10}  {'time':>7}  {'ok':>6}  result")
    print(f"  {'─'*3}  {'─'*3}  {'─'*4}  {'─'*9}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*20}")

    winners = []

    for n_pairs in range(1, args.max_pairs + 1):
        subset = dataset[:n_pairs]
        found_for_n = False

        for n_h in harmonics_options:
            D = n_pairs + n_h * 2 + 2
            loss, elapsed, correct, details, strategy, timed_out, hit = run_trial(
                subset, n_h, args.timeout, args.threshold, end_zones)

            if timed_out:
                result = f"TIMEOUT"
            elif hit and correct == n_pairs:
                result = f"WIN ✓"
            elif hit:
                result = f"threshold ok, {correct}/{n_pairs} correct"
            else:
                result = f"plateau"

            print(f"  {n_pairs:>3}  {n_h:>3}  {D:>4}  {strategy:>9}  {loss:>10.6f}  "
                  f"{elapsed:>6.1f}s  {correct:>3}/{n_pairs:<2}  {result}")

            for ok, q, got_q, a, got_a in details:
                sym = "✓" if ok else "✗"
                print(f"         {sym}  q={got_q!r:<8} want={q!r:<8}  a={got_a!r} want={a!r}")

            sys.stdout.flush()

            if hit and correct == n_pairs:
                winners.append({"n": n_pairs, "H": n_h, "D": D,
                                 "loss": loss, "t": elapsed, "strategy": strategy})
                found_for_n = True
                break

            if timed_out:
                print(f"         → timeout at H={n_h}, skipping larger H for N={n_pairs}")
                break

        print()

    print(sep)
    if winners:
        print(f"  Winning configurations:")
        for w in winners:
            print(f"    N={w['n']}  H={w['H']}  D={w['D']}  "
                  f"loss={w['loss']:.6f}  t={w['t']:.1f}s  via {w['strategy']}")
    else:
        print(f"  No configuration won within {args.timeout}s per trial.")
        print(f"  Suggestions: --timeout higher, --threshold looser, fewer harmonics.")
    print()


if __name__ == "__main__":
    main()
