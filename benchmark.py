#!/usr/bin/env python3
"""Progressive training benchmark.

Trains on 1 pair, then 2, then 3, ... measuring wall time and final loss
for each N. This reveals how training complexity scales with dataset size.

Usage:
    python3 benchmark.py [dataset.jsonl] [--base MODEL] [--max N] [--de-iters K]
"""

import argparse
import json
import math
import time
import sys

from sinmachine import (_VOCAB_SIZE, MODELS_DIR, build_default_perm)
from trainer import _simulate_params, multijoint_loss


def _benchmark_run(dataset, n_pairs, harmonics, base_dt, base_feedback,
                   de_iters, de_popsize, seeds, perm):
    """Train on the first n_pairs entries. Returns (loss, elapsed_sec, correct)."""
    subset = dataset[:n_pairs]
    n = len(harmonics)

    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError:
        print("  scipy required")
        sys.exit(1)

    bounds = [(0, 2 * math.pi)] * n_pairs
    for _ in range(n):
        bounds += [(0.1, 30.0), (0.001, 5.0)]
    bounds += [(0.01, 1.0), (-0.5, 0.5)]

    def objective(x):
        return multijoint_loss(list(x), n, subset, perm)

    t0 = time.perf_counter()
    best_x, best_loss = None, float("inf")

    for seed in seeds:
        r = differential_evolution(
            objective, bounds,
            maxiter=de_iters, tol=1e-6, seed=seed,
            popsize=de_popsize,
            mutation=(0.5, 1.5), recombination=0.9,
            disp=False, workers=1,
            updating="deferred", polish=False,
            init="sobol",
        )
        if r.fun < best_loss:
            best_loss, best_x = r.fun, r.x

    elapsed = time.perf_counter() - t0

    # count correct answers
    phis = best_x[:n_pairs]
    h = [(abs(best_x[n_pairs + i*2]), abs(best_x[n_pairs + i*2+1])) for i in range(n)]
    at = sum(a for _, a in h) or 1e-9
    dt = abs(best_x[n_pairs + n*2]) + 1e-6
    pf = best_x[n_pairs + n*2 + 1]

    correct = 0
    details = []
    for phi, (q, a) in zip(phis, subset):
        ys_q, t_end, phi_end = _simulate_params(0.0, phi, h, at, dt, pf, len(q))
        ys_a, _, _ = _simulate_params(t_end, phi_end, h, at, dt, pf, len(a))
        got_q = "".join(perm[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_q)
        got_a = "".join(perm[max(0, min(_VOCAB_SIZE-1, int((y+1)/2*_VOCAB_SIZE)))] for y in ys_a)
        ok = got_q == q and got_a == a
        if ok:
            correct += 1
        details.append((ok, q, got_q, a, got_a))

    return best_loss, elapsed, correct, details


def main():
    parser = argparse.ArgumentParser(description="Progressive training benchmark")
    parser.add_argument("dataset", nargs="?", default="datasets/simple-sums-noend.jsonl")
    parser.add_argument("--base", default="dense")
    parser.add_argument("--max", type=int, default=None, help="max pairs to test (default: all)")
    parser.add_argument("--de-iters", type=int, default=2000, help="DE max iterations per run")
    parser.add_argument("--de-pop", type=int, default=12, help="DE population size multiplier")
    parser.add_argument("--seeds", default="42,7", help="comma-separated DE seeds")
    parser.add_argument("--multi-end", action="store_true", default=True, help="use vocabulary-level multi-END (default)")
    parser.add_argument("--single-end", action="store_true", help="disable multi-END for control experiments")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    multi_end = not args.single_end
    perm = build_default_perm(multi_end)

    # load base model
    import pathlib
    base_path = pathlib.Path("models") / f"{args.base}.json"
    with open(base_path) as f:
        base = json.load(f)
    harmonics = [tuple(h) for h in base["harmonics"]]
    n_harmonics = len(harmonics)

    # load dataset
    dataset = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                dataset.append((obj["q"], obj["a"]))

    max_pairs = args.max or len(dataset)
    end_label = "multi-END" if multi_end else "single-END"

    print(f"\n{'─'*68}")
    print(f"  SinMachine — Progressive Training Benchmark")
    print(f"{'─'*68}")
    print(f"  Dataset     : {args.dataset}  ({len(dataset)} pairs total)")
    print(f"  Base model  : {args.base}  ({n_harmonics} harmonics)")
    print(f"  DE iters    : {args.de_iters}  popsize={args.de_pop}  seeds={seeds}")
    print(f"  END mode    : {end_label}")
    print(f"{'─'*68}")
    print(f"  {'N':>3}  {'loss':>10}  {'time':>8}  {'correct':>9}  {'rate':>6}  {'s/pair':>7}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*7}")

    results = []
    cumulative = 0.0

    for n in range(1, max_pairs + 1):
        loss, elapsed, correct, details = _benchmark_run(
            dataset, n, harmonics, base["dt"], base["phase_feedback"],
            args.de_iters, args.de_pop, seeds, perm
        )
        cumulative += elapsed
        rate = correct / n
        per_pair = elapsed / n

        marker = "  " if correct == n else " ✗"
        print(f"{marker} {n:>3}  {loss:>10.6f}  {elapsed:>7.1f}s  "
              f"{correct:>4}/{n:<4}  {rate:>5.0%}  {per_pair:>6.1f}s")
        sys.stdout.flush()

        results.append({
            "n": n, "loss": loss, "elapsed": elapsed,
            "correct": correct, "rate": rate, "per_pair": per_pair,
            "details": details,
        })

        # print details for wrong answers
        for ok, q, got_q, a, got_a in details:
            if not ok:
                print(f"       ✗ q={got_q!r:<8} want={q!r:<8}  a={got_a!r:<5} want={a!r}")

    print(f"{'─'*68}")
    print(f"  Total time: {cumulative:.1f}s")

    # time projection
    if len(results) >= 3:
        # fit linear regression on log-log scale to estimate scaling
        import math as _m
        xs = [_m.log(r["n"]) for r in results]
        ys = [_m.log(max(r["elapsed"], 0.01)) for r in results]
        n_pts = len(xs)
        mx = sum(xs) / n_pts
        my = sum(ys) / n_pts
        slope = sum((x - mx)*(y - my) for x, y in zip(xs, ys)) / max(sum((x - mx)**2 for x in xs), 1e-9)
        intercept = my - slope * mx

        print(f"\n  Scaling exponent α ≈ {slope:.2f}  (time ∝ N^α)")
        if len(dataset) > max_pairs:
            n_full = len(dataset)
            t_est = math.exp(intercept + slope * math.log(n_full))
            print(f"  Estimated time for N={n_full}: {t_est:.0f}s  ({t_est/60:.1f} min)")

    print()


if __name__ == "__main__":
    main()
