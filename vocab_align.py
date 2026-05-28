#!/usr/bin/env python3
"""Vocabulary Alignment — iterative progressive swapping.

The optimizer consistently produces tokens adjacent to the correct ones
(e.g., '1*3<3' instead of '1+1=2').  Instead of fighting the optimizer,
we progressively swap the vocabulary positions of the 'wrong' and 'right'
chars so that what the model naturally produces becomes correct.

State is maintained as a full permutation of the vocab (a list of 97 chars),
guaranteeing that the mapping is always a valid bijection.

Each iteration:
  1. Train for --budget seconds (using current perm for char→y targets)
  2. Decode result: compare got vs target
  3. For each mismatched position: propose swap perm[idx_got] ↔ perm[idx_target]
     (skip conflicts: same source/destination idx needed by multiple positions)
  4. Apply non-conflicting swaps atomically on the perm array
  5. Retrain with the new perm
  6. Repeat until correct or max-rounds

Usage:
    python3 vocab_align.py --target "1+1=2" --budget 10 --rounds 8
"""

import argparse
import math
import time

import numpy as np
from scipy.optimize import differential_evolution

from sinmachine import _VOCAB_SIZE, _DEFAULT_END_ZONES, _END_CHAR


# ── vocab as a permutation ────────────────────────────────────────

def build_default_perm():
    """Default vocab: perm[i] = char at y-index i (97 chars total)."""
    return [chr(i) for i in range(32, 127)] + ['\x02', '\x03']


def char_to_idx(c, perm):
    """Return the y-index of char c in the current permutation."""
    try:
        return perm.index(c)
    except ValueError:
        # Fallback for chars outside the vocab (shouldn't happen in normal use)
        return max(0, min(_VOCAB_SIZE - 1, ord(c) - 32))


def y_center(idx):
    return ((idx + 0.5) / _VOCAB_SIZE) * 2 - 1


# ── simulation ────────────────────────────────────────────────────

def simulate(x, n_h, sequence, perm, end_zones, teacher_forced=False):
    phi = x[0]
    harmonics = [(abs(x[1 + i * 2]), abs(x[2 + i * 2])) for i in range(n_h)]
    at = sum(a for _, a in harmonics) or 1e-9
    dt = abs(x[1 + n_h * 2]) + 1e-6
    pf = x[2 + n_h * 2]

    targets_y = [y_center(char_to_idx(c, perm)) for c in sequence]
    correct_idxs = [char_to_idx(c, perm) for c in sequence]

    t, cur_phi, total = 0.0, phi, 0.0
    for step, (ty, cidx) in enumerate(zip(targets_y, correct_idxs)):
        y = sum(a * math.sin(w * t + cur_phi) for w, a in harmonics) / at
        idx = max(0, min(_VOCAB_SIZE - 1, int((y + 1) / 2 * _VOCAB_SIZE)))
        if sequence[step] == _END_CHAR:
            total += min((y - c) ** 2 for c, _ in end_zones)
        else:
            total += (y - ty) ** 2
        fb_idx = cidx if teacher_forced else idx
        cur_phi += pf * (fb_idx / _VOCAB_SIZE) * 2 * math.pi
        t += dt
    return total / len(sequence)


def decode(x, n_h, n_steps, perm):
    phi = x[0]
    harmonics = [(abs(x[1 + i * 2]), abs(x[2 + i * 2])) for i in range(n_h)]
    at = sum(a for _, a in harmonics) or 1e-9
    dt = abs(x[1 + n_h * 2]) + 1e-6
    pf = x[2 + n_h * 2]

    t, cur_phi, out = 0.0, phi, []
    for _ in range(n_steps):
        y = sum(a * math.sin(w * t + cur_phi) for w, a in harmonics) / at
        idx = max(0, min(_VOCAB_SIZE - 1, int((y + 1) / 2 * _VOCAB_SIZE)))
        out.append(perm[idx])
        cur_phi += pf * (idx / _VOCAB_SIZE) * 2 * math.pi
        t += dt
    return "".join(out)


# ── training ──────────────────────────────────────────────────────

def train(sequence, n_h, perm, budget_sec, seed=42,
          end_zones=None, teacher_forced=False):
    if end_zones is None:
        end_zones = _DEFAULT_END_ZONES

    bounds = ([(0, 2 * math.pi)]
              + [(0.1, 12.0), (0.001, 2.0)] * n_h
              + [(0.01, 0.4), (-0.4, 0.4)])

    deadline = time.perf_counter() + budget_sec
    best_x, best_loss = None, float("inf")

    class _CB:
        def __call__(self, xk, convergence):
            nonlocal best_x, best_loss
            loss = simulate(list(xk), n_h, sequence, perm, end_zones, teacher_forced)
            if loss < best_loss:
                best_loss = loss
                best_x = list(xk)
            return best_loss < 1e-8 or time.perf_counter() >= deadline

    try:
        differential_evolution(
            lambda x: simulate(list(x), n_h, sequence, perm, end_zones, teacher_forced),
            bounds,
            maxiter=50000, tol=1e-10, seed=seed,
            popsize=10, mutation=(0.5, 1.5), recombination=0.9,
            disp=False, workers=1, updating="deferred", polish=False,
            init="sobol", callback=_CB(),
        )
    except Exception:
        pass

    return best_x, best_loss


# ── progressive swap ──────────────────────────────────────────────

def apply_swaps(got, target, perm):
    """Propose and apply non-conflicting swaps on the permutation array.

    Conflicts:
      - Same source idx maps to two different target idxs
      - Two different source idxs map to the same target idx
    All conflicting proposals are dropped; the rest are applied atomically.
    The permutation remains a valid bijection after each call.
    """
    proposed = {}   # idx_g → idx_t
    src_conflicts = set()   # idx_g values with conflicting targets
    tgt_seen = {}           # idx_t → first idx_g that claimed it
    tgt_conflicts = set()   # idx_t values claimed by multiple sources

    for g, t in zip(got, target):
        if g == t:
            continue
        idx_g = char_to_idx(g, perm)
        idx_t = char_to_idx(t, perm)
        if idx_g == idx_t:
            continue  # already at the right position

        if idx_g in proposed:
            if proposed[idx_g] != idx_t:
                src_conflicts.add(idx_g)
        else:
            proposed[idx_g] = idx_t

        if idx_t in tgt_seen:
            if tgt_seen[idx_t] != idx_g:
                tgt_conflicts.add(idx_t)
        else:
            tgt_seen[idx_t] = idx_g

    # Drop proposals where either end of the swap is in conflict
    all_conflicts = set()
    for idx_g, idx_t in proposed.items():
        if idx_g in src_conflicts or idx_t in tgt_conflicts:
            all_conflicts.add(idx_g)

    new_perm = list(perm)
    applied = []

    for idx_g, idx_t in proposed.items():
        if idx_g in all_conflicts:
            continue
        # Atomic swap in the permutation array
        new_perm[idx_g], new_perm[idx_t] = new_perm[idx_t], new_perm[idx_g]
        applied.append((perm[idx_g], idx_g, idx_t, perm[idx_t]))

    return new_perm, applied, list(all_conflicts)


# ── main loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="1+1=2", help="sequence to memorise")
    parser.add_argument("--harmonics", type=int, default=2, help="number of harmonics")
    parser.add_argument("--budget", type=float, default=10.0, help="seconds per round")
    parser.add_argument("--rounds", type=int, default=8, help="max swap rounds")
    parser.add_argument("--teacher", action="store_true", help="teacher-forced feedback")
    args = parser.parse_args()

    seq = args.target
    n_h = args.harmonics
    perm = build_default_perm()
    end_zones = _DEFAULT_END_ZONES

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  Vocabulary Alignment — progressive swap")
    print(sep)
    print(f"  Target   : {seq!r}  ({len(seq)} chars)")
    print(f"  Harmonics: {n_h}  Budget: {args.budget}s/round  Max rounds: {args.rounds}")
    print(sep)

    for rnd in range(args.rounds):
        # Show current idx and y for each unique char in target
        unique_chars = sorted(set(seq))
        mapping_str = "  ".join(
            f"{c!r}→idx{char_to_idx(c, perm)}(y={y_center(char_to_idx(c, perm)):+.3f})"
            for c in unique_chars
        )
        print(f"\n  Round {rnd + 1}  vocab: {mapping_str}")

        t0 = time.perf_counter()
        best_x, loss = train(seq, n_h, perm, args.budget,
                             seed=42 + rnd, end_zones=end_zones,
                             teacher_forced=args.teacher)
        elapsed = time.perf_counter() - t0

        if best_x is None:
            print(f"  Training failed (no improvement).")
            break

        got = decode(best_x, n_h, len(seq), perm)
        ok = "✓" if got == seq else "✗"

        print(f"  {ok}  got={got!r}  want={seq!r}  loss={loss:.6f}  t={elapsed:.1f}s")
        matches = sum(1 for g, t in zip(got, seq) if g == t)
        print(f"  Chars correct: {matches}/{len(seq)}")

        if got == seq:
            print(f"\n  ✓ CONVERGED in round {rnd + 1}!")
            # Show final permutation diff from default
            default = build_default_perm()
            diffs = [(i, default[i], perm[i]) for i in range(_VOCAB_SIZE) if perm[i] != default[i]]
            print(f"  Permutation changes: {len(diffs)} positions remapped")
            for idx, d, p in diffs[:20]:
                print(f"    idx {idx:3d}: default={d!r} → now={p!r}")
            break

        new_perm, applied, conflicts = apply_swaps(got, seq, perm)

        if not applied and not conflicts:
            print(f"  No swaps possible (all mismatches are conflicting). Stopping.")
            break

        if applied:
            print(f"  Swaps applied:", end="")
            for char_g, idx_g, idx_t, char_t in applied:
                print(f"  {char_g!r}(idx {idx_g})↔{char_t!r}(idx {idx_t})", end="")
            print()
        else:
            print(f"  No non-conflicting swaps this round.")
        if conflicts:
            print(f"  Conflicts (skipped): source indices {conflicts}")

        perm = new_perm

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
