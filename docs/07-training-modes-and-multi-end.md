# 07 — Training Modes and Multi-END Experiments

## What broke the original joint trainer on multi-pair datasets

The original `--joint` mode used a **single φ for all pairs in the dataset**.
For a single pair (hello-world) this is correct: one entry phase, one answer.
For 20 pairs (simple-sums) it is fundamentally impossible: "1+1=" and "1+2="
need different starting phases by definition, so the optimizer was searching for
something that cannot exist.

This was discovered by observing that the simple-sums training consumed 100% CPU
for 11+ minutes with no output — the DE had no feasible region to converge to.

## Three new training modes

### 1. `--multijoint` — one φ per pair (correct multi-pair)

```
params: [φ_0, φ_1, ..., φ_N, ω_0, A_0, ..., ω_K, A_K, dt, feedback]
```

No bilevel inner search. All N phis and all model params are optimised
simultaneously in a single DE run. For simple-sums (20 pairs, 8 harmonics):
38 params total. Each evaluation is O(N · steps) — much faster than bilevel
which does N inner phase searches per outer evaluation.

This is the correct generalisation of `--joint` to multi-pair datasets.

### 2. `--stream` — dataset as a single waveform

All QA pairs are concatenated into one long sequence:
```
1+1=2\x031+2=3\x03...4+4=8\x03
```

Find ONE φ from which this entire stream emerges. Inference works by scanning
the generated text for the question as a substring, then reading the following
chars as the answer. No phase search at inference time — O(1) retrieval.

This makes the model a **content-addressable waveform memory**: the dataset is
encoded in the harmonic function and retrieved by pattern matching.

### 3. `--bilevel` (default) — original approach

Inner: phase search per question. Outer: answer continuation MSE.
Correct but slow for large datasets. Useful for exploration.

## Multi-END: spreading the END token across y-space

### The problem with a single END token

END is at token index 96 (out of 97), corresponding to y ≈ +0.99 — the global
maximum of the normalised harmonic function. For a sum of 8 sinusoids to
simultaneously reach their positive peaks requires a rare alignment. The
optimizer must find (t, φ) where this alignment occurs at exactly the right
step number, while also satisfying all preceding character constraints.

Without END: training hello-world converged quickly.
With END: loss 0.000180 after long DE runs, still not perfect.

### The solution: vocabulary-level multi-END zones

Instead of one target y ≈ +0.99, define three zones spread across y-space:

```
END_LOW  : center y = -0.90  (below arithmetic chars y ∈ [-0.76, -0.39])
END_MID  : center y = +0.40  (above arithmetic chars)
END_HIGH : center y = +0.90  (near the sinusoidal peak)
```

Each zone has half-width ≈ 3 token buckets ≈ 0.062 in y-space.
Total END coverage: ~3 × 6 buckets = 18 out of 97 ≈ 18% of y-space,
versus the original 1 bucket ≈ 1%.

**At training**: the loss for an END position is the minimum squared distance
to any zone center:
```python
loss_end = min((y_gen - center)**2 for center, _ in end_zones)
```

The optimizer can choose which END zone to aim for — whichever is easiest to
reach at the right time step. This significantly relaxes the constraint.

**At inference**: generation stops when y falls inside any zone.

**Safety for arithmetic**: arithmetic chars ('0'–'8', '+', '=') map to
y ∈ [-0.76, -0.39]. None of the three END zones intersect this range, so
there are no false END triggers during arithmetic generation.

### Using multi-END

Multi-END is now the default for new training, benchmark, and vocabulary-aligned
experiments. Use `--single-end` only for control experiments that intentionally
test the old single-END constraint.

```bash
make hello-world-me     # legacy explicit target, now equivalent to default multi-END
make simple-sums-me     # legacy explicit target, now equivalent to default multi-END
```

Or directly:
```bash
python3 trainer.py dataset.jsonl --joint --multi-end
python3 trainer.py dataset.jsonl --multijoint --multi-end
```

The trained model saves `end_zones` in its JSON so inference automatically
uses the correct zones.

## Experiments planned

| Experiment | Mode | END | Goal |
|---|---|---|---|
| simple-sums-noend | multijoint | none (length-based) | baseline: can 8 harmonics learn 20 pairs? |
| simple-sums | multijoint | single (high) | baseline with END |
| simple-sums-me | multijoint | 3 zones | test multi-END benefit |
| hello-world-me | joint | 3 zones | compare vs hello-world loss 0.000180 |
| simple-sums-stream | stream | part of sequence | scan-based inference |

## Key insight from this session

The fundamental tension in SinMachine multi-pair training:

- A harmonic function has bounded expressivity in phase space
- N pairs require N distinct φ-entry-points that all satisfy their constraints
- The model params must simultaneously make all N trajectories feasible
- This is a hard combinatorial geometry problem

Multi-END reduces the per-pair constraint count (the END step is more flexible),
making the overall problem easier without sacrificing the core mechanism.
