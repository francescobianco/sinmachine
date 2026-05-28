# 08 — Arithmetic Infeasibility and the Vocabulary Problem

## What we found (progressive benchmark session)

### Timing is not the bottleneck

With multi-joint mode (one φ per pair, no inner search):
- N=1 pair, D=19 params: ~0.013ms/eval → 2000 iters takes 61s
- N=20 pairs, D=38 params: ~0.184ms/eval → 2000 iters takes ~210s
- Scaling exponent α ≈ 0.61 (sub-linear in N)

Training 20 pairs in 3 minutes is feasible. The issue is not speed.

### The optimizer gets stuck at loss ≈ 0.000730

For "1+1=2" with 1, 2, or 8 harmonics, with any optimizer tried
(DE multi-seed, Nelder-Mead multi-start, basin-hopping, teacher forcing),
the loss plateaus at ≈ 0.000730 and never decreases.

RMSE per char ≈ √0.000730 ≈ 0.027. Bucket width = 2/97 ≈ 0.021.
The optimizer consistently produces tokens that are 1 bucket ADJACENT to the
correct ones (e.g., '1*3<3' instead of '1+1=2').

### Feasibility check: no solution exists for 2 harmonics

Exhaustive grid search over 960,000 parameter combinations (phi, ω₁, ω₂, dt)
with amplitude ratio as a free variable via linear programming:

**Result: 0 solutions found for "1+1=2" with 2 harmonics and pf=0.**

This means the constraint system for "1+1=2" is infeasible with 2 sinusoids.

## Root cause: the ASCII vocabulary clusters arithmetic chars in a narrow y-band

Arithmetic characters in the current ASCII-linear vocabulary:

| char | ASCII | idx | y_center |
|------|-------|-----|----------|
| '+'  |  43   |  11 | -0.763   |
| '1'  |  49   |  17 | -0.639   |
| '2'  |  50   |  18 | -0.619   |
| '3'  |  51   |  19 | -0.598   |
| '4'  |  52   |  20 | -0.577   |
| '5'  |  53   |  21 | -0.556   |
| '='  |  61   |  29 | -0.392   |

All arithmetic chars span y ∈ [-0.77, -0.39] — only 38% of the full [-1, 1] range.

A sum of sinusoids can ALWAYS pass through any 5 arbitrary y values with
sufficient harmonics (Fourier argument). But when 5 target y values cluster in
a 0.38-wide band out of a 2.0-wide space:

1. The optimizer has to produce values with precision ≈ 0.021 (1 bucket)
2. Many parameter combinations produce values in adjacent (wrong) buckets
3. The MSE landscape has many local minima at "1 bucket off" solutions
4. These minima dominate the optimization landscape

This is fundamentally different from letter pairs ('abc', 'cba') which:
- Span higher y values (idx 65-96) where sinusoids spend more time
- Are often monotone (rising or falling), natural for sinusoids
- Have stronger phase feedback (idx 65-96 → large Δφ per step)

## What works vs what fails

| Sequence      | Type         | Loss      | Converges |
|---------------|--------------|-----------|-----------|
| 'abc'         | letters, mono| 0.000000  | ✓ 0.9s   |
| 'cba'         | letters, mono| 0.000000  | ✓ 0.6s   |
| 'aba'         | letters, non-mono | 0.000071 | ✗ wrong bucket |
| '1+1=2'       | arithmetic   | 0.000728  | ✗ always wrong |
| '1+2=3'       | arithmetic   | 0.000978  | ✗ always wrong |

## Two ways forward

### Option A: Specialized math vocabulary

Remap arithmetic chars to spread across the full y-space:

```
'0' → idx  5  (y ≈ -0.887)
'1' → idx 14  (y ≈ -0.711)
'2' → idx 24  (y ≈ -0.505)
'+' → idx 65  (y ≈ +0.351)
'=' → idx 89  (y ≈ +0.845)
```

Chars spread from -0.887 to +0.845 → full y coverage.
Arithmetic sequences would become as easy as letter sequences.
Requires a configurable vocabulary mapping in sinmachine.py.

### Option B: Accept the architecture limit and use letter pairs

The architecture WORKS for letter pairs (proven: 'abc' perfect in 0.9s).
A demo like "q"→"a" (question→answer) is meaningful and achievable.
Arithmetic requires Option A.

## Teacher forcing (attempted)

Instead of using generated token idx for phase feedback, use the CORRECT
token idx. This "decouples" the feedback error cascade:

```python
cur_phi += pf * (correct_idx / N) * 2π   # forced
```

Result: same plateau loss. The feedback is not the primary bottleneck —
the narrow y-band is.

## Key insight for architecture

SinMachine's expressivity is limited by the vocabulary's y-space distribution.
Chars with similar y values (like arithmetic chars) are hard to distinguish
because the sinusoidal function's local variations are smaller than the bucket
differences between them. 

The fix is not more harmonics or better optimizers — it's a vocabulary remap
that matches the statistical properties of the training data to the geometry
of the harmonic function's output space.
