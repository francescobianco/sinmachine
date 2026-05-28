# 05 — Hello World: The Minimum Demonstrable Goal

## The challenge

Find (or train) a model such that:

```
input:  "hello"
output: "world"
```

This is the minimum Turing test for the SinMachine.

## What must happen mechanically

1. `search_phase("hello", model)` → finds φ_h such that `generate(φ_h, 5) ≈ "hello"`
2. From φ_h, advance 5 steps → end state `(t_end, φ_end)`
3. `generate(t_end, φ_end, 5)` → must produce `"world"`

## Why it is non-trivial

With the current model (sum of sinusoids with fixed parameters) there is no
structural reason why the continuation of "hello" should be "world".
The function knows nothing of hello or world.

The problem has two levels:

**Level 1 — expressivity**: does there exist φ_h such that `generate(φ_h, 5) = "hello"` exactly?
With the current model probably not — the search loss will be > 0.

**Level 2 — coherence**: even having found an approximate φ_h, the continuation
produces something random, not "world".

## How to solve it

Training must optimise {ωᵢ, Aᵢ, dt, feedback} so that:

```
∃ φ_h : generate(φ_h, 5) = "hello"  AND  generate_continuation(φ_h, 5, 5) = "world"
```

This is the bilevel problem introduced in `03-phase-search.md`, applied to the
minimum case.

## Why this goal is useful

- It is verifiable: the result is binary (works / does not work)
- It is minimal: 10 characters total, a single pair
- If it works, it proves that the harmonic structure can encode at least one semantic relation
- If it does not work, it tells us where the bottleneck is (expressivity? depth? feedback?)

## The model name

The model that satisfies this constraint is named `hello-world`.
