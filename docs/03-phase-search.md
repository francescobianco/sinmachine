# 03 — Phase Search: The Correct Inverse Mechanism

## The central idea

The question is not encoded. It is **searched for**.

The question emerged from some point in the harmonic space. Our task is to find
that point — the coherent point from which the question would have naturally come
out had we been listening to the function from there.

From that point, we keep listening. What comes after is the answer.

## Formal schema

```
search(q, model)  →  φ_q = argmin_φ  d(generate(φ, |q|), q)
simulate(φ_q, |q| steps)  →  end state: (t_end, φ_end)
simulate(t_end, φ_end, N)  →  answer
```

Question and answer are **two adjacent segments of the same continuous trajectory**.

## Why it works (in principle)

The search space is compact: φ ∈ [0, 2π].
This is a 1D problem. We are not optimising model parameters —
we are searching for the entry point into a function that already exists.

Grid search + local refinement is sufficient for a prototype.

## The periodicity property

The harmonic function is periodic. This has an important consequence:

**If a sequence does not exist in the function, no search will ever find it.**

The loss stays high. This is not a system error — it is information:
the model does not have that sequence in its structure.
This is a natural test of **model expressivity**.

Note: the trajectory is not strictly periodic because token feedback creates
output-dependent dynamics (a discrete dynamical system). It could be periodic,
eventually periodic, or aperiodic. But the space of φ₀ values to search is always
[0, 2π] — the search is always finite and complete.

## The backward search

If you search for "ciao" starting from "o" going backwards — you search for the
points that generate "o" as the last character, then among those find the ones that
have "a" as the second-to-last, and so on.

This is an inverse search over the trajectory. More expensive but possible.
It could be useful for understanding the internal structure of the model.

## Consequences for training

Training is no longer:
```
loss = d(generate(SHA256(q)), a)          ← arbitrary phase, wrong
```

But:
```
φ_q* = argmin_φ d(generate(φ, |q|), q)   ← inner search
loss  = d(generate_continuation(φ_q*), a) ← outer loss
```

This is a bilevel optimisation problem. Computationally heavier but semantically correct.
