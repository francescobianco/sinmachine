# 04 — Continuous Stream Training

## The problem with Q&A training

Training the model on explicit (question, answer) pairs might not be the right approach.

The reason: Q&A pairs impose from the outside a discrete structure
(this thing is a question, that other thing is an answer)
that may not reflect how knowledge is organised in the harmonic function.

## The alternative: continuous streams

Continuous text — books, articles, transcribed dialogues —
contains questions and answers as natural subsequences,
without them being explicitly labelled.

If the model learns to reproduce the structure of a rich enough stream,
questions and answers are both there as local patterns of the harmonic trajectory.

Training becomes:
```
given a text T of length L,
find {ωᵢ, Aᵢ, dt, feedback} such that
there exists φ₀ for which generate(φ₀, L) ≈ T
```

## The emergent behaviour hypothesis

Questions and answers — as linguistic patterns — might **accelerate emergent
behaviours** in the model.

The intuition: a question is a pattern that creates expectation, an answer is a
pattern that resolves it. If the model has seen enough of these patterns in a
continuous stream, it might have learned to connect expectation-creating structures
with resolution structures — without this being explicitly taught.

This is an open hypothesis. It is not proven.
It is worth keeping and verifying empirically.

## Practical note

The current `datasets/sample.jsonl` is structured as Q&A. This is useful for testing
the phase search mechanism. Continuous stream training is the next step.
