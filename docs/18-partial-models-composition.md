# 18 - Partial Models and Composition by Mathematical Properties

## The Idea

Instead of training one model on the full dataset, train **partial models** — one
per domain or subset of Q→A pairs — and then compose them by exploiting the
mathematical properties of sinusoidal sums.

## Core Mathematical Property

The sum of two SinMachine models is still a SinMachine model:

```
y_A(t, φ) = Σᵢ Aᵢ sin(ωᵢ t + φ) / Σ Aᵢ
y_B(t, φ) = Σⱼ Bⱼ sin(ωⱼ t + φ) / Σ Bⱼ

y_{A+B}(t, φ) = (Σᵢ Aᵢ sin(ωᵢ t + φ) + Σⱼ Bⱼ sin(ωⱼ t + φ)) / (Σ Aᵢ + Σ Bⱼ)
```

That is: **just concatenate the harmonic lists** and renormalise.

## Implications

- Model A: trained on "greetings"  (hello→world, ciao→mondo, ...)
- Model B: trained on "arithmetic" (1+1=→2, 2+2=→4, ...)
- Model A+B: union of harmonic lists → handles both domains?

If the access phases (φ_A for greetings, φ_B for arithmetic) remain valid in
the composed model, composition is free — zero retraining.

## Conditions for It to Work

Composition preserves phases when the two models do not interfere:
- The harmonics of A and B are orthogonal in the sense that their access phases
  remain distinct (do not collide in φ-space).
- The target domains have characters in different y-ranges so the amplitudes
  do not disturb each other.

## Problems to Solve

1. **Phase interference:** renormalisation changes the effective amplitude of
   every harmonic when a second model is added. Phases found during training of A
   might no longer work in the composed model A+B.

2. **φ-space collision:** two different Q→A pairs might require the same φ,
   creating a conflict.

3. **Scalability:** K composed models have K × N_harmonics total harmonics.
   Inference cost (phase search) grows linearly with the total count.

## Experiment to Run

1. Train two minimal models:
   - M₁: "hello→world" (in progress — not converging yet, see docs/16)
   - M₂: "ciao→mondo"  (to train)
2. Compose M₁+M₂ by concatenating harmonics
3. Test: `query("hello", M₁₊₂)` and `query("ciao", M₁₊₂)`
4. Measure whether composition preserves both answers

This is a direct falsifiable test, runnable in a few minutes once M₁ converges.

## Connection to the Key Question (docs/17)

If composition works, SinMachine could grow by accumulation — adding domains
without global retraining. This is a structural advantage over LLMs (which require
fine-tuning on the full dataset every time).
If it does not work, phase interference becomes the fundamental limit.