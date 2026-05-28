# 01 — Core Hypothesis

## Starting point

Autoregressive LLMs generate text as:

```
P(token_n | token_0 ... token_n-1)
```

Each token conditions the next. Text is the cause of text.

The SinMachine hypothesis is different:

```
y(t) = Σ sin(ωᵢ t + φ)
token = Quantize(y(t))
```

Text does not cause text. Text is a **discrete projection** of an underlying
continuous dynamics.

## Parameter semantics

```
φ  — phase:     access coordinate / entry point
ω  — frequency: depth and density of structure
y  — value:     instantaneous observation
t  — time:      position in the listening
```

**Slow frequency** → broad structure, long-range relations, depth.
**Fast frequency** → local variation, fine-grained surface detail.

This mirrors something intuitive in language: high-level semantic and grammatical
structures change slowly; lexical details change rapidly.

## Generation as listening

Generation is not prediction. It is **listening** to the function from a given point.

```
φ₀ → token₁
token₁ + φ₀ → φ₁
φ₁ → token₂
...
```

The token feedback on the phase creates a self-determined trajectory: the output of
each step influences the listening point of the next.

## Theoretical goal

Not to outperform LLMs. To demonstrate that an alternative representation of
knowledge exists — harmonic, compact, deterministic — from which language can emerge.

Research question:

> Can knowledge be compressed into a harmonic function,
> and text be extracted as the sampling of that function?
