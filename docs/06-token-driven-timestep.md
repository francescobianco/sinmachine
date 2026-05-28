# 06 — Token-Driven Time Step

## The idea

Instead of advancing time by a fixed `dt`:

```
t → t + dt          # current: uniform sampling
```

The token index drives the next time position:

```
t → t + f(token_idx)   # proposed: token-determined step
```

The token no longer just encodes a character — it also determines **when the next
reading will happen**. Rhythm is intrinsic to the sequence itself.

## Two feedback channels

The current system has one feedback channel:

```
token_idx → Δφ   (phase feedback, already present)
```

With token-driven time step, there are two:

```
token_idx → Δφ   (phase feedback)
token_idx → Δt   (time feedback — new)
```

The trajectory becomes a curve in (t, φ) space that self-modifies along both axes.
The next sample location depends on what the current sample produced.

## Emergent rhythm

Low-index characters (space, punctuation, low ASCII) → small time step → dense,
rapid reading. High-index characters (uppercase, lowercase, special tokens) →
large time step → sparse, slow reading.

The reading rhythm is not imposed from outside but emerges from the content.
This is closer to how music works: each note has both pitch and duration.

## Scaling options

Using `token_idx` directly (0–96) risks aliasing with high-frequency harmonics.
For a harmonic with ω=13, one period is 2π/13 ≈ 0.48 units. A step of 96 spans
~200 periods. Possible scaling approaches:

```python
# A — normalised to [0, dt_max]
t_step = (token_idx / _VOCAB_SIZE) * dt_max

# B — offset + token contribution
t_step = dt_base * (1 + token_idx / _VOCAB_SIZE)

# C — literal (pure, requires low-frequency harmonics)
t = t + token_idx
```

Option C is the purest. It requires the model to be calibrated with frequencies
low enough that steps of O(100) don't cause aliasing.

## Consequence for the phase search

The phase search still works the same way — we search φ₀ ∈ [0, 2π].
But now the trajectory in time is non-uniform, making the dynamics richer and
potentially harder to optimise. The model must learn both what to say (token value)
and implicitly when to say the next thing (token index).

## Relation to START and END

With token-driven time steps:
- START (idx 95) → large time step into the first reading
- END (idx 96) → largest possible step — but there is no next reading, so it
  acts purely as a termination signal

This is consistent with the intended semantics: START and END are at the top of
the index space, giving them the largest temporal weight.

## Open question

Does the variable time step create richer or more constrained expressivity?
With uniform dt, the trajectory samples a fixed grid of the harmonic function.
With token-driven dt, the trajectory samples an adaptive grid — potentially
capturing more structure in fewer steps, or creating more complex interference
patterns.

This is to be verified empirically.
