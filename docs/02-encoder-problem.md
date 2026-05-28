# 02 — The Encoder Problem: Why SHA256 Is Wrong

## The first prototype used SHA256

```python
def encode_question(question: str) -> float:
    digest = hashlib.sha256(question.encode()).digest()
    n = int.from_bytes(digest[:4], "big")
    return (n / 0xFFFFFFFF) * 2 * math.pi
```

The question was hashed into an initial phase φ₀. The system generated the answer
starting from that φ₀.

## Why it is wrong

SHA256 breaks the internal coherence of the system.

If the answer must emerge **from the same function** from which the question emerged,
the entry point must be found *inside* that function — not assigned from outside
by an arbitrary hash function.

With SHA256:
- "what is time?" and "what is light?" could have nearby or distant phases at random
- there is no relationship between the phase and the content of the question
- the causal link question→answer is severed from the outside

## The empirical test confirmed it

When the trainer was run on Q&A pairs minimising the loss, the model parameters
collapsed pathologically (amplitudes near zero, out-of-scale frequencies, inverted
feedback) to compensate for the fact that the entry phases were arbitrary.

The model was trying to reshape the entire waveform to reach arbitrary entry points.
It is like trying to straighten a path knowing only where you arrive, without knowing
where you started from.

## The correction

The question is not an input to encode.
The question is an output of the system — it is already there, somewhere in the function.

The correct mechanism is **phase search**:
find φ_q such that `generate(φ_q) ≈ question`.

See `03-phase-search.md`.
