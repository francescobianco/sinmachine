# 09 — Vocabulary Alignment: Convergence on "1+1=2"

## The problem recap

Arithmetic chars all cluster in y ∈ [-0.77, -0.39] (38% of the [-1, 1] range).
With 2 harmonics the optimizer gets stuck at loss ≈ 0.000730 — always 1 bucket off.
No solution exists with the default ASCII linear vocab mapping.

## The insight

The optimizer doesn't fail because sinusoids can't represent the sequence.
It fails because the y-targets for arithmetic chars are so close together that
many parameter combinations land in the adjacent (wrong) bucket.

User observation: "if '1+1=3' exists, change '3' with '2' in the dictionary position."

Instead of fighting the optimizer, we let it converge to whatever it naturally
finds, then remap the vocabulary so that the model's natural output IS correct.
Progressive swapping: iterate until the permutation matches what the waveform
naturally produces.

## Implementation: vocab_align.py

State is maintained as a **full permutation array** (97-element list, perm[i] = char
at y-index i), not a dict of overrides. This guarantees the mapping is always a
valid bijection — no two chars can share the same y-index.

Each round:
1. Train with current perm (char targets use perm-based y positions)
2. Decode and compare got vs target
3. Propose swaps for each mismatch: perm[idx_got] ↔ perm[idx_target]
4. Drop conflicts (same source idx → different targets, OR different sources → same target)
5. Apply remaining swaps atomically on the array
6. Repeat

Key bug fixed in v2: the first implementation used an overrides dict, which allowed
multiple chars to be assigned to the same idx (last write wins). After round 1
with 5 simultaneous swaps, both '1' and '6' were assigned to idx 18, creating a
state that could never be resolved. The permutation array guarantees each index
is held by exactly one char.

## Experiment results

```
python3 vocab_align.py --target "1+1=2" --harmonics 2 --budget 10 --rounds 14
```

Total wall time: ~2 minutes (12 rounds × ~10s each)

| Round | Got      | Correct | Loss       |
|-------|----------|---------|------------|
| 1     | '.0355'  | 0/5     | 0.009689   |
| 2     | '2.4:3'  | 0/5     | 0.002531   |
| 3     | '.1255'  | 0/5     | 0.003888   |
| 4     | '1.355'  | 1/5     | 0.004724   |
| 5     | '.1455'  | 0/5     | 0.002967   |
| 6–8   | '12255'  | 1/5     | 0.003650   |
| 9     | '1+465'  | 2/5     | 0.003492   |
| 10    | '1+122'  | 4/5     | 0.000079   |
| 11    | '555=2'  | 2/5     | 0.000090   |
| **12**| **'1+1=2'**| **5/5**| **0.000000** |

## Final permutation

11 vocab positions were remapped from their ASCII defaults:

| idx | default | final |
|-----|---------|-------|
| 11  | '+'     | '0'   |
| 14  | '.'     | '3'   |
| 16  | '0'     | '4'   |
| 17  | '1'     | '.'   |
| 18  | '2'     | '+'   |
| 19  | '3'     | '5'   |
| 20  | '4'     | '1'   |
| 21  | '5'     | '='   |
| 22  | '6'     | '2'   |
| 26  | ':'     | '6'   |
| 29  | '='     | ':'   |

Final y positions of arithmetic chars:
- '+' → idx 18  (y ≈ -0.619)
- '1' → idx 20  (y ≈ -0.577)
- '2' → idx 22  (y ≈ -0.536)
- '=' → idx 21  (y ≈ -0.557)

The chars are now spread across 4 distinct indices (18, 20, 21, 22) instead of
clustered at 11, 17, 18, 29. More importantly, the ORDER along y matches what
the 2-harmonic waveform naturally visits — making the sequence feasible.

## What this proves

1. **The architecture is not the bottleneck.** 2 harmonics CAN encode "1+1=2".
   The obstacle was the vocabulary layout, not expressivity.

2. **Vocabulary alignment works.** Progressive swapping converges even when
   the early rounds have many conflicts and seemingly random vocab shuffling.

3. **Non-monotone sequences are feasible** with the right vocab. "1+1=2" has
   the repeating char '1' at positions 0 and 2 — the waveform visits the same
   y-bucket twice, which is natural for a harmonic function.

4. **The permutation invariant is essential.** An overrides dict allows
   inconsistency (two chars → same idx). The permutation array guarantees a
   valid bijection at every step.

## Next steps

- Apply the converged permutation to train a full model (save the perm as part
  of the model JSON, use it for both training targets and decoding)
- Test if the same permutation generalizes: can "1+2=3", "2+2=4" be trained
  with the same vocab, or does each sequence need its own alignment?
- Combine vocab alignment with multi-pair training: align vocab for the full
  simple-sums dataset, then run multijoint training
