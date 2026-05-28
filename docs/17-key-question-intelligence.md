# 17 - Key Question: Can SinMachine Be Intelligent?

## The Question

> Does there exist a harmonic function (or sum of harmonic functions) such that
> SinMachine behaves intelligently on par with an LLM — i.e., producing coherent
> and generalising answers to arbitrary questions?

This is the investment question: if the answer is provably no, the project should
be abandoned. If the answer is yes (or "yes under certain conditions"), it is worth
continuing.

## What We Know So Far

**In favour:**
- With 8+ harmonics the model can be optimised to produce exactly the pair
  "hello→world" (the fit is underdetermined: 18 params for 11 targets).
- The identity-concept experiment showed minimal generalisation on unseen pairs
  (docs/13-14), suggesting some structure emerges from optimisation.
- The chaos-1000 model (1000 random harmonics) passively contains thousands of
  words: 24 unique words in 200k phases, 9175 two-word co-occurrences in 300k
  (experiment `chaos_listen.py` / `chaos_query.py`).

**Against:**
- The untrained chaos model failed the query pipeline completely (loss ≥ 0.02
  for every query, 0/N chars correct): chaos is not intelligence.
- Phase_feedback makes the optimisation landscape chaotic: even with large budget
  (12+ min, DE 3 seeds) the trainer does not converge for hello-world.
- It is unclear whether a single waveform can encode thousands of distinct Q→A
  pairs — parameter count grows O(N_harmonics) but constraints grow
  O(N_pairs × sequence_len).

## Path to an Answer

1. **Theoretical limit (approximation theory):**
   A sum of K sinusoids with shared phase lives in the class
   `SPAN{sin(ω₁t+φ), ..., sin(ωₖt+φ)}`. Is this class dense in the space of
   all finite token sequences? In what sense?
   → Study: density theorems for harmonic systems.

2. **Empirical scaling:**
   Train models on 1, 2, 4, 8, 16 Q→A pairs and measure:
   - Achievable loss (does it approach 0?)
   - Generalisation (does it work on unseen pairs?)
   - Computational cost (polynomial or exponential scaling?)
   → `benchmark.py` already does something close to this.

3. **Discriminating experiment:**
   Train on 10 Q→A pairs with regular structure (e.g., "letter→next letter":
   a→b, b→c, ...) and test on unseen pairs (j→k?).
   If generalisation emerges: the model learned the rule.
   If not: SinMachine is a harmonic lookup table without abstraction.

## Status (2026-05-28)

Open question. The minimum interesting result demonstrated so far:
- A single pair can in principle be learned exactly (if the optimiser converges).
- The identity concept shows a primitive form of generalisation.
- No data yet on 10+ pairs with regular structure.