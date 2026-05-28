# 13 - Interrupted Held-Out Generalization Attempts

## Why this note exists

This project uses `docs/` as an experiment diary, not only as polished
documentation. Failed and interrupted attempts must stay visible because they
show which claims have not been proven yet.

This note records an interrupted attempt to demonstrate SinMachine
generalization. The important correction is that the first implementation was
outside the SinMachine hypothesis. It used an explicit cyclic-successor
algorithm, not a harmonic SinMachine model. That was off-topic and was removed
from the model path.

## Correct criterion

The goal is not to show that some algorithm can generalize. The goal is to show
that a SinMachine can generalize.

For this repo, that means the proof must use the existing SinMachine mechanics:

```text
models/*.json with harmonics, dt, phase_feedback, optional perm
phase search over the question
harmonic continuation for the answer
datasets/*.jsonl for train/eval pairs
```

A separate hand-coded rule such as "successor is rotation on ten digits" does
not count, even if it predicts a held-out case correctly.

## Attempt A - explicit cyclic-successor model

A script was created that learned one parameter `theta` on a circular digit
space:

```text
0> -> 1
1> -> 2
...
8> -> 9
held out: 9> -> 0
```

It correctly predicted `9> -> 0`, and `0` was not present in the training
answers. However, this was not evidence for SinMachine generalization because
the cyclic structure was put directly into the algorithm. It bypassed the
harmonic model, phase search, and continuation dynamics.

Decision: discard this as a proof.

## Attempt B - SinMachine next-digit with delimiter

Datasets were created:

```text
datasets/next-digit-heldout-train.jsonl
datasets/next-digit-heldout-eval.jsonl
```

Training set:

```text
0> -> 1
1> -> 2
...
8> -> 9
```

Evaluation holdout:

```text
9> -> 0
```

A lightweight benchmark was run with:

```bash
python3 benchmark.py datasets/next-digit-heldout-train.jsonl \
  --base dense --max 9 --de-iters 300 --de-pop 8 --seeds 42
```

Result: the model did not even fit the training set. At `N=9` it reached
`0/9` exact pairs. The frequent confusion around the second question character
(`>`, `=`, `?`, `A`, etc.) suggests the default ASCII vocabulary geometry makes
this version too hard for the current trainer/budget.

Decision: this does not prove or disprove generalization; it only shows that
this setup is not yet a viable demonstration.

## Attempt C - SinMachine next-digit without delimiter

Datasets were created:

```text
datasets/next-digit-single-heldout-train.jsonl
datasets/next-digit-single-heldout-eval.jsonl
```

Training set:

```text
0 -> 1
1 -> 2
...
8 -> 9
```

Evaluation holdout:

```text
9 -> 0
```

A lightweight benchmark was run with:

```bash
python3 benchmark.py datasets/next-digit-single-heldout-train.jsonl \
  --base dense --max 9 --de-iters 300 --de-pop 8 --seeds 42
```

Observed result: early small subsets sometimes fit, but the full `N=9` training
set did not. At `N=9` it reached `4/9` exact pairs.

Decision: still not a valid generalization proof. The model must first fit the
training set before an unseen case can mean anything.

## Attempt D - identity-bits heldout

Because `identity-bits` was the strongest micro-model in earlier notes, a
smaller heldout direction was started. Datasets were created:

```text
datasets/identity-bits-heldout-train.jsonl
datasets/identity-bits-heldout-eval.jsonl
datasets/identity-bits-combo-heldout-train.jsonl
datasets/identity-bits-combo-heldout-eval.jsonl
```

The first split held out `11 -> 11` after training on the other simple identity
examples. A benchmark with a larger lightweight budget fit the first four
progressive subsets, but at `N=5` reached only `4/5`; it failed one training
pair (`00 -> 00` decoded as `00 -> 10`).

A smaller combinatorial split was then prepared:

```text
train: 0->0, 1->1, 00->00, 01->01
eval:  10->10
```

This experiment was interrupted before a final saved SinMachine model and
holdout result were produced.

## Lessons

1. A proof must use SinMachine's harmonic dynamics, not a separate algorithm.
2. Train/eval datasets belong in `datasets/`, even for failed attempts.
3. A heldout answer is meaningful only after exact training reconstruction is
   achieved.
4. `docs/` must record failed attempts because they define the boundary of what
   has actually been shown.
5. The next useful step is likely a focused heldout experiment on
   `identity-bits` or a vocabulary-permutation experiment, not `next-digit` with
   the default ASCII geometry.
