# 14 - Minimal Identity Abstraction Proof

## Goal

Create the smallest possible experiment showing that a SinMachine can represent
an abstract concept and apply it to inputs not present in the training dataset.

The concept is identity/copy:

```text
copy(x) = x
```

This is intentionally tiny. It is not a claim that the current trainer can
already learn arbitrary abstractions. It is a constructive proof that the
SinMachine representation can encode one abstraction and apply it outside the
examples listed in training data.

## Files

Datasets:

```text
datasets/identity-concept-train.jsonl
datasets/identity-concept-eval.jsonl
```

Model:

```text
models/identity-concept.json
```

Verifier:

```text
experiments/prove_identity_concept.py
```

Make target:

```bash
make prove-identity-concept
```

## Dataset split

Training data:

```text
0 -> 0
1 -> 1
```

Evaluation data:

```text
2 -> 2
A -> A
```

The evaluation questions `2` and `A` are not present in the training dataset.

## SinMachine model

The model is a standard SinMachine model, not a separate algorithm:

```json
{
  "harmonics": [[1.0, 1.0]],
  "dt": 6.283185307179586,
  "phase_feedback": 0.0
}
```

It uses one oscillator and samples once per full period (`2*pi`). After phase
search reconstructs the question token, the continuation lands on the same
point of the harmonic trajectory one period later. Therefore the same token is
read again.

This is the abstraction: identity is represented as periodic invariance of the
harmonic dynamics.

## Verification

Command run:

```bash
make prove-identity-concept
```

Observed result:

```text
train '0' -> '0' want '0' ok recon='0' loss=0.00000000
train '1' -> '1' want '1' ok recon='1' loss=0.00000000
eval  '2' -> '2' want '2' ok recon='2' loss=0.00000000
eval  'A' -> 'A' want 'A' ok recon='A' loss=0.00000007

train correct : 2/2
eval correct  : 2/2
eval unseen q : ['2', 'A']
error rate    : 0%
```

Acceptance threshold: this experiment is accepted at `resolution=2000` because
the measured error rate is 0%, below the 20% threshold.

The same model can also be queried through the normal SinMachine entry point:

```bash
python3 sinmachine.py --model identity-concept A
```

The answer stream repeats `A`, and the trace shows time advancing by one full
period at each answer step.

## What this proves

This proves a narrow but important point: a SinMachine harmonic model can encode
a concept that applies beyond the explicit training examples. The held-out
inputs are not memorized rows in the training dataset.

## What this does not prove

This does not prove that the current optimizer can discover this model from the
training dataset. It is a constructive representation proof, not a training
proof.

Next useful step: use this as the minimum baseline, then test whether the
trainer can recover an equivalent invariant model from
`datasets/identity-concept-train.jsonl`.

## END evolution status

After multi-END was promoted to a default vocabulary-level mechanism, END
datasets were added for the identity concept:

```text
datasets/identity-concept-end-train.jsonl
datasets/identity-concept-end-eval.jsonl
```

These expect:

```text
0 -> 0<end>
1 -> 1<end>
2 -> 2<end>
A -> A<end>
```

The verifier now understands END correctly: `query()` removes END from the
returned answer and exposes termination through `ended=True`.

The first `models/identity-concept.json` model still proves copy abstraction,
but does not terminate. A second model was then constructed:

```text
models/identity-concept-end.json
```

It uses one harmonic, vocabulary-level multi-END, a custom `perm`, and model
resolution `2000`. The construction places the concept tokens in a small region
of token space where phase feedback reflects the trajectory: first continuation
returns to the same token, second continuation lands on `<end>`.

Command run:

```bash
make prove-identity-concept-end
```

Observed result:

```text
train '0' -> '0' want '0<end>' ok ended recon='0' loss=0.00000000
train '1' -> '1' want '1<end>' ok ended recon='1' loss=0.00000000
eval  '2' -> '2' want '2<end>' ok ended recon='2' loss=0.00000000
eval  'A' -> 'A' want 'A<end>' ok ended recon='A' loss=0.00000000

train correct : 2/2
eval correct  : 2/2
eval unseen q : ['2', 'A']
```

Direct query also works through the normal SinMachine entry point:

```bash
python3 sinmachine.py --model identity-concept-end A
```

The trace shows `A` followed by `<end>`. This is the current minimal positive
example of copy-once plus termination.
