# 15 - Multi-END As Default

## Decision

Multi-END should be treated as the default setup for new SinMachine experiments.

The reason is practical and structural: a single `<end>` bucket gives the
harmonic trajectory only one narrow stop region in token space. Adding multiple
END synonyms into the vocabulary permutation creates several valid stop regions
without adding a separate rule outside the model.

This is still a SinMachine-native mechanism because END remains part of the
model vocabulary (`perm`). Inference stops when the sampled token is one of the
known END chars.

## Correct implementation

The preferred implementation is vocabulary-level multi-END:

```text
<end>
<end2>
<end3>
<end4>
```

placed at different positions in `perm`.

This is better than only using an `end_zones` loss helper during training,
because `sinmachine.py` decodes and stops from the model vocabulary. If a zone
was used in loss but the saved `perm` does not contain an END synonym there,
inference will not stop.

## Consequence

New concept experiments should include multi-END in the saved model unless the
experiment is explicitly testing the single-END constraint.

For tiny abstraction experiments, multi-END is not a shortcut. It simply gives
the harmonic dynamics more legitimate places to terminate, making `<end>` less
rare in token space.
