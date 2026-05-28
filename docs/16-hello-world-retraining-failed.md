# 16 - Hello-World Retraining Failed (2026-05-28)

## What Was Attempted

Retraining the `hello-world` model via `make hello-world`:

```
python3 trainer.py datasets/hello-world.jsonl --base dense --output hello-world --joint
```

Joint optimization: DE 3 seeds (42, 7, 137) + Nelder-Mead, base `dense` (8 harmonics),
phase_feedback free in [-0.5, 0.5].

## Result

Final loss: `0.002257`. The model generates `"fhjmq"` instead of `"hello"` — 0/5 chars
correct. Runtime: ~12 minutes at 100% CPU.

## Diagnosis

`phase_feedback = -0.5` (at the bound) causes a phase shift of ~2.4 rad per token.
Each step depends chaotically on the previous one: the optimization landscape is
highly multimodal and DE cannot find the global minimum even with a large budget.

The fitting problem is actually underdetermined (18 free parameters for 11 target
values), so exact solutions exist. The failure is not insufficient model capacity —
it is the optimization difficulty induced by phase_feedback.

## Plan for Next Session

Train with `phase_feedback = 0`. Without feedback the trajectory is a pure sum of
sinusoids: smooth landscape, well-defined gradient, fast convergence expected.
Script `retrain_hello_world.py` is already written and ready (DE 60 seeds + NM +
10k random restart). Just run it.