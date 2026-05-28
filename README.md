# SinMachine

An experimental harmonic generative model. A research prototype.

> Language as discrete sampling of a continuous harmonic dynamics.

## Hypothesis

Classical LLMs generate text autoregressively:

```
P(token_n | token_0 ... token_n-1)
```

SinMachine attempts a different approach:

```
y(t) = Σ Aᵢ sin(ωᵢ t + φ)
token = Quantize(y(t))
```

Text is not the cause of text. Text is a **discrete projection** of a continuous
underlying dynamics. A question is not encoded into a phase — it is **searched for**:
we find the phase φ from which the question would have naturally emerged, then
continue reading from there to obtain the answer.

```
search(question)  →  φ_q              coherent entry point
simulate(φ_q, |q| steps)  →  end state (t_end, φ_end)
simulate(t_end, φ_end, N)  →  answer
```

## Current state

This is an early prototype. The core machinery works:

- Phase search over [0, 2π] finds the coherent origin of a text fragment
- Multi-harmonic function with configurable depth (number of harmonics)
- Bilevel training: inner phase search + outer answer-continuation loss
- Token space: printable ASCII 32–126

The open research question is whether a harmonic model can be trained to encode
semantic relations — starting with the minimum goal: `"hello" → "world"`.

## Usage

```bash
make chat                          # interactive chat (default model)
make chat MODEL=sparse             # chat with a specific model
make run Q="what is light?"        # single question
make run Q="hello" MODEL=dense
make train                         # train on datasets/sample.jsonl
make train DATASET=datasets/hello-world.jsonl BASE=default OUTPUT=hello-world
make list-models                   # show available models
```

Or directly:

```bash
python3 sinmachine.py --chat --model default
python3 sinmachine.py --model sparse "hello"
python3 trainer.py datasets/hello-world.jsonl --base default --output hello-world
```

## Structure

```
sinmachine/
├── sinmachine.py          core: harmonic function, phase search, query pipeline
├── trainer.py             bilevel inverse optimisation
├── Makefile
│
├── models/
│   ├── default.json       4 harmonics (ω = 1, 3, 7, 13)
│   ├── sparse.json        1 harmonic — minimal oscillator
│   ├── dense.json         8 harmonics (primes up to 17)
│   └── hello-world.json   [to be trained]
│
├── datasets/
│   ├── sample.jsonl       10 Q&A pairs for general testing
│   └── hello-world.jsonl  single pair: "hello" → "world"
│
└── docs/                  reasoning diary
    ├── 00-preface.md
    ├── 01-core-hypothesis.md
    ├── 02-encoder-problem.md
    ├── 03-phase-search.md
    ├── 04-continuous-stream-training.md
    └── 05-hello-world-minimum-goal.md
```

## Models

| name          | harmonics | description |
|---------------|-----------|-------------|
| `default`     | 4         | prime-ish frequencies 1, 3, 7, 13 |
| `sparse`      | 1         | single oscillator, regular trajectory |
| `dense`       | 8         | richer spectrum, primes up to 17 |
| `hello-world` | TBD       | to be found/trained |

## Docs

The `docs/` directory is a thinking diary, not technical documentation.
It records reasoning, architectural decisions, and open hypotheses as they emerge.

## Requirements

```bash
pip install scipy   # optional, improves training (falls back to hill climbing)
```

No other dependencies beyond the Python standard library.
