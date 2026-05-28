#!/usr/bin/env python3
"""SinMachine — harmonic generative model prototype.

The question is not encoded. It is searched for: we find the phase φ ∈ [0, 2π]
from which it would have naturally emerged in the harmonic function.
From that point we continue reading to obtain the answer.

The vocabulary is part of the model: each model carries a 'perm' (permutation
of 97 chars) that defines the y-space ↔ char mapping used for both training
and decoding.  Special tokens are written as HTML-like tags:
  <start>  →  '\x02'  (STX, beginning of sequence)
  <end>    →  '\x03'  (ETX, primary end-of-sequence)
  <end2>   →  '\x1c'  (FS,  end synonym 2)
  <end3>   →  '\x1d'  (GS,  end synonym 3)
  <end4>   →  '\x1e'  (RS,  end synonym 4)

Multi-END is a vocabulary decision: placing multiple end-synonym chars at
different positions in the perm gives the waveform several y-regions that
all decode as "stop".  No separate end_zones mechanism is needed.
"""

import json
import math
import pathlib
import sys

MODELS_DIR = pathlib.Path(__file__).parent / "models"

# ── special tokens ────────────────────────────────────────────────
#
# Tags are the human-readable form; internal chars are what live in the perm.
# Use parse_tags() / display_tags() to convert between the two.

_START_CHAR = '\x02'   # <start>
_END_CHAR   = '\x03'   # <end>  (primary END)

SPECIAL_TOKENS = {
    '<start>': '\x02',
    '<end>':   '\x03',
    '<end2>':  '\x1c',
    '<end3>':  '\x1d',
    '<end4>':  '\x1e',
}
_TOKEN_DISPLAY = {v: k for k, v in SPECIAL_TOKENS.items()}   # char → tag

# All chars that are end-of-sequence synonyms (by default just the primary)
_ALL_END_CHARS = {'\x03', '\x1c', '\x1d', '\x1e'}


def parse_tags(s: str) -> str:
    """'hello<end>' → 'hello\x03'  (tag → internal char)"""
    for tag, char in SPECIAL_TOKENS.items():
        s = s.replace(tag, char)
    return s


def display_tags(s: str) -> str:
    """'hello\x03' → 'hello<end>'  (internal char → tag)"""
    for char, tag in _TOKEN_DISPLAY.items():
        s = s.replace(char, tag)
    return s


# ── default vocabulary ────────────────────────────────────────────
#
# 95 printable ASCII chars (32–126) + <start> + <end> = 97 slots.
# <start> sits at idx 95 (y ≈ +0.969), <end> at idx 96 (y ≈ +0.990).
# Models may replace this with a custom permutation.

VOCAB = [chr(i) for i in range(32, 127)] + [_START_CHAR, _END_CHAR]
_VOCAB_SIZE = len(VOCAB)   # 97

START_IDX = VOCAB.index(_START_CHAR)   # 95
END_IDX   = VOCAB.index(_END_CHAR)     # 96

# Multi-END is the default vocabulary geometry for new experiments.
# These indices are spread across y-space and keep the primary <end> at idx 96.
_DEFAULT_MULTI_END_IDXS = [4, 67, 92]
_END_ZONE_HALF = 3.0 / _VOCAB_SIZE   # ≈3 token buckets wide


def build_default_perm(multi_end: bool = True) -> list[str]:
    """Build the default model vocabulary permutation.

    With multi_end=True, several END synonyms are placed directly in token
    space. This is the runtime mechanism: inference stops because decoded
    tokens are END chars, not because of a separate loss-only zone list.
    """
    perm = list(VOCAB)
    if multi_end:
        aliases = ['\x1c', '\x1d', '\x1e']
        for alias, pos in zip(aliases, _DEFAULT_MULTI_END_IDXS):
            try:
                alias_pos = perm.index(alias)
                perm[alias_pos], perm[pos] = perm[pos], perm[alias_pos]
            except ValueError:
                perm[pos] = alias
    return perm


def y_center(idx: int) -> float:
    return ((idx + 0.5) / _VOCAB_SIZE) * 2 - 1


def end_indices(perm: list[str]) -> list[int]:
    return [i for i, c in enumerate(perm) if c in _ALL_END_CHARS]


def char_to_idx(c: str, perm: list[str] = None) -> int:
    if perm is None:
        perm = build_default_perm()
    try:
        return perm.index(c)
    except ValueError:
        return max(0, min(_VOCAB_SIZE - 1, ord(c) - 32))


# ── model ─────────────────────────────────────────────────────────

def load_model(name: str = "default") -> dict:
    path = MODELS_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    harmonics = [tuple(h) for h in data["harmonics"]]
    perm = [parse_tags(c) for c in data.get("perm", build_default_perm(data.get("multi_end", True)))]

    # Derive end_chars from the perm itself: any slot holding a known end synonym
    end_chars = {c for c in perm if c in _ALL_END_CHARS}

    return {
        "name":           data["name"],
        "description":    data.get("description", ""),
        "harmonics":      harmonics,
        "amp_total":      sum(a for _, a in harmonics),
        "dt":             data["dt"],
        "phase_feedback": data["phase_feedback"],
        "steps":          data["steps"],
        "perm":           perm,
        "end_chars":      end_chars,
        **{k: data[k] for k in ("_stream_phi", "_stream", "_training_phis", "resolution")
           if k in data},
    }


# ── core: harmonic function ───────────────────────────────────────

def harmonic(t: float, phi: float, model: dict) -> float:
    """y(t) = Σ Aᵢ sin(ωᵢ t + φ), normalised to [-1, 1]"""
    y = sum(a * math.sin(omega * t + phi) for omega, a in model["harmonics"])
    return y / model["amp_total"]


def map_to_token(y: float) -> int:
    idx = int((y + 1) / 2 * _VOCAB_SIZE)
    return max(0, min(_VOCAB_SIZE - 1, idx))


def model_char_to_y(c: str, model: dict) -> float:
    """char → y bucket centre using the model's vocab permutation."""
    perm = model.get("perm", build_default_perm())
    idx = char_to_idx(c, perm)
    return y_center(idx)


def update_phase(phi: float, token_idx: int, phase_feedback: float) -> float:
    return phi + phase_feedback * (token_idx / _VOCAB_SIZE) * 2 * math.pi


def is_end_char(c: str, model: dict) -> bool:
    """True if char c is any end-of-sequence token for this model."""
    return c in model.get("end_chars", {_END_CHAR})


# ── simulation ────────────────────────────────────────────────────

def char_to_y(c: str, perm: list[str] = None) -> float:
    """char -> y bucket centre using the default multi-END perm unless provided."""
    return y_center(char_to_idx(c, perm))


# Backward-compat constants for trainer.py / benchmark.py
_DEFAULT_END_ZONES  = [(y_center(END_IDX), _END_ZONE_HALF)]
_MULTI_END_ZONES    = [(y_center(i), _END_ZONE_HALF) for i in end_indices(build_default_perm())]


def _simulate(t0: float, phi0: float, model: dict, steps: int,
              stop_on_end: bool = False):
    """Run the sampler from (t0, phi0). Returns (trace, t_final, phi_final)."""
    perm = model.get("perm", build_default_perm())
    phi = phi0
    t = t0
    trace = []
    for _ in range(steps):
        y = harmonic(t, phi, model)
        idx = map_to_token(y)
        c = perm[idx]
        trace.append((t, y, idx, c))
        if stop_on_end and is_end_char(c, model):
            break
        phi = update_phase(phi, idx, model["phase_feedback"])
        t += model["dt"]
    return trace, t, phi


def generate(phi0: float, model: dict, steps: int = None) -> list[tuple]:
    """Generate from an explicit initial phase."""
    if steps is None:
        steps = model["steps"]
    trace, _, _ = _simulate(0.0, phi0, model, steps)
    return trace


# ── phase search ──────────────────────────────────────────────────

def _phase_loss(phi: float, targets_y: list, model: dict) -> float:
    t = 0.0
    current_phi = phi
    total = 0.0
    for y_tgt in targets_y:
        y = harmonic(t, current_phi, model)
        total += (y - y_tgt) ** 2
        idx = map_to_token(y)
        current_phi = update_phase(current_phi, idx, model["phase_feedback"])
        t += model["dt"]
    return total / len(targets_y)


def search_phase(target: str, model: dict, resolution: int = 2000) -> tuple[float, float]:
    """Search for φ ∈ [0, 2π] from which the text would naturally emerge.

    The question is not encoded — it is searched for.  High loss means the
    sequence doesn't exist in the current waveform, not a system error.
    """
    targets_y = [model_char_to_y(c, model) for c in target]

    best_phi, best_loss = 0.0, float("inf")
    for i in range(resolution):
        phi = (i / resolution) * 2 * math.pi
        loss = _phase_loss(phi, targets_y, model)
        if loss < best_loss:
            best_loss = loss
            best_phi = phi

    step = (2 * math.pi) / resolution
    for _ in range(8):
        step /= 4
        improved = False
        for delta in [-step * 2, -step, step, step * 2]:
            phi = best_phi + delta
            loss = _phase_loss(phi, targets_y, model)
            if loss < best_loss:
                best_loss = loss
                best_phi = phi
                improved = True
        if not improved:
            break

    return best_phi, best_loss


# ── query: full pipeline ──────────────────────────────────────────

def query(question: str, model: dict,
          answer_len: int = 40, resolution: int = None) -> dict:
    """Find the coherent phase from which the question emerged, then read the answer.

    Tags in the question are parsed automatically (<end> etc.).
    The answer is returned with tags substituted for special chars.
    """
    if resolution is None:
        resolution = model.get("resolution", 2000)
    question = parse_tags(question)
    phi_q, search_loss = search_phase(question, model, resolution)
    trace_q, t_end, phi_end = _simulate(0.0, phi_q, model, len(question))
    reconstructed = "".join(c for _, _, _, c in trace_q)
    trace_a, _, _ = _simulate(t_end, phi_end, model, answer_len, stop_on_end=True)

    answer_chars = [c for _, _, _, c in trace_a if not is_end_char(c, model)]
    answer = display_tags("".join(answer_chars))
    reconstructed_display = display_tags(reconstructed)

    match = sum(1 for a, b in zip(reconstructed, question) if a == b)
    ended = any(is_end_char(c, model) for _, _, _, c in trace_a)
    return {
        "phi_q":          phi_q,
        "search_loss":    search_loss,
        "reconstructed":  reconstructed_display,
        "match":          match,
        "answer":         answer,
        "ended":          ended,
        "trace_q":        trace_q,
        "trace_a":        trace_a,
    }


# ── stream inference: scan-based retrieval ───────────────────────

def stream_query(question: str, model: dict, scan_steps: int = None) -> dict:
    """Retrieve the answer by scanning the model's generated text stream."""
    if scan_steps is None:
        stream = model.get("_stream", "")
        scan_steps = max(model["steps"], len(stream) * 2)

    phi0 = model.get("_stream_phi", 0.0)
    trace, _, _ = _simulate(0.0, phi0, model, scan_steps, stop_on_end=False)
    text = "".join(c for _, _, _, c in trace)

    question_internal = parse_tags(question)
    pos = text.find(question_internal)
    if pos == -1:
        return {"found": False, "answer": "", "text": text, "pos": -1}

    answer_chars = []
    for _, _, _, c in trace[pos + len(question_internal):]:
        if is_end_char(c, model):
            break
        answer_chars.append(c)

    return {
        "found":  True,
        "answer": display_tags("".join(answer_chars)),
        "text":   text,
        "pos":    pos,
    }


# ── display ───────────────────────────────────────────────────────

def ascii_wave(trace: list, width: int = 44) -> str:
    lines = []
    for t, y, idx, char in trace:
        pos = int((y + 1) / 2 * (width - 1))
        bar = " " * pos + "●"
        tag = _TOKEN_DISPLAY.get(char)
        display = tag if tag else (repr(char) if char == " " else char)
        label   = f"[{tag:<6}]" if tag else f"[{idx:3d}]   "
        lines.append(f"  t={t:5.2f}  y={y:+.3f}  {label} {display:<8} {bar}")
    return "\n".join(lines)


def respond(question: str, model: dict) -> None:
    sep = "─" * 68
    print(f"\n  Searching φ_q for: {display_tags(question)!r} ...")
    result = query(question, model)

    ended_mark = "  ■ <end>" if result["ended"] else ""
    print(f"\n  Model      : {model['name']}  ({len(model['harmonics'])} harmonics)")
    print(f"  φ_q found  : {result['phi_q']:.6f} rad  ({math.degrees(result['phi_q']):.1f}°)")
    print(f"  Search loss: {result['search_loss']:.6f}")
    print(f"  Reconstructed : {result['reconstructed']!r}")
    print(f"  Original      : {display_tags(question)!r}")
    print(f"  Match         : {result['match']}/{len(question)} chars")
    print(f"\n  — question trajectory —")
    print(ascii_wave(result["trace_q"]))
    print(f"\n  — answer trajectory —{ended_mark}")
    print(ascii_wave(result["trace_a"]))
    print(sep)
    print(f"  ◉  {result['answer']}")
    print(f"{sep}\n")


def chat(model: dict) -> None:
    sep = "─" * 68
    end_tags = ", ".join(_TOKEN_DISPLAY.get(c, c)
                         for c in sorted(model.get("end_chars", {_END_CHAR})))
    print(f"\n{sep}")
    print(f"  SinMachine  —  chat mode  [{model['name']}]  (exit to quit)")
    print(f"  Vocab: {len(model['perm'])} slots  |  End tokens: {end_tags}")
    print(sep + "\n")
    while True:
        try:
            question = input("  ▶  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  bye.\n")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("\n  bye.\n")
            break
        respond(question, model)


def list_models() -> None:
    for path in sorted(MODELS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        n = len(data["harmonics"])
        has_perm = "perm" in data
        marker = "[perm]" if has_perm else "[multi-END]"
        end_note = ""
        if has_perm:
            perm = [parse_tags(c) for c in data["perm"]]
            end_tags = [_TOKEN_DISPLAY.get(c, c) for c in perm if c in _ALL_END_CHARS]
            end_note = f"  end_chars={end_tags}"
        print(f"  {path.stem:<14}  {n} harmonics  {marker:<11}  "
              f"{data.get('description', '')}{end_note}")


# ── entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list-models" in args:
        list_models()
        sys.exit(0)

    model_name = "default"
    if "--model" in args:
        i = args.index("--model")
        model_name = args[i + 1]
        args = args[:i] + args[i + 2:]

    model = load_model(model_name)

    if "--chat" in args or not args:
        chat(model)
    else:
        question = " ".join(a for a in args if not a.startswith("--"))
        sep = "─" * 68
        print(f"\n{sep}")
        print("  SinMachine  —  phase search prototype")
        print(sep)
        respond(question, model)
