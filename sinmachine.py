#!/usr/bin/env python3
"""SinMachine — harmonic generative model prototype.

The question is not encoded. It is searched for: we find the phase φ ∈ [0, 2π]
from which it would have naturally emerged in the harmonic function.
From that point we continue reading to obtain the answer.

When the END token is produced, the machine stops listening.
"""

import json
import math
import pathlib
import sys

MODELS_DIR = pathlib.Path(__file__).parent / "models"

# ── token space ───────────────────────────────────────────────────
#
# 95 printable ASCII (32–126) + 2 special tokens at the top of the y range:
#   index 95  →  START  (y ≈ +0.969)
#   index 96  →  END    (y ≈ +0.990)
#
# Physically: START and END live near y = +1, the positive extreme of the wave.

_START_CHAR = '\x02'   # STX — beginning of sequence
_END_CHAR   = '\x03'   # ETX — end of sequence

VOCAB = [chr(i) for i in range(32, 127)] + [_START_CHAR, _END_CHAR]
_VOCAB_SIZE = len(VOCAB)   # 97

START_IDX = VOCAB.index(_START_CHAR)   # 95
END_IDX   = VOCAB.index(_END_CHAR)     # 96

# Human-readable labels for display
_DISPLAY = {_START_CHAR: '<S>', _END_CHAR: '<E>'}


# ── model ─────────────────────────────────────────────────────────

def load_model(name: str = "default") -> dict:
    path = MODELS_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    harmonics = [tuple(h) for h in data["harmonics"]]
    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "harmonics": harmonics,
        "amp_total": sum(a for _, a in harmonics),
        "dt": data["dt"],
        "phase_feedback": data["phase_feedback"],
        "steps": data["steps"],
    }


# ── core: harmonic function ───────────────────────────────────────

def harmonic(t: float, phi: float, model: dict) -> float:
    """y(t) = Σ Aᵢ sin(ωᵢ t + φ), normalised to [-1, 1]"""
    y = sum(a * math.sin(omega * t + phi) for omega, a in model["harmonics"])
    return y / model["amp_total"]


def map_to_token(y: float) -> int:
    idx = int((y + 1) / 2 * _VOCAB_SIZE)
    return max(0, min(_VOCAB_SIZE - 1, idx))


def update_phase(phi: float, token_idx: int, phase_feedback: float) -> float:
    return phi + phase_feedback * (token_idx / _VOCAB_SIZE) * 2 * math.pi


# ── multi-END zones ───────────────────────────────────────────────
#
# A model can define multiple END zones spread across the y-space.
# Each zone is a (center_y, half_width) pair; any y inside stops generation.
# Default: single high zone (backward-compatible).
# Defined AFTER char_to_y so we can use it for the default center.

_END_ZONE_HALF = 3.0 / _VOCAB_SIZE   # ≈3 token buckets wide

_MULTI_END_ZONES = [
    (-0.90, _END_ZONE_HALF),   # END_LOW  — below arithmetic chars
    (+0.40, _END_ZONE_HALF),   # END_MID  — above arithmetic chars
    (+0.90, _END_ZONE_HALF),   # END_HIGH — near the sinusoidal peak
]


def char_to_y(c: str) -> float:
    """Char → centre of its quantisation bucket in [-1, 1].

    Handles special tokens START and END explicitly.
    Uses bucket centre so MSE training pushes y into the correct discrete region.
    """
    if c == _START_CHAR:
        idx = START_IDX
    elif c == _END_CHAR:
        idx = END_IDX
    else:
        idx = max(0, min(_VOCAB_SIZE - 1, ord(c) - 32))
    return ((idx + 0.5) / _VOCAB_SIZE) * 2 - 1


_DEFAULT_END_ZONES = [(char_to_y(_END_CHAR), _END_ZONE_HALF)]


def get_end_zones(model: dict) -> list:
    return model.get("end_zones", _DEFAULT_END_ZONES)


def is_end_y(y: float, model: dict) -> bool:
    return any(abs(y - ctr) <= hw for ctr, hw in get_end_zones(model))


# ── simulation ────────────────────────────────────────────────────

def _simulate(t0: float, phi0: float, model: dict, steps: int,
              stop_on_end: bool = False):
    """Run the sampler from (t0, phi0). Returns (trace, t_final, phi_final).

    If stop_on_end is True, halts as soon as the END token is produced.
    """
    phi = phi0
    t = t0
    trace = []
    for _ in range(steps):
        y = harmonic(t, phi, model)
        idx = map_to_token(y)
        trace.append((t, y, idx, VOCAB[idx]))
        if stop_on_end and is_end_y(y, model):
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
    """MSE between generated y values and target y values."""
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

    The question is not encoded — it is searched for. This is the inverse mechanism.
    If the sequence does not exist in the function, the loss stays high, signalling
    insufficient model expressivity rather than a system error.
    """
    targets_y = [char_to_y(c) for c in target]

    best_phi = 0.0
    best_loss = float("inf")
    for i in range(resolution):
        phi = (i / resolution) * 2 * math.pi
        loss = _phase_loss(phi, targets_y, model)
        if loss < best_loss:
            best_loss = loss
            best_phi = phi

    # local refinement
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
          answer_len: int = 40, resolution: int = 2000) -> dict:
    """Find the coherent phase from which the question emerged, then read the answer.

    Generation stops automatically when the END token is produced.

    Pipeline:
        search(q)  → φ_q, loss
        simulate(φ_q, |q| steps) → question reconstruction + end state (t_end, φ_end)
        simulate(t_end, φ_end, answer_len, stop_on_end=True) → answer
    """
    phi_q, search_loss = search_phase(question, model, resolution)
    trace_q, t_end, phi_end = _simulate(0.0, phi_q, model, len(question))
    reconstructed = "".join(c for _, _, _, c in trace_q)
    trace_a, _, _ = _simulate(t_end, phi_end, model, answer_len, stop_on_end=True)

    # strip the END token from display, keep it in trace
    answer_chars = [c for _, _, idx, c in trace_a if idx != END_IDX]
    answer = "".join(answer_chars)

    match = sum(1 for a, b in zip(reconstructed, question) if a == b)
    ended = any(idx == END_IDX for _, _, idx, _ in trace_a)
    return {
        "phi_q": phi_q,
        "search_loss": search_loss,
        "reconstructed": reconstructed,
        "match": match,
        "answer": answer,
        "ended": ended,
        "trace_q": trace_q,
        "trace_a": trace_a,
    }


# ── stream inference: scan-based retrieval ───────────────────────

def stream_query(question: str, model: dict, scan_steps: int = None) -> dict:
    """Retrieve the answer by scanning the model's generated text stream.

    Used with stream-trained models: run the model from _stream_phi for
    many steps, then find the question as a substring and read what follows.
    No phase search involved — O(1) per query after the model is run.
    """
    if scan_steps is None:
        stream = model.get("_stream", "")
        scan_steps = max(model["steps"], len(stream) * 2)

    phi0 = model.get("_stream_phi", 0.0)
    trace, _, _ = _simulate(0.0, phi0, model, scan_steps, stop_on_end=False)
    text = "".join(c for _, _, _, c in trace)

    pos = text.find(question)
    if pos == -1:
        return {"found": False, "answer": "", "text": text, "pos": -1}

    answer_chars = []
    answer_start = pos + len(question)
    for _, _, idx, c in trace[answer_start:]:
        if idx == END_IDX:
            break
        answer_chars.append(c)

    return {
        "found": True,
        "answer": "".join(answer_chars),
        "text": text,
        "pos": pos,
    }


# ── display ───────────────────────────────────────────────────────

def ascii_wave(trace: list, width: int = 44) -> str:
    lines = []
    for t, y, idx, char in trace:
        pos = int((y + 1) / 2 * (width - 1))
        bar = " " * pos + "●"
        display = _DISPLAY.get(char, repr(char) if char == " " else char)
        label = f"[{idx+32:3d}]" if idx < 95 else f"[{_DISPLAY[char]}]"
        lines.append(f"  t={t:5.2f}  y={y:+.3f}  {label} {display:<5} {bar}")
    return "\n".join(lines)


def respond(question: str, model: dict) -> None:
    sep = "─" * 68
    print(f"\n  Searching φ_q for: '{question}' ...")
    result = query(question, model)

    ended_mark = "  ■ END" if result["ended"] else ""
    print(f"\n  Model      : {model['name']}  ({len(model['harmonics'])} harmonics)")
    print(f"  φ_q found  : {result['phi_q']:.6f} rad  ({math.degrees(result['phi_q']):.1f}°)")
    print(f"  Search loss: {result['search_loss']:.6f}")
    print(f"  Reconstructed : '{result['reconstructed']}'")
    print(f"  Original      : '{question}'")
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
    print(f"\n{sep}")
    print(f"  SinMachine  —  chat mode  [{model['name']}]  (exit to quit)")
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
        print(f"  {path.stem:<12}  {n} harmonics  —  {data.get('description', '')}")


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
