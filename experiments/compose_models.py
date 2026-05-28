#!/usr/bin/env python3
"""Compose two SinMachine models by harmonic concatenation (docs/18).

Usage:
    python3 experiments/compose_models.py M1 M2 --output M1_M2
    python3 experiments/compose_models.py hello-world ciao-mondo --output hello-ciao

The composed model has all harmonics of M1 + all harmonics of M2.
Renormalisation is automatic (amp_total = sum of all amplitudes).

Then runs the sinmachine query pipeline on both original questions to check
whether the composed model preserves both answers.
"""

import json, sys, math, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sinmachine import load_model, query, display_tags, MODELS_DIR, _TOKEN_DISPLAY

SEP = "─" * 66


def compose(name_a: str, name_b: str, output_name: str) -> dict:
    """Concatenate harmonics of two models into one."""
    ma = load_model(name_a)
    mb = load_model(name_b)

    # perm must be compatible (same vocab layout)
    if ma["perm"] != mb["perm"]:
        print("  WARNING: models have different perms — using perm from model A")

    harmonics = list(ma["harmonics"]) + list(mb["harmonics"])
    amp_total  = sum(a for _, a in harmonics)

    # Use average dt and phase_feedback (both should be 0 if trained with retrain script)
    dt       = (ma["dt"] + mb["dt"]) / 2
    feedback = (ma["phase_feedback"] + mb["phase_feedback"]) / 2

    composed = {
        "name":           output_name,
        "description":    f"composed: {name_a} + {name_b}  ({len(harmonics)} harmonics total)",
        "harmonics":      [[round(o, 6), round(a, 6)] for o, a in harmonics],
        "dt":             round(dt, 6),
        "phase_feedback": round(feedback, 6),
        "steps":          max(ma["steps"], mb["steps"]),
        "multi_end":      True,
        "perm":           [_TOKEN_DISPLAY.get(c, c) for c in ma["perm"]],
    }
    out_path = MODELS_DIR / f"{output_name}.json"
    with open(out_path, "w") as f:
        json.dump(composed, f, indent=2)
    print(f"  Composed model saved → {out_path}")
    print(f"  Harmonics: {len(ma['harmonics'])} + {len(mb['harmonics'])} = {len(harmonics)}")
    print(f"  dt={dt:.6f}  phase_feedback={feedback:.6f}")
    return composed


def test_composed(model, pairs: list[tuple[str, str]]) -> None:
    """Run query pipeline for each (question, expected_answer) pair."""
    print()
    print(f"  Testing composed model '{model['name']}' on {len(pairs)} pairs:")
    print()
    ok = 0
    for q, expected in pairs:
        r = query(q, model, answer_len=len(expected) + 10, resolution=4000)
        match_q = r["match"] == len(q)
        answer  = r["answer"].strip()
        # strip end tags from expected for comparison
        exp_clean = expected.replace("<end>", "").replace("<end2>", "") \
                            .replace("<end3>", "").replace("<end4>", "")
        match_a = answer.startswith(exp_clean) or exp_clean in answer
        status  = "✓" if (match_q and match_a) else ("~" if match_q else "✗")
        print(f"  {status}  q={q!r:<10}  recon={r['reconstructed']!r:<10}"
              f"  loss={r['search_loss']:.4f}  match={r['match']}/{len(q)}")
        print(f"       expected={exp_clean!r}  got={answer[:20]!r}")
        print()
        if match_q and match_a:
            ok += 1
    print(f"  Result: {ok}/{len(pairs)} pairs correct in composed model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", help="first model name")
    parser.add_argument("model_b", help="second model name")
    parser.add_argument("--output", default=None, help="output model name")
    parser.add_argument("--qa-a", nargs=2, metavar=("Q", "A"),
                        help="test pair for model A  e.g. --qa-a hello world")
    parser.add_argument("--qa-b", nargs=2, metavar=("Q", "A"),
                        help="test pair for model B  e.g. --qa-b ciao mondo")
    args = parser.parse_args()

    out_name = args.output or f"{args.model_a}+{args.model_b}"

    print(SEP)
    print(f"  COMPOSE  {args.model_a}  +  {args.model_b}  →  {out_name}")
    print(SEP)

    composed_data = compose(args.model_a, args.model_b, out_name)
    composed_model = load_model(out_name)

    pairs = []
    if args.qa_a:
        pairs.append((args.qa_a[0], args.qa_a[1]))
    if args.qa_b:
        pairs.append((args.qa_b[0], args.qa_b[1]))

    if pairs:
        test_composed(composed_model, pairs)
    else:
        print()
        print("  No test pairs specified. Use --qa-a Q A --qa-b Q A to test.")

    print(SEP)