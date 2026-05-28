#!/usr/bin/env python3
"""Verify a tiny SinMachine abstraction experiment.

The script does not implement the copy rule. It only loads a standard
SinMachine model and checks train/eval JSONL pairs through sinmachine.query().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sinmachine import _ALL_END_CHARS, display_tags, load_model, query


def load_pairs(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                yield obj["q"], obj["a"]


def expected_answer(want: str) -> tuple[str, bool]:
    chars = []
    ended = False
    for ch in want:
        if ch in _ALL_END_CHARS:
            ended = True
            break
        chars.append(ch)
    return "".join(chars), ended


def check_split(label: str, pairs, model, resolution: int) -> tuple[int, int]:
    ok_count = 0
    total = 0
    for q, want in pairs:
        want_answer, want_ended = expected_answer(want)
        result = query(q, model, answer_len=max(len(want), 1), resolution=resolution)
        got = result["answer"][:len(want_answer)]
        ok = (
            result["reconstructed"] == q
            and got == want_answer
            and (not want_ended or result["ended"])
        )
        total += 1
        ok_count += int(ok)
        mark = "ok" if ok else "no"
        end_note = " ended" if result["ended"] else " open"
        print(
            f"{label:<5} {display_tags(q)!r} -> {display_tags(got)!r} "
            f"want {display_tags(want)!r} {mark}{end_note} "
            f"recon={display_tags(result['reconstructed'])!r} loss={result['search_loss']:.8f}"
        )
    return ok_count, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove minimal SinMachine identity abstraction")
    parser.add_argument("--model", default="identity-concept")
    parser.add_argument("--train", default="datasets/identity-concept-train.jsonl")
    parser.add_argument("--eval", default="datasets/identity-concept-eval.jsonl")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-error-rate", type=float, default=0.0,
                        help="accept if total error rate is <= this threshold")
    args = parser.parse_args()

    model = load_model(args.model)
    train_pairs = list(load_pairs(Path(args.train)))
    eval_pairs = list(load_pairs(Path(args.eval)))
    train_questions = {q for q, _ in train_pairs}

    print("SinMachine minimal abstraction proof")
    print(f"model      : {args.model}")
    print(f"train data : {args.train}")
    print(f"eval data  : {args.eval}")
    print(f"concept    : identity/copy")
    print()

    train_ok, train_total = check_split("train", train_pairs, model, args.resolution)
    eval_ok, eval_total = check_split("eval", eval_pairs, model, args.resolution)

    unseen = [q for q, _ in eval_pairs if q not in train_questions]
    print()
    print(f"train correct : {train_ok}/{train_total}")
    print(f"eval correct  : {eval_ok}/{eval_total}")
    print(f"eval unseen q : {unseen}")

    total_ok = train_ok + eval_ok
    total = train_total + eval_total
    error_rate = 1.0 - (total_ok / max(total, 1))
    print(f"error rate    : {error_rate:.0%}")

    if error_rate > args.max_error_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
