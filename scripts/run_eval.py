"""Run the ground-truth eval and print the accuracy score.

    ./.venv/bin/python scripts/run_eval.py

Exits non-zero if any figure is trusted-but-wrong (a value contradicting the expert's ground
truth), so it can serve as a gate. Cases live in data/eval_cases.jsonl (gitignored runtime data),
captured from expert corrections.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.cases import EvalStore  # noqa: E402
from src.eval.harness import Outcome, evaluate  # noqa: E402

STORE = Path(__file__).resolve().parents[1] / "data" / "eval_cases.jsonl"


def main() -> int:
    cases = EvalStore(STORE).load()
    if not cases:
        print("No eval cases yet. They are captured from expert corrections.")
        return 0
    res = evaluate(cases)
    print(f"accuracy: {res.accuracy:.0%} ({res.matches}/{res.total} match) | "
          f"trusted-wrong: {res.trusted_wrong}")
    for r in res.results:
        if r.outcome != Outcome.MATCH:
            print(f"  [{r.outcome.value}] {r.case.company}/{r.case.figure}: {r.detail}")
    return 1 if res.trusted_wrong > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
