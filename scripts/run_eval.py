"""CI-style eval gate for the research pipeline (SPEC v4 §4 W5).

    ./.venv/bin/python scripts/run_eval.py

Runs three offline gates and prints a clear pass/fail summary; exits non-zero if ANY gate fails,
so it can block a build:

  1. ground-truth replay  -- no figure is trusted-but-wrong vs an expert correction
     (data/eval_cases.jsonl, gitignored runtime data captured from expert corrections).
  2. red-team resistance  -- the adversarial suite (unit trap, period mixing, phantom figure,
     as-of/staleness) is fully resisted; a regression here fails the gate.
  3. numeric-exact-match  -- every India golden-set figure matches ground truth EXACTLY after
     unit normalization (crore/lakh/million); a scale error drops the rate below threshold.

Deterministic and offline: the red-team suite injects fake LLM clients and the golden set is inline
text, so no network or key is needed and the gate runs in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.cases import EvalStore  # noqa: E402
from src.eval.harness import Outcome, evaluate  # noqa: E402
from src.eval.numeric_match import GOLDEN_FIXTURES, numeric_exact_match  # noqa: E402
from src.eval.redteam import RedTeamReport, run_redteam  # noqa: E402

STORE = Path(__file__).resolve().parents[1] / "data" / "eval_cases.jsonl"

# The numeric-exact-match gate is all-or-nothing by default: on a real-money tool a single wrong
# figure (a crore read as million) is a hard failure, not a tolerable drop.
NUMERIC_THRESHOLD = 1.0


def ground_truth_gate() -> tuple[bool, str]:
    """No figure the system TRUSTS may contradict an expert-established ground truth."""
    cases = EvalStore(STORE).load()
    if not cases:
        return True, "ground-truth: no cases yet (captured from expert corrections)"
    res = evaluate(cases)
    lines = [f"ground-truth: {res.matches}/{res.total} match, trusted-wrong={res.trusted_wrong}"]
    for r in res.results:
        if r.outcome != Outcome.MATCH:
            lines.append(f"    [{r.outcome.value}] {r.case.company}/{r.case.figure}: {r.detail}")
    return res.trusted_wrong == 0, "\n".join(lines)


def redteam_gate(report: RedTeamReport | None = None) -> tuple[bool, str]:
    """Every adversarial case must be resisted (no confident wrong number leaks through)."""
    report = report if report is not None else run_redteam()
    lines = [f"red-team: {report.resisted}/{report.total} adversarial cases resisted"]
    for o in report.outcomes:
        mark = "ok  " if o.resisted else "LEAK"
        lines.append(f"    [{mark}] {o.attack_class}: {o.detail}")
    return report.all_resisted, "\n".join(lines)


def numeric_gate(fixtures=GOLDEN_FIXTURES, threshold: float = NUMERIC_THRESHOLD) -> tuple[bool, str]:
    """Every golden figure must match ground truth EXACTLY after unit normalization."""
    res = numeric_exact_match(fixtures)
    lines = [f"numeric-exact-match: {res.matched}/{res.total} exact "
             f"({res.exact_match_rate:.0%}, threshold {threshold:.0%})"]
    for m in res.mismatches:
        lines.append(f"    [MISS] {m.fixture}/{m.figure.label}: {m.detail}")
    return res.exact_match_rate >= threshold, "\n".join(lines)


def main() -> int:
    gates = [ground_truth_gate(), redteam_gate(), numeric_gate()]
    all_ok = True
    for ok, summary in gates:
        print(summary)
        print(f"  -> {'PASS' if ok else 'FAIL'}\n")
        all_ok = all_ok and ok
    print("EVAL GATE: " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
