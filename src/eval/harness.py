"""Replay ground-truth cases against the pipeline and score accuracy.

Each case is replayed on its stored figure snapshot (deterministic, no network). Outcomes:
- MATCH: the system trusts a value that matches the expert's ground truth.
- WITHHELD: the system did not trust the figure (conflict/single-source). Not a mistake: it
  asserted nothing false.
- TRUSTED_WRONG: the system trusts a value that contradicts ground truth. This is the mistake
  the loop exists to catch; it must stay at zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..pipeline import build_company_report
from ..research.report import Report
from ..research.verification import SourcedValue
from .cases import GroundTruth


class Outcome(str, Enum):
    MATCH = "match"
    WITHHELD = "withheld"
    TRUSTED_WRONG = "trusted_wrong"
    MISSING = "missing"


def _within(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


@dataclass
class CaseResult:
    case: GroundTruth
    outcome: Outcome
    system_value: float | None
    detail: str


@dataclass
class EvalResult:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def matches(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.MATCH)

    @property
    def trusted_wrong(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.TRUSTED_WRONG)

    @property
    def withheld(self) -> int:
        """Figures the system correctly refused to trust (conflict/single-source). The SAFE
        outcome: it asserted nothing false, so this is never counted as an error."""
        return sum(1 for r in self.results if r.outcome == Outcome.WITHHELD)

    @property
    def missing(self) -> int:
        return sum(1 for r in self.results if r.outcome == Outcome.MISSING)

    @property
    def accuracy(self) -> float:
        """Coverage-weighted correctness: matches / total. Counts safe WITHHOLDS in the
        denominator, so it reads LOW for a (correctly) cautious run -- use trusted_accuracy for the
        parent-facing headline, and keep this only as a coverage signal."""
        return self.matches / self.total if self.total else 1.0

    @property
    def trusted_accuracy(self) -> float:
        """Precision over the figures the system actually TRUSTED: matches / (matches +
        trusted_wrong). WHY (real money, honest metric): a WITHHELD figure (sources conflicted, so
        the system refused to assert a value) is the SAFE outcome, not an error, so it must not drag
        the headline correctness number the way coverage-based `accuracy` does -- that would read a
        cautious, correct run as inaccurate to a non-expert. This is 1.0 exactly while trusted_wrong
        is 0 (the goal), and 1.0 vacuously when nothing was trusted (it asserted nothing false)."""
        trusted = self.matches + self.trusted_wrong
        return self.matches / trusted if trusted else 1.0


def ground_truth_from_report(report: Report, figure: str, correct_value: float,
                             tolerance: float = 0.02, note: str = "",
                             reviewer: str = "") -> GroundTruth:
    """Capture a ground-truth case from a report: the correct value plus the figure snapshot
    the system used, so it can be replayed deterministically later."""
    snapshot = {f.name: [[sv.value, sv.source_id, sv.locator] for sv in f.sources]
                for f in report.figures}
    return GroundTruth(company=report.company, figure=figure, correct_value=correct_value,
                       snapshot=snapshot, tolerance=tolerance, note=note, reviewer=reviewer)


def _snapshot_to_figures(snapshot: dict[str, list]) -> dict[str, list[SourcedValue]]:
    return {name: [SourcedValue(value=row[0], source_id=row[1],
                                locator=row[2] if len(row) > 2 else "")
                   for row in rows]
            for name, rows in snapshot.items()}


def evaluate(cases: list[GroundTruth]) -> EvalResult:
    results: list[CaseResult] = []
    for case in cases:
        report = build_company_report(case.company, _snapshot_to_figures(case.snapshot))
        figure = next((f for f in report.figures if f.name == case.figure), None)
        if figure is None:
            results.append(CaseResult(case, Outcome.MISSING, None, "figure not in report"))
        elif not figure.is_trustworthy:
            results.append(CaseResult(case, Outcome.WITHHELD, None,
                                      "system did not trust this figure (not a false assertion)"))
        elif _within(figure.value, case.correct_value, case.tolerance):
            results.append(CaseResult(case, Outcome.MATCH, figure.value,
                                      "trusted value matches ground truth"))
        else:
            results.append(CaseResult(case, Outcome.TRUSTED_WRONG, figure.value,
                                      f"trusted {figure.value:,.0f} but ground truth is "
                                      f"{case.correct_value:,.0f}"))
    return EvalResult(results)
