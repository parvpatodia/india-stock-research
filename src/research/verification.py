"""Cross-verification of figures. The owner's first rule: every number checked twice or thrice.

A figure is trustworthy only when independent sources agree within tolerance, or a computed
identity holds (segments sum to a total, a balance balances). One source is usable but marked
not cross-verified. Disagreement is a CONFLICT: the value is withheld and flagged, never shown
as a fact. This is deterministic and has no LLM in it, so it cannot hallucinate.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum

_EPS = 1e-9


class VerificationStatus(str, Enum):
    VERIFIED = "verified"            # >= 2 independent sources agree within tolerance
    SINGLE_SOURCE = "single_source"  # one source only; usable but not cross-verified
    CONFLICT = "conflict"            # sources disagree beyond tolerance; do not trust
    UNVERIFIABLE = "unverifiable"    # no usable value at all


@dataclass(frozen=True)
class SourcedValue:
    value: float | None
    source_id: str
    locator: str = ""  # e.g. "FY24 annual report p.42"


@dataclass(frozen=True)
class VerifiedFigure:
    name: str
    status: VerificationStatus
    value: float | None                 # agreed value if VERIFIED/SINGLE_SOURCE, else None
    sources: tuple[SourcedValue, ...]
    note: str = ""

    @property
    def is_trustworthy(self) -> bool:
        # WHY: only a cross-verified figure may be stated as fact. Single-source and conflict
        # are explicitly NOT trustworthy, so the report layer cannot present them as verified.
        return self.status == VerificationStatus.VERIFIED


def _close(a: float, b: float, rel_tolerance: float) -> bool:
    return abs(a - b) <= rel_tolerance * max(abs(a), abs(b), _EPS)


def _is_clique(vals: list[float], rel_tolerance: float) -> bool:
    """True iff EVERY pair of vals is mutually within tolerance of each other -- a genuine
    clique, not merely a 'star' of values each close to one shared pivot. WHY (real money, HIGH
    severity): a star can chain together values that do not actually agree with each other (A
    close to B, B close to C, but A vs C beyond tolerance), which would wrongly report "N sources
    agree" while including a pair that genuinely disagrees -- exactly the failure this whole
    module exists to catch. Real, live scenario, not hypothetical: app.py wires yfinance +
    Screener + the annual report (a genuine 3rd source, auto-added "to break ties" whenever an
    LLM is configured), so a 3-way chained (dis)agreement is a real production possibility."""
    return all(_close(vals[i], vals[j], rel_tolerance)
              for i in range(len(vals)) for j in range(i + 1, len(vals)))


def verify_figure(name: str, values: list[SourcedValue],
                  rel_tolerance: float = 0.02) -> VerifiedFigure:
    """Cross-check a figure across sources using a consensus rule.

    VERIFIED when the largest CLIQUE of mutually-agreeing values (every pair within tolerance of
    every other, not just each close to one shared pivot -- see _is_clique) covers >= 2 DISTINCT
    sources; the value is that clique's median and any disagreeing source is named as a withheld
    outlier. This lets a third source break a two-source tie (e.g. the annual report confirms one
    of two disagreeing aggregators) without ever loosening the 2% tolerance. One source ->
    SINGLE_SOURCE; two or more with no agreeing clique -> CONFLICT. WHY 2%: gross errors (wrong
    scale/figure/hallucination) are far larger than 2%; sub-2% deltas are rounding. Expert
    sign-off is final.
    """
    usable = [v for v in values if v.value is not None]
    if not usable:
        return VerifiedFigure(name, VerificationStatus.UNVERIFIABLE, None, tuple(values),
                              "no usable value")

    distinct_sources = {v.source_id for v in usable}
    if len(distinct_sources) < 2:
        return VerifiedFigure(name, VerificationStatus.SINGLE_SOURCE, usable[0].value,
                              tuple(values), "only one independent source; not cross-verified")

    from itertools import combinations
    best_sources: set[str] = set()
    best_values: list[float] = []
    for size in range(2, len(usable) + 1):
        for combo in combinations(usable, size):
            sources = {v.source_id for v in combo}
            if len(sources) < 2 or len(sources) <= len(best_sources):
                continue
            if _is_clique([v.value for v in combo], rel_tolerance):
                best_sources = sources
                best_values = [v.value for v in combo]

    if len(best_sources) >= 2:
        outliers = sorted(distinct_sources - best_sources)
        note = f"{len(best_sources)} independent sources agree"
        if outliers:
            note += f"; outlier source(s) withheld: {', '.join(outliers)}"
        return VerifiedFigure(name, VerificationStatus.VERIFIED, statistics.median(best_values),
                              tuple(values), note)

    return VerifiedFigure(name, VerificationStatus.CONFLICT, None, tuple(values),
                          "independent sources disagree beyond tolerance")


def verify_identity(name: str, total: SourcedValue, parts: list[SourcedValue],
                    rel_tolerance: float = 0.01) -> VerifiedFigure:
    """Computed-identity check: do the parts sum to the stated total (segments -> revenue,
    line items -> balance)? Holds -> VERIFIED; not -> CONFLICT."""
    part_vals = [p.value for p in parts if p.value is not None]
    all_sources = (total, *parts)
    if total.value is None or not part_vals:
        return VerifiedFigure(name, VerificationStatus.UNVERIFIABLE, None, all_sources,
                              "missing total or parts")
    part_sum = sum(part_vals)
    scale = max(abs(total.value), _EPS)
    if abs(part_sum - total.value) / scale <= rel_tolerance:
        return VerifiedFigure(name, VerificationStatus.VERIFIED, total.value, all_sources,
                              "parts sum to the stated total")
    return VerifiedFigure(name, VerificationStatus.CONFLICT, None, all_sources,
                          f"parts sum to {part_sum:,.2f}, total states {total.value:,.2f}")
