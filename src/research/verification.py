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


def _agree(values: list[float], rel_tolerance: float) -> bool:
    spread = max(values) - min(values)
    scale = max(abs(statistics.fmean(values)), _EPS)
    return (spread / scale) <= rel_tolerance


def verify_figure(name: str, values: list[SourcedValue],
                  rel_tolerance: float = 0.02) -> VerifiedFigure:
    # WHY 2%: cross-verification catches gross errors (wrong scale, wrong figure, hallucination),
    # which are far larger than 2%. Two independent sources reporting a financial figure within
    # 2% is strong corroboration; sub-2% deltas are rounding/definitional noise. The expert
    # sign-off (report.py) remains the final check, so this is not the last line of defense.
    """Cross-check a figure across sources. VERIFIED needs >=2 DISTINCT sources agreeing."""
    usable = [v for v in values if v.value is not None]
    if not usable:
        return VerifiedFigure(name, VerificationStatus.UNVERIFIABLE, None, tuple(values),
                              "no usable value")

    distinct_sources = {v.source_id for v in usable}
    nums = [v.value for v in usable]

    if len(distinct_sources) < 2:
        return VerifiedFigure(name, VerificationStatus.SINGLE_SOURCE, usable[0].value,
                              tuple(values), "only one independent source; not cross-verified")

    if _agree(nums, rel_tolerance):
        return VerifiedFigure(name, VerificationStatus.VERIFIED, statistics.median(nums),
                              tuple(values), f"{len(distinct_sources)} independent sources agree")

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
