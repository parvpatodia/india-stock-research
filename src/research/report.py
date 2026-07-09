"""Report + expert-review lifecycle. The safety gate.

A report is DRAFT until the human expert approves it; only APPROVED is trusted. Parents see
approved reports; a draft is clearly labeled unreviewed. Rejections capture corrections that
feed the eval/regression loop. Approval is blocked while any figure is in CONFLICT unless the
expert explicitly acknowledges it. The verdict is always a caveated opinion, never certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum

from .claims import Claim
from .verification import VerificationStatus, VerifiedFigure

VERDICT_CAVEAT = (
    "This verdict is a caveated opinion drawn from the cited sources, not a recommendation, "
    "prediction, or guarantee. Markets are uncertain. Your expert reviews and approves before "
    "any decision, and you decide."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ReviewStatus(str, Enum):
    DRAFT = "draft"          # generated, not yet reviewed
    APPROVED = "approved"    # expert approved; trusted
    REJECTED = "rejected"    # expert rejected with corrections


class ValuationTier(str, Enum):
    CHEAP = "cheap"
    FAIR = "fair"
    EXPENSIVE = "expensive"
    UNKNOWN = "unknown"


class QualityTier(str, Enum):
    STRONG = "strong"
    MIXED = "mixed"
    WEAK = "weak"
    UNKNOWN = "unknown"


class Leaning(str, Enum):
    CONSTRUCTIVE = "constructive"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Verdict:
    valuation: ValuationTier
    quality: QualityTier
    leaning: Leaning
    confidence: Confidence
    reasons: tuple[str, ...] = ()   # each reason should reference cited/verified figures
    caveat: str = VERDICT_CAVEAT    # always present; the verdict is never shown as certainty
    # current P/E as a fraction of its own historical median (margin-of-safety magnitude); None
    # when the median is unavailable. Lets the ranker weigh HOW cheap, not just the CHEAP tier.
    valuation_ratio: float | None = None


@dataclass(frozen=True)
class ReviewEvent:
    status: ReviewStatus
    reviewer: str
    timestamp: str
    note: str = ""
    corrections: tuple[str, ...] = ()  # populated on rejection; feeds the eval loop


@dataclass(frozen=True)
class Report:
    company: str
    claims: tuple[Claim, ...] = ()
    figures: tuple[VerifiedFigure, ...] = ()
    verdict: Verdict | None = None
    status: ReviewStatus = ReviewStatus.DRAFT
    audit: tuple[ReviewEvent, ...] = ()
    created_at: str = field(default_factory=_now)
    # Plain-language "why" points (deterministic, from cross-verified figures). Presentation
    # only; kept as strings so this module needs no import from the analysis layer.
    insights: tuple[str, ...] = ()
    # Structured multi-year signal for the ranker (sales/profit growing or margins improving),
    # computed from cross-verified series — NOT parsed from the prose above, so a wording change
    # can never flip a scoring input. Plain bool: no analysis-layer import needed.
    trend_improving: bool = False

    @property
    def is_trusted(self) -> bool:
        # WHY: the hard safety gate (owner's locked decision). Nothing is trusted, and nothing
        # should reach the parents as reviewed, until the human expert has approved it.
        return self.status == ReviewStatus.APPROVED

    @property
    def conflicts(self) -> tuple[VerifiedFigure, ...]:
        return tuple(f for f in self.figures if f.status == VerificationStatus.CONFLICT)

    @property
    def uncrossverified(self) -> tuple[VerifiedFigure, ...]:
        """Figures not cross-verified (conflict or single-source). The review panel surfaces
        these so the expert never signs off blind."""
        return tuple(f for f in self.figures if not f.is_trustworthy)

    def approve(self, reviewer: str, note: str = "",
                acknowledge_conflicts: bool = False) -> "Report":
        if self.conflicts and not acknowledge_conflicts:
            raise ValueError(
                f"{len(self.conflicts)} figure(s) in CONFLICT; resolve them, or approve with "
                "acknowledge_conflicts=True and a note explaining why.")
        return self._transition(ReviewStatus.APPROVED, reviewer, note)

    def reject(self, reviewer: str, note: str = "",
               corrections: tuple[str, ...] = ()) -> "Report":
        return self._transition(ReviewStatus.REJECTED, reviewer, note, tuple(corrections))

    def _transition(self, status: ReviewStatus, reviewer: str, note: str,
                    corrections: tuple[str, ...] = ()) -> "Report":
        if not reviewer or not reviewer.strip():
            raise ValueError("a review action requires a named reviewer (accountability)")
        event = ReviewEvent(status=status, reviewer=reviewer.strip(), timestamp=_now(),
                            note=note, corrections=corrections)
        return replace(self, status=status, audit=self.audit + (event,))
