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
    # WHY (real money, UI honesty): general sector/business-model context (e.g. "banks' asset
    # quality isn't in the free feeds", "real estate normally runs higher debt") that is NOT
    # itself a cross-verified figure. Kept structurally separate from `reasons` -- which the app
    # renders under a "Why (each from cross-verified figures)" header -- so that header's claim
    # stays literally true instead of quietly including generic disclosure text alongside it.
    sector_caveats: tuple[str, ...] = ()
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

    @property
    def no_data_found(self) -> bool:
        """True when NOTHING was found for this symbol from ANY source (every figure is
        UNVERIFIABLE, or there are no figures at all -- all() on an empty tuple is True, which
        correctly counts as "found nothing" here too). WHY: distinct from ordinary thin coverage
        (some figures cross-verify, others don't) -- this specifically signals the SYMBOL itself
        is likely wrong, e.g. a company's common name differs from its exact trading ticker
        (live-verified: Page Industries trades as PAGEIND, not PAGE; typing PAGE returns zero
        data from either source, not just weak data). The UI uses this to show an actionable hint
        ("check the exact symbol") instead of the generic "insufficient data" message, which
        would otherwise look identical for a real company with genuinely poor disclosure."""
        return all(f.status == VerificationStatus.UNVERIFIABLE for f in self.figures)

    def approve(self, reviewer: str, note: str = "",
                acknowledge_conflicts: bool = False) -> "Report":
        if self.no_data_found:
            # WHY (real money, UI honesty): unlike a CONFLICT, there's no legitimate override --
            # zero data from any source almost always means the ticker itself is wrong (e.g.
            # Page Industries trades as PAGEIND, not PAGE), not a judgment call to sign off on.
            # Approving anyway would create a nonsensical "reviewed" audit entry for a thesis
            # built on nothing, and persist that same emptiness to the Sheet and the eval loop.
            raise ValueError(
                "no data at all was found for this symbol from any source; this usually means "
                "the exact ticker is wrong (e.g. Page Industries trades as PAGEIND, not PAGE) -- "
                "double-check the symbol and re-research before approving.")
        if self.conflicts and not acknowledge_conflicts:
            raise ValueError(
                f"{len(self.conflicts)} figure(s) in CONFLICT; resolve them, or approve with "
                "acknowledge_conflicts=True and a note explaining why.")
        # WHY (real money, accountability): acknowledge_conflicts is a documented human judgment
        # call overriding a genuine data disagreement -- the whole point of the audit trail. The
        # UI (app.py) shows the checkbox and the note field as two independent controls, so
        # without this a reviewer could tick "I checked ... and accept them" and approve with the
        # note left blank, leaving zero record of WHY the conflict was judged safe to override.
        if self.conflicts and acknowledge_conflicts and not note.strip():
            raise ValueError(
                "acknowledging a conflict requires a note explaining why it's safe to override "
                "(the audit trail must record the reasoning, not just that a box was checked).")
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


def most_recent_by_symbol(reports: dict[str, "Report"], symbol: str) -> "Report | None":
    """From a {key: Report} map keyed like 'SYM (live/label)' (the app's session-state reports
    dict), return the Report for `symbol` with the LATEST created_at, or None if none match.

    WHY (real money): a plain "last match found while iterating the dict" pick is NOT the most
    recently researched report. Python dict iteration order tracks INSERTION order; updating an
    EXISTING key in place does not move it. Re-researching the same symbol under a different key
    (e.g. toggling an annual-report URL override changes the label, hence the key) inserts a new,
    later key; going back to the ORIGINAL key afterward updates it in place, so it stays at its
    earlier dict position. A naive "last one seen in the loop" pick then silently returns the
    OLDER, differently-keyed report instead of the just-refreshed one. This fed both the Ask tab's
    grounding and the Invest tab's approved-name resolution (which sums real rupees), so picking by
    actual timestamp, not iteration position, matters for both.
    """
    best: Report | None = None
    for key, rep in reports.items():
        if key.split(" ")[0] == symbol and (best is None or rep.created_at > best.created_at):
            best = rep
    return best
