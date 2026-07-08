"""Guided decision-support: an evidence-lean stance + transparent position sizing.

This is NOT buy/sell advice. The system never issues a buy or sell call (CLAUDE.md, report.py).
It reports which way the VERIFIED evidence leans, and does transparent arithmetic against a
per-stock cap the expert sets, so the "how much" is math with its basis shown, never a
black-box order. Every output is caveated and only becomes actionable after the expert approves
the underlying report; unreviewed or thinly-evidenced names surface as INSUFFICIENT_DATA and are
never suggested.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..constants import CONCENTRATION_TOP_HOLDING_WARN
from ..research.report import Confidence, Leaning, Verdict

# Per-stock ceiling for sizing. Reuse the concentration-flag threshold so staying within the
# cap also keeps the book below the "concentrated" warning (one source of truth). Expert-tunable.
DEFAULT_POSITION_CAP = CONCENTRATION_TOP_HOLDING_WARN

STANCE_CAVEAT = (
    "This is the direction the verified evidence leans and arithmetic against your own per-stock "
    "limit, not a buy or sell call, a price target, or a promise. Your expert approves the "
    "research first, and you decide."
)


class Stance(str, Enum):
    FAVORABLE = "favorable"                   # evidence leans in favour (cheap/fair + strong)
    NEUTRAL = "neutral"                       # evidence is mixed
    UNFAVORABLE = "unfavorable"               # evidence leans against (expensive or weak)
    INSUFFICIENT_DATA = "insufficient_data"   # not enough VERIFIED data to lean; the safety catch

    @property
    def label(self) -> str:
        return {
            "favorable": "Evidence leans favorable",
            "neutral": "Evidence is mixed / neutral",
            "unfavorable": "Evidence leans unfavorable",
            "insufficient_data": "Not enough verified data",
        }[self.value]


def stance_from_verdict(verdict: Verdict | None) -> Stance:
    """Collapse a caveated Verdict into a plain evidence-lean.

    Safety catch: a verdict with an UNKNOWN leaning, or LOW confidence (fewer than half its
    metrics cross-verified), reads as INSUFFICIENT_DATA, not a confident lean. Real money on
    shaky data must read as "we don't know", which is why half the sample portfolio lands here.
    """
    if verdict is None or verdict.leaning == Leaning.UNKNOWN:
        return Stance.INSUFFICIENT_DATA
    if verdict.confidence == Confidence.LOW:
        return Stance.INSUFFICIENT_DATA
    if verdict.leaning == Leaning.CONSTRUCTIVE:
        return Stance.FAVORABLE
    if verdict.leaning == Leaning.CAUTIOUS:
        return Stance.UNFAVORABLE
    return Stance.NEUTRAL


@dataclass(frozen=True)
class SizingAdvice:
    cap_pct: float          # per-stock ceiling as a fraction of the book (e.g. 0.25)
    cap_value: float        # rupee ceiling = cap_pct * portfolio_value
    current_value: float    # rupees currently held in this name
    room: float             # cap_value - current_value (negative => already over the cap)

    @property
    def over_cap(self) -> bool:
        return self.room < 0

    @property
    def headroom(self) -> float:
        return max(self.room, 0.0)


def position_sizing(current_value: float, portfolio_value: float,
                    cap_pct: float = DEFAULT_POSITION_CAP) -> SizingAdvice:
    """Transparent per-stock cap math. No opinion: given the book size and the expert's per-stock
    ceiling, this is the rupee room left in a name (or how far it is already over the cap)."""
    cap_value = cap_pct * portfolio_value
    return SizingAdvice(cap_pct=cap_pct, cap_value=cap_value,
                        current_value=current_value, room=cap_value - current_value)


@dataclass(frozen=True)
class AllocationCandidate:
    symbol: str
    stance: Stance
    current_value: float = 0.0   # rupees currently held (0 if not yet owned)


@dataclass(frozen=True)
class Allocation:
    symbol: str
    amount: float                # rupees suggested to add
    stance: Stance
    reason: str


@dataclass(frozen=True)
class AllocationPlan:
    amount: float                            # the lump sum asked about
    allocations: tuple[Allocation, ...]
    uninvested: float                        # rupees that could not be placed within the caps
    notes: tuple[str, ...] = ()
    caveat: str = STANCE_CAVEAT

    @property
    def invested(self) -> float:
        return sum(a.amount for a in self.allocations)


# Only FAVORABLE and NEUTRAL names are eligible; the order also ranks favorable first.
_STANCE_ORDER = {Stance.FAVORABLE: 0, Stance.NEUTRAL: 1}


def suggest_allocation(amount: float,
                       candidates: list[AllocationCandidate],
                       portfolio_value: float,
                       cap_pct: float = DEFAULT_POSITION_CAP) -> AllocationPlan:
    """Spread a lump sum across FAVORABLE (then NEUTRAL) names, each capped at the per-stock
    ceiling of the post-deploy book. UNFAVORABLE and INSUFFICIENT_DATA names are never suggested.
    Whatever cannot be placed inside the caps is reported as uninvested, never forced in.

    Callers pass only expert-APPROVED candidates; the gate is upstream (a draft is not actionable).
    """
    if amount <= 0:
        return AllocationPlan(amount=max(amount, 0.0), allocations=(), uninvested=0.0,
                              notes=("Enter an amount to invest.",))

    # WHY: cap against the eventual book size after the cash is deployed, not today's smaller book.
    base = portfolio_value + amount
    eligible = [c for c in candidates if c.stance in _STANCE_ORDER]

    def room(candidate: AllocationCandidate) -> float:
        return max(cap_pct * base - candidate.current_value, 0.0)

    # FAVORABLE before NEUTRAL; within a tier, most headroom first (deploys cash, keeps balance).
    eligible.sort(key=lambda c: (_STANCE_ORDER[c.stance], -room(c)))

    remaining = amount
    allocations: list[Allocation] = []
    for candidate in eligible:
        placeable = min(remaining, room(candidate))
        if placeable > 0:
            reason = (f"{candidate.stance.label.lower()}; room "
                      f"under a {cap_pct:.0%} per-stock cap")
            allocations.append(Allocation(candidate.symbol, placeable, candidate.stance, reason))
            remaining -= placeable
        if remaining <= 0:
            break

    notes: list[str] = []
    if not eligible:
        notes.append("No approved name is favorable or neutral, so nothing is suggested. Approve "
                     "more research, or the evidence simply does not support adding right now.")
    elif remaining > 0:
        notes.append(f"₹{remaining:,.0f} could not be placed without breaching your "
                     f"{cap_pct:.0%} per-stock cap on the eligible names. Add more approved names "
                     "or raise the cap; do not force it into one stock.")

    return AllocationPlan(amount=amount, allocations=tuple(allocations),
                          uninvested=remaining, notes=tuple(notes))
