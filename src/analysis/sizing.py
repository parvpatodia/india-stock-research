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
from ..research.report import Confidence, Leaning, QualityTier, ValuationTier, Verdict

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


# Expert-tunable degree weights for ranking refinement. Business quality and margin of safety
# (cheapness) are the substance of a long-term value pick; confidence modulates how much to lean
# on it. Kept here beside stance_from_verdict because both derive plain signals from a Verdict.
_QUALITY_DEGREE = {QualityTier.STRONG: 1.0, QualityTier.MIXED: 0.5,
                   QualityTier.WEAK: 0.0, QualityTier.UNKNOWN: 0.0}
_VALUATION_DEGREE = {ValuationTier.CHEAP: 1.0, ValuationTier.FAIR: 0.5,
                     ValuationTier.EXPENSIVE: 0.0, ValuationTier.UNKNOWN: 0.0}
_CONFIDENCE_DEGREE = {Confidence.HIGH: 1.0, Confidence.MEDIUM: 0.6, Confidence.LOW: 0.2}
_STRENGTH_WEIGHTS = (0.40, 0.35, 0.25)  # (quality, valuation, confidence); sum to 1 -> result in [0,1]


def verdict_strength(verdict: Verdict | None) -> float:
    """A [0,1] conviction score used ONLY to order names that already tie on the coarse signals
    (stance + strong/cheap/room/trend flags). It captures DEGREE the binary flags throw away: a
    CHEAP+STRONG+HIGH-confidence name outranks a FAIR+MIXED+MEDIUM one. Never crosses a whole-point
    flag band (the ranker caps it below 1), so it refines ties without ever promoting a
    weaker-evidenced name. Deterministic, explainable, no forecasting."""
    if verdict is None:
        return 0.0
    q = _QUALITY_DEGREE.get(verdict.quality, 0.0)
    v = _VALUATION_DEGREE.get(verdict.valuation, 0.0)
    c = _CONFIDENCE_DEGREE.get(verdict.confidence, 0.0)
    wq, wv, wc = _STRENGTH_WEIGHTS
    return wq * q + wv * v + wc * c


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

    eligible = [c for c in candidates if c.stance in _STANCE_ORDER]

    def allocate(base: float) -> list[Allocation]:
        """Greedy fill: each name's addition capped so its final value <= cap_pct * base."""
        def room(candidate: AllocationCandidate) -> float:
            return max(cap_pct * base - candidate.current_value, 0.0)
        ordered = sorted(eligible, key=lambda c: (_STANCE_ORDER[c.stance], -room(c)))
        remaining = amount
        placed: list[Allocation] = []
        for candidate in ordered:
            take = min(remaining, room(candidate))
            if take > 0:
                placed.append(Allocation(
                    candidate.symbol, take, candidate.stance,
                    f"{candidate.stance.label.lower()}; kept under your {cap_pct:.0%} per-stock cap"))
                remaining -= take
            if remaining <= 0:
                break
        return placed

    # WHY: cap against the REALIZED book (current holdings + what actually gets placed), not the
    # optimistic full-deploy book. If the caps bind, less is placed and the book is smaller, so
    # re-solve until the base is self-consistent -> no name can exceed cap_pct of the book it ends
    # up in (the same holdings basis the Concentration tab uses). Converges monotonically.
    base = portfolio_value + amount
    allocations: list[Allocation] = []
    for _ in range(40):
        allocations = allocate(base)
        placed_total = sum(a.amount for a in allocations)
        new_base = portfolio_value + placed_total
        if abs(new_base - base) < 1.0:
            break
        base = new_base
    remaining = amount - sum(a.amount for a in allocations)

    notes: list[str] = []
    if not eligible:
        notes.append("No approved name is favorable or neutral, so nothing is suggested. Approve "
                     "more research, or the evidence simply does not support adding right now.")
    elif remaining > 1.0:
        notes.append(f"₹{remaining:,.0f} could not be placed without pushing a name past your "
                     f"{cap_pct:.0%} per-stock cap of the resulting book. Add more approved names, "
                     "raise the cap, or leave the rest as cash; do not force it into one stock.")

    return AllocationPlan(amount=amount, allocations=tuple(allocations),
                          uninvested=max(remaining, 0.0), notes=tuple(notes))


@dataclass(frozen=True)
class Guidance:
    """Long-term (not trading) hold / trim / accumulate guidance with thesis-based triggers."""
    headline: str
    points: tuple[str, ...]


def _money(x: float) -> str:
    return f"₹{x:,.0f}"


def long_term_guidance(stance: Stance, sizing: SizingAdvice, verdict: Verdict | None,
                       held: bool) -> Guidance:
    """Turn the stance + cap math + verdict into plain long-term guidance. Thesis-based, never a
    dated sell/buy call: it says what to do now and the conditions under which to revisit."""
    if verdict is None or stance == Stance.INSUFFICIENT_DATA:
        return Guidance(
            "Not enough verified data to guide a decision.",
            ("Don't act on this until the figures cross-verify across sources, or your expert "
             "reviews and approves it. Withholding is the safe choice here.",))

    expensive = verdict.valuation == ValuationTier.EXPENSIVE
    cheap_or_fair = verdict.valuation in (ValuationTier.CHEAP, ValuationTier.FAIR)
    weak = verdict.quality == QualityTier.WEAK

    # Conditions to revisit, drawn from what the verdict actually flagged.
    triggers = []
    if expensive:
        triggers.append("the price stays well above its own history")
    if verdict.quality in (QualityTier.WEAK, QualityTier.MIXED):
        triggers.append("cash conversion, margins, or debt get worse")
    if not triggers:
        triggers.append("the fundamentals or valuation change materially")
    revisit = "Revisit if " + ", or ".join(triggers) + "."

    if weak:
        return Guidance(
            "Long-term: review whether the thesis still holds.",
            ("The concern here is business quality, not just price, and for a long-term holding "
             "weak fundamentals matter more than a cheap price.",
             "Decide if the reasons you own it still apply. " + revisit))

    if stance == Stance.UNFAVORABLE:  # strong business but expensive
        pts = ["It looks like a solid business, but it's priced above its own history, so new "
               "money would buy in expensive, not a spot to add."]
        if held and sizing.over_cap:
            pts.append(f"You hold {_money(sizing.current_value)}, already over your "
                       f"{sizing.cap_pct:.0%} cap ({_money(sizing.cap_value)}); trimming toward "
                       "the cap on strength is reasonable.")
        elif held:
            pts.append("Holding for the long term is reasonable; just don't add at this price.")
        pts.append(revisit)
        return Guidance("Long-term: hold a quality business, but don't add here.", tuple(pts))

    if stance == Stance.FAVORABLE and cheap_or_fair:
        if sizing.over_cap:
            return Guidance(
                "Long-term: positive, but you're already at your cap.",
                (f"You hold {_money(sizing.current_value)}, at/over your {sizing.cap_pct:.0%} "
                 f"limit ({_money(sizing.cap_value)}). The thesis is good, but adding more "
                 "concentrates risk. Hold; rebalance toward the cap if you wish.",
                 revisit))
        return Guidance(
            "Long-term: reasonable to accumulate, within your cap.",
            (f"The evidence is favorable and there's about {_money(sizing.headroom)} of room "
             f"before your {sizing.cap_pct:.0%} per-stock limit.",
             "Add gradually over time rather than all at once, you're investing long term, not "
             "timing the market.",
             revisit))

    return Guidance(
        "Long-term: hold.",
        ("Nothing in the verified data points to adding or trimming right now.", revisit))
