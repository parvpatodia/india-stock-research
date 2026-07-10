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
# Margin-of-safety mapping: current P/E as a fraction of its own history median -> [0,1] degree.
# At/above _MOS_RICH (its median-plus) there is no margin of safety; at _MOS_DEEP (deeply below its
# own history) it is full. Continuous, so a deeper discount ranks above a barely-cheap name.
_MOS_RICH = 1.20   # P/E >= 120% of history -> degree 0
_MOS_DEEP = 0.40   # P/E <= 40% of history -> degree 1


def _valuation_degree(verdict: Verdict) -> float:
    ratio = verdict.valuation_ratio
    if ratio is None:                       # median unavailable -> fall back to the coarse tier
        return _VALUATION_DEGREE.get(verdict.valuation, 0.0)
    return max(0.0, min(1.0, (_MOS_RICH - ratio) / (_MOS_RICH - _MOS_DEEP)))


def verdict_strength(verdict: Verdict | None) -> float:
    """A [0,1] conviction score used ONLY to order names that already tie on the coarse signals
    (stance + strong/cheap/room/trend flags). It captures DEGREE the binary flags throw away: a
    deeply-cheap, STRONG, HIGH-confidence name outranks a barely-cheap FAIR/MIXED/MEDIUM one. The
    valuation term weighs the actual margin of safety (P/E vs its own median) when known, not just
    the CHEAP tier. Never crosses a whole-point flag band (the ranker caps it below 1), so it
    refines ties without ever promoting a weaker-evidenced name. Deterministic, no forecasting."""
    if verdict is None:
        return 0.0
    q = _QUALITY_DEGREE.get(verdict.quality, 0.0)
    v = _valuation_degree(verdict)
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
        """Even (water-filling) spread, favorable tier first then neutral: hand out the amount in
        equal shares across a tier's names, capping each at its room (cap_pct*base - current) and
        redistributing any overflow to the others, so the lump DIVERSIFIES instead of pouring into
        one name whenever the per-stock cap is loose. Whatever a tier can't absorb flows to the next."""
        cap_value = cap_pct * base
        placed: dict[str, float] = {}
        remaining = amount
        for tier in (0, 1):
            names = [c for c in eligible if _STANCE_ORDER[c.stance] == tier]
            rooms = {c.symbol: max(cap_value - c.current_value, 0.0) for c in names}
            takes = {c.symbol: 0.0 for c in names}
            active = [c for c in names if rooms[c.symbol] > 1e-9]
            while remaining > 1e-6 and active:
                share = remaining / len(active)
                still = []
                for c in active:
                    give = min(share, rooms[c.symbol] - takes[c.symbol])
                    takes[c.symbol] += give
                    remaining -= give
                    if rooms[c.symbol] - takes[c.symbol] > 1e-6:
                        still.append(c)
                if len(still) == len(active):    # no name hit its cap this round -> fully spread
                    break
                active = still
            placed.update({s: t for s, t in takes.items() if t > 1e-9})
        return [Allocation(c.symbol, placed[c.symbol], c.stance,
                           f"{c.stance.label.lower()}; spread evenly, kept under your "
                           f"{cap_pct:.0%} per-stock cap")
                for c in sorted(eligible, key=lambda c: (_STANCE_ORDER[c.stance], c.symbol))
                if c.symbol in placed]

    # WHY: cap against the TOTAL money under consideration (current holdings + the whole lump sum
    # being invested), not just whatever ends up deployed. Money left uninvested is still the
    # investor's own money, not lost or committed elsewhere -- capping against only the deployed
    # subset instead creates a self-referential trap: with few eligible names relative to a tight
    # cap (e.g. 4 names at a 20% cap -- 4*20%=80%<100%, so no allocation using ONLY these 4 names
    # can ever keep each <=20% of a book made entirely of them), each round of "less got placed"
    # shrinks the book, which shrinks the cap, spiraling toward placing almost nothing at all
    # (live-verified: ~13 rupees of a 100,000 ask) instead of the obviously sound even split.
    base = portfolio_value + amount
    allocations = allocate(base)
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

    if stance == Stance.UNFAVORABLE:
        # WHY (real money, over-confidence): reached here WITHOUT weak quality (handled above),
        # via expensive valuation alone -- assemble_verdict's leaning logic flags CAUTIOUS on
        # EXPENSIVE valuation regardless of quality, so quality here can be STRONG, MIXED, or
        # completely UNKNOWN (e.g. price data cross-verified but no quality signal did). Only
        # claim "solid business" when quality was actually verified as strong; never overstate
        # confidence in a business whose quality is merely unflagged-as-weak, or unverified.
        if verdict.quality == QualityTier.STRONG:
            opening = ("It looks like a solid business, but it's priced above its own history, "
                      "so new money would buy in expensive, not a spot to add.")
            headline = "Long-term: hold a quality business, but don't add here."
        elif verdict.quality == QualityTier.MIXED:
            opening = ("Business quality is mixed, not a red flag but not fully clean either, "
                      "and it's priced above its own history, so new money would buy in "
                      "expensive on top of that.")
            headline = "Long-term: mixed quality and priced expensive; don't add here."
        else:  # UNKNOWN -- never claim quality either way when it was never verified
            opening = ("Business quality couldn't be verified from public sources, and it's "
                      "priced above its own history, so there isn't a strong enough basis to "
                      "call this solid, or to buy in at this price.")
            headline = "Long-term: quality unverified and priced expensive; don't add here."
        pts = [opening]
        if held and sizing.over_cap:
            pts.append(f"You hold {_money(sizing.current_value)}, already over your "
                       f"{sizing.cap_pct:.0%} cap ({_money(sizing.cap_value)}); trimming toward "
                       "the cap on strength is reasonable.")
        elif held:
            pts.append("Holding for the long term is reasonable; just don't add at this price.")
        pts.append(revisit)
        return Guidance(headline, tuple(pts))

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
