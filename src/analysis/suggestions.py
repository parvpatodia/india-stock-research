"""Rank stocks for long-term fit (the daily-suggestions engine's core).

Transparent, deterministic scoring over already-computed research: a name is eligible only if the
evidence is FAVORABLE or NEUTRAL and there is room under the per-stock cap; it scores higher for a
strong balance sheet, a cheap valuation, and improving multi-year trends. No forecasting, no LLM,
nothing hidden: the score is a small sum of the same signals shown on the stock's page, so a pick
can always be explained by its reasons. UNFAVORABLE / INSUFFICIENT_DATA names are never suggested.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sizing import Stance

_STANCE_BASE = {Stance.FAVORABLE: 2.0, Stance.NEUTRAL: 1.0}
_STRENGTH_CAP = 0.999   # < 1 so conviction only breaks ties WITHIN a whole-point flag band


@dataclass(frozen=True)
class Candidate:
    symbol: str
    stance: Stance
    quality_strong: bool
    valuation_cheap: bool
    has_room: bool          # room to add under the per-stock cap
    trend_improving: bool   # sales/profit growing or margins improving
    reason: str = ""        # one-line plain 'why', shown with the pick
    strength: float = 0.0   # [0,1] conviction (degree behind the flags); ties-only refinement


@dataclass(frozen=True)
class RankedPick:
    symbol: str
    stance: Stance
    score: float
    reason: str


def score_candidate(c: Candidate) -> float:
    """Sum of long-term-fit signals. Returns 0 for ineligible (unfavorable/insufficient).

    The coarse whole-point flags decide the band; a sub-1 conviction term (verdict_strength)
    orders names within a band so the genuinely cheaper/stronger/more-confident pick surfaces
    first instead of resolving alphabetically. Capped below 1 so it never crosses a flag band.
    """
    base = _STANCE_BASE.get(c.stance)
    if base is None:
        return 0.0
    refine = min(max(c.strength, 0.0), _STRENGTH_CAP)
    return base + c.quality_strong + c.valuation_cheap + c.has_room + c.trend_improving + refine


def rank_picks(candidates: list[Candidate]) -> list[RankedPick]:
    """Eligible names (FAVORABLE/NEUTRAL with room) ranked best-first. Names with no room to add,
    or an unfavorable/insufficient stance, are excluded, so a suggestion is always actionable."""
    picks = []
    for c in candidates:
        if c.stance not in _STANCE_BASE or not c.has_room:
            continue
        picks.append(RankedPick(c.symbol, c.stance, score_candidate(c), c.reason))
    picks.sort(key=lambda p: (-p.score, p.symbol))
    return picks
