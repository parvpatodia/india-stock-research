"""Multi-year trend analysis.

What several years of financials reveal that one year hides: sales and profit growth (CAGR) and
whether margins have been improving or under pressure. Each year's value is cross-verified across
sources first (>=2 distinct sources agreeing within tolerance), so a trend is built only from
values that agree, never a single source's uncorroborated series. Deterministic, no LLM.
"""
from __future__ import annotations

import statistics


def verified_series(per_source: dict[str, dict[int, float]],
                    rel_tol: float = 0.02) -> dict[int, float]:
    """Cross-verify a per-year series across sources. For each fiscal year, keep the value only if
    >=2 distinct sources agree within rel_tol (median of the agreeing cluster); else drop the year.

    per_source: {source_id: {year: value}}.
    """
    years: set[int] = set()
    for yearmap in per_source.values():
        years |= set(yearmap)
    out: dict[int, float] = {}
    for year in years:
        values = [ym[year] for ym in per_source.values() if ym.get(year) is not None]
        if len(values) < 2:
            continue
        best: list[float] | None = None
        for pivot in values:
            cluster = [v for v in values
                       if abs(v - pivot) <= rel_tol * max(abs(v), abs(pivot), 1e-9)]
            if len(cluster) >= 2 and (best is None or len(cluster) > len(best)):
                best = cluster
        if best:
            out[year] = statistics.median(best)
    return out


def cagr(series: dict[int, float]) -> tuple[float, int] | None:
    """Compound annual growth rate (%) from the earliest to latest year, and the span in years.
    Needs >=3 years and positive endpoints (CAGR is undefined through zero/negative)."""
    if len(series) < 3:
        return None
    years = sorted(series)
    first, last = years[0], years[-1]
    v0, v1, span = series[first], series[last], last - first
    if v0 <= 0 or v1 <= 0 or span <= 0:
        return None
    rate = ((v1 / v0) ** (1 / span) - 1) * 100
    return rate, span


# Single source of truth for the trend thresholds (%/yr), shared by the prose and the structured
# signal so the scoring flag can never drift from the words shown on the page.
_GROWTH_MIN = 3.0   # CAGR above this reads as "growing" / counts as improving
_MARGIN_MIN = 2.0   # profit CAGR exceeding sales CAGR by this reads as margins improving


def _word(rate: float) -> str:
    return "growing" if rate > _GROWTH_MIN else "shrinking" if rate < -_GROWTH_MIN else "roughly flat"


def trend_improving(revenue_series: dict[int, float],
                    profit_series: dict[int, float]) -> bool:
    """Structured multi-year signal for the ranker: True iff sales OR profit have compounded above
    the growth floor, or margins have been improving. WHY (real money): the suggestion score must
    read this from the numbers, not by substring-matching the plain-language insight prose, so a
    wording change can never silently flip a scoring input. Shares the thresholds with trend_points
    so the flag and the words always agree. Needs >=3 cross-verified years (see cagr) or returns
    False — no history, no claimed trend."""
    rev = cagr(revenue_series)
    prof = cagr(profit_series)
    growing = (rev is not None and rev[0] > _GROWTH_MIN) or (prof is not None and prof[0] > _GROWTH_MIN)
    margins_up = rev is not None and prof is not None and prof[0] > rev[0] + _MARGIN_MIN
    return bool(growing or margins_up)


# Max-min spread of year-over-year PROFIT growth rates (percentage points) beyond which earnings
# read as cyclical/lumpy rather than smoothly compounding. Live-verified: JSW Steel swung +115%
# then -61% YoY (175pp); TCS swung only 8pp; HUL sat right at this 40pp boundary (one unusual
# year), a reasonable case to flag either way.
_VOLATILITY_SWING = 40.0

# REVENUE swings LESS than profit for the same underlying volatility (operating leverage
# amplifies revenue moves into larger profit moves), so a lower, separately-calibrated threshold
# is needed or genuine lumpiness in project-based revenue would go undetected. Live-verified
# across 8 real names: a clean gap separates smooth names (TCS 2.3pp, HUL 6.0pp, Oberoi Realty
# 10.5pp, JSW Steel's REVENUE specifically 13.1pp -- confirming operating leverage: its PROFIT
# swings 175pp but its revenue only 13pp) from three independent real-estate developers
# clustering tightly at 37-39pp (Brigade, DLF, Sobha, percentage-of-completion revenue
# recognition). 25.0 sits with wide margin on both sides of that real-data gap.
_REVENUE_VOLATILITY_SWING = 25.0


def _yoy_swing(series: dict[int, float]) -> tuple[float, int] | None:
    """Max-min spread of year-over-year growth rates (percentage points) plus the year count, or
    None if there are fewer than 2 usable growth rates. A zero-base year is skipped, not divided
    by, so this never crashes or fabricates a rate. Shared by the profit and revenue volatility
    checks so their swing math can never drift apart."""
    years = sorted(series)
    growths: list[float] = []
    for a, b in zip(years, years[1:]):
        v0, v1 = series[a], series[b]
        if v0 == 0:
            continue
        growths.append((v1 - v0) / abs(v0) * 100)
    if len(growths) < 2:
        return None
    return max(growths) - min(growths), len(years)


def earnings_volatility_point(profit_series: dict[int, float]) -> str | None:
    """A caveat when profit has swung sharply year to year, common in cyclical/commodity
    businesses and lumpy, project-based revenue recognition (e.g. real estate), where a single
    year's ROE/margin reading can badly misrepresent long-term earning power. WHY (no blind
    spots): the ratio suite (ROE, margins, ...) is computed from the LATEST year only; without
    this, a cyclical name at a temporary peak or trough would look uniformly strong/weak with no
    signal that the reading is timing-dependent."""
    result = _yoy_swing(profit_series)
    if result is None:
        return None
    swing, n_years = result
    if swing < _VOLATILITY_SWING:
        return None
    return (f"Profit has swung sharply year to year (a {swing:.0f}-percentage-point range in "
            f"annual growth over the last {n_years} years) — common in cyclical or "
            f"project-based businesses. A single year's ROE/margin may not represent its "
            f"long-term earning power; weigh the multi-year trend, not just the latest year.")


def revenue_volatility_point(revenue_series: dict[int, float]) -> str | None:
    """The same construct as earnings_volatility_point, for REVENUE. WHY: live-verified against
    real Brigade Enterprises data (a real-estate developer): 4 cross-verified REVENUE years
    swinging sharply, but only 1 cross-verified PROFIT year (percentage-of-completion accounting
    makes profit recognition lumpier and harder to cross-verify year to year), so
    earnings_volatility_point can NEVER fire for a name like this even though the real lumpiness
    is right there in the revenue data. trend_points prefers the profit signal when both fire
    (the bottom line is more decision-relevant) and falls back to this one only when profit data
    is too thin to judge, so the reader is never shown two near-duplicate sentences."""
    result = _yoy_swing(revenue_series)
    if result is None:
        return None
    swing, n_years = result
    if swing < _REVENUE_VOLATILITY_SWING:
        return None
    return (f"Revenue has swung sharply year to year (a {swing:.0f}-percentage-point range in "
            f"annual growth over the last {n_years} years) — common in project-based or "
            f"cyclical businesses (e.g. real estate revenue recognized by project completion). "
            f"A single year's figures may not represent the long-term run rate.")


def trend_points(revenue_series: dict[int, float],
                 profit_series: dict[int, float]) -> list[str]:
    """Plain-language multi-year track-record points, from cross-verified yearly series."""
    points: list[str] = []
    rev = cagr(revenue_series)
    prof = cagr(profit_series)
    if rev:
        rate, span = rev
        points.append(f"Track record: sales have been {_word(rate)} about {rate:.0f}% a year over "
                      f"the last {span} years (cross-verified).")
    if prof:
        rate, span = prof
        points.append(f"Track record: profit has been {_word(rate)} about {rate:.0f}% a year over "
                      f"the last {span} years.")
    if rev and prof:
        if prof[0] > rev[0] + _MARGIN_MIN:
            points.append("Profit has grown faster than sales, so margins have been improving.")
        elif prof[0] < rev[0] - _MARGIN_MIN:
            points.append("Profit has grown slower than sales, so margins have been under pressure.")
    # Prefer the profit signal (the bottom line, more decision-relevant); fall back to revenue
    # only when profit data is too thin to judge (see revenue_volatility_point's WHY). Never both,
    # to avoid two near-duplicate sentences.
    volatility = earnings_volatility_point(profit_series) or revenue_volatility_point(revenue_series)
    if volatility:
        points.append(volatility)
    return points
