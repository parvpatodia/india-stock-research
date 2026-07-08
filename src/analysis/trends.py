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


def _word(rate: float) -> str:
    return "growing" if rate > 3 else "shrinking" if rate < -3 else "roughly flat"


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
        if prof[0] > rev[0] + 2:
            points.append("Profit has grown faster than sales, so margins have been improving.")
        elif prof[0] < rev[0] - 2:
            points.append("Profit has grown slower than sales, so margins have been under pressure.")
    return points
