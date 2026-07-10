"""Multi-year trend analysis.

What several years of financials reveal that one year hides: sales and profit growth (CAGR) and
whether margins have been improving or under pressure. Each year's value is cross-verified across
sources first (>=2 distinct sources agreeing within tolerance), so a trend is built only from
values that agree, never a single source's uncorroborated series. Deterministic, no LLM.
"""
from __future__ import annotations

from ..research.verification import SourcedValue, verify_figure


def verified_series(per_source: dict[str, dict[int, float]],
                    rel_tol: float = 0.02) -> dict[int, float]:
    """Cross-verify a per-year series across sources. For each fiscal year, keep the value only if
    >=2 distinct sources agree within rel_tol (median of the agreeing cluster); else drop the year.

    per_source: {source_id: {year: value}}.

    WHY delegate to verify_figure (regression, HIGH severity): this used to run its own,
    independent "star cluster around a pivot" clustering -- every value merely close to ONE
    shared pivot, not mutually close to every OTHER member. That can chain together a pair of
    sources that do NOT actually agree with each other (A close to B, B close to C, but A vs C
    beyond tolerance), wrongly reporting "all sources agree" for a year they genuinely disagree
    on. verify_figure's clique-based clustering already closes this; reusing it here (instead of
    maintaining a second, divergence-prone copy of the same safety-critical logic) means a future
    fix to the consensus rule can never apply to figures but silently miss year-by-year trends.
    """
    years: set[int] = set()
    for yearmap in per_source.values():
        years |= set(yearmap)
    out: dict[int, float] = {}
    for year in years:
        values = [SourcedValue(ym[year], source_id) for source_id, ym in per_source.items()
                  if ym.get(year) is not None]
        result = verify_figure(f"fy{year}", values, rel_tolerance=rel_tol)
        if result.is_trustworthy:
            out[year] = result.value
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


# Cumulative operating-cash-flow-to-profit bands, mirroring the single-year earnings_quality bands
# (framework._OCF_STRONG / _OCF_WEAK). WHY a MULTI-YEAR cumulative view in addition to the
# single-year one: one year's conversion is lumpy (working-capital timing), so a professional
# judges quality of earnings on whether profit converts to cash OVER TIME. A chronic cumulative
# gap is a far stronger signal of aggressive recognition / receivables build-up than any one year.
_CUM_OCF_STRONG = 0.80
_CUM_OCF_WEAK = 0.50


def cash_conversion_quality_point(ocf_series: dict[int, float],
                                  profit_series: dict[int, float]) -> str | None:
    """Multi-year quality-of-earnings check: cumulative operating cash flow vs cumulative net
    profit across the years where BOTH cross-verified. None if fewer than 3 common years or the
    cumulative profit isn't positive (the ratio isn't meaningful through a loss-making period)."""
    years = sorted(set(ocf_series) & set(profit_series))
    if len(years) < 3:
        return None
    cum_ocf = sum(ocf_series[y] for y in years)
    cum_profit = sum(profit_series[y] for y in years)
    if cum_profit <= 0:
        return None
    ratio = cum_ocf / cum_profit
    n = len(years)
    if ratio < 0:
        return (f"Over the last {n} years, cumulative operating cash flow was actually NEGATIVE "
                "despite cumulative reported profit -- a serious quality-of-earnings red flag: the "
                "business consumed cash from operations across the period while reporting profits. "
                "Check receivables, working capital, and revenue recognition (cross-verified).")
    if ratio < _CUM_OCF_WEAK:
        return (f"Over the last {n} years, cumulative operating cash flow was only {ratio:.0%} of "
                "cumulative reported profit -- a persistent gap that can signal aggressive revenue "
                "recognition or a receivables/working-capital build-up; check why profit isn't "
                "converting to cash (cross-verified).")
    if ratio >= _CUM_OCF_STRONG:
        return (f"Over the last {n} years, cumulative operating cash flow was {ratio:.0%} of "
                "cumulative reported profit -- reported profits have been well backed by real cash "
                "(cross-verified).")
    return (f"Over the last {n} years, cumulative operating cash flow was {ratio:.0%} of "
            "cumulative reported profit -- profits are only partly backed by cash; worth watching "
            "(cross-verified).")


# Minimum ABSOLUTE change in debt/equity to call a multi-year direction. WHY absolute, not
# relative: a move from 0.04 to 0.10 is a big RELATIVE jump but the leverage is negligible either
# way (live-verified: DMART 0.04->0.10), so a relative test would cry wolf on rock-solid balance
# sheets; a >=0.15 absolute shift in D/E is a real change in capital structure across the range
# (live-verified: Reliance held ~0.44 flat -> steady; Suzlon 1.76 -> 0.06 -> clear deleveraging).
_LEVERAGE_TREND_BAND = 0.15


def leverage_trend_point(debt_series: dict[int, float],
                         equity_series: dict[int, float]) -> str | None:
    """Plain-language multi-year leverage (debt/equity) trend, oldest vs latest, from
    CROSS-VERIFIED debt and equity series. WHY (real money, CA-level rigor): a single-year D/E
    snapshot hides whether the balance sheet is getting riskier -- debt rising faster than equity
    (D/E climbing) is a core leverage-risk signal, and steady deleveraging is a genuine positive.
    Built only from years where BOTH debt and equity cross-verified and equity is positive (D/E
    is meaningless through zero/negative net worth); None if fewer than 2 such years exist. The
    caller skips this for banks/NBFCs, whose leverage is their business model, not a risk signal."""
    de = {y: debt_series[y] / equity_series[y]
          for y in set(debt_series) & set(equity_series) if equity_series[y] > 0}
    if len(de) < 2:
        return None
    years = sorted(de)
    first, last = de[years[0]], de[years[-1]]
    span = f"{first:.2f} in FY{years[0]} to {last:.2f} in FY{years[-1]}"
    if abs(last - first) < _LEVERAGE_TREND_BAND:
        return f"Leverage (debt/equity) has stayed roughly steady, {span} (cross-verified)."
    if last > first:
        return (f"Leverage (debt/equity) has risen, {span} -- the balance sheet has taken on "
                "relatively more debt over these years; check whether it funded productive growth "
                "or covered cash shortfalls (cross-verified).")
    return (f"Leverage (debt/equity) has fallen, {span} -- the company has been deleveraging, "
            "usually a positive for balance-sheet resilience (cross-verified).")


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
    # Prefer the profit signal (the bottom line, more decision-relevant) whenever there is ENOUGH
    # profit data to judge it at all -- whether it turns out volatile or confirmed smooth. Fall
    # back to revenue only when profit data ITSELF is too thin to compute a swing (see
    # revenue_volatility_point's WHY). WHY this distinction matters (regression, adversarial
    # review): `earnings_volatility_point(...) or revenue_volatility_point(...)` would ALSO fall
    # back whenever profit merely came back smooth (swing under threshold), not just when it was
    # thin -- both cases return None from earnings_volatility_point, so the `or` couldn't tell
    # them apart. That showed "steady profit growth" right next to "but revenue swung sharply,
    # don't trust a single year" for the SAME business: a confirmed-smooth bottom line despite one
    # lumpy revenue year is a reasonable case to show NO caveat at all (the business absorbed the
    # swing before it reached earnings), not a case to caveat on revenue's behalf.
    if _yoy_swing(profit_series) is not None:
        volatility = earnings_volatility_point(profit_series)
    else:
        volatility = revenue_volatility_point(revenue_series)
    if volatility:
        points.append(volatility)
    return points
