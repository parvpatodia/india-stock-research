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
    profit across the years where BOTH cross-verified. None unless there are >=3 such years and
    EVERY one is profitable.

    WHY require every year profitable, not just a positive cumulative (real money, HIGH severity;
    found by adversarial review): this ratio is a quality-of-earnings read premised on a
    consistently profitable period. A single loss year netting against profit years can shrink
    cumulative profit to a near-zero residual, blowing the ratio up into a nonsensical, falsely
    reassuring verdict (e.g. profit -100/+50/+55 cr -> cumulative +5 cr, OCF 135 cr -> "2700% --
    well backed by real cash") for a company whose earnings quality is actually murky -- a real
    risk for cyclicals (steel, commodities). Such a period is better served by the single-year
    earnings-quality point and the earnings-volatility caveat, so this abstains."""
    years = sorted(set(ocf_series) & set(profit_series))
    if len(years) < 3:
        return None
    if any(profit_series[y] <= 0 for y in years):
        return None
    cum_ocf = sum(ocf_series[y] for y in years)
    cum_profit = sum(profit_series[y] for y in years)   # > 0: every year is profitable
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
    # WHY (found by adversarial review): an oldest-vs-latest read alone hides a leverage episode
    # that resolved -- 0.20 -> 1.50 -> 0.25 reads "steady" on the endpoints, erasing a real spike.
    # Surface a materially higher intra-period peak (above both endpoints, at a middle year) so a
    # temporary leverage-up a CA would want to know about is not silently averaged away.
    peak_year = max(years, key=lambda y: de[y])
    spike = ""
    if peak_year not in (years[0], years[-1]) and de[peak_year] > max(first, last) + _LEVERAGE_TREND_BAND:
        spike = f", though it spiked to {de[peak_year]:.2f} in FY{peak_year} in between"
    if abs(last - first) < _LEVERAGE_TREND_BAND:
        return f"Leverage (debt/equity) has stayed roughly steady, {span}{spike} (cross-verified)."
    if last > first:
        return (f"Leverage (debt/equity) has risen, {span}{spike} -- the balance sheet has taken "
                "on relatively more debt over these years; check whether it funded productive "
                "growth or covered cash shortfalls (cross-verified).")
    return (f"Leverage (debt/equity) has fallen, {span}{spike} -- the company has been "
            "deleveraging, usually a positive for balance-sheet resilience (cross-verified).")


def margins_improving(revenue_series: dict[int, float],
                      profit_series: dict[int, float]) -> bool | None:
    """End-to-end net-margin direction from revenue vs profit growth: True (margins expanded),
    False (under pressure), or None (roughly flat, or not comparable).

    WHY compare over the COMMON window, not each series' own endpoints (real money, CA-level
    correctness): the margin identity margin_last / margin_first = (profit_last/profit_first) /
    (revenue_last/revenue_first) only holds when both ratios span the SAME first and last year.
    Cross-verified revenue and profit routinely cover DIFFERENT year windows (a source leaves one
    figure's newest or oldest cell empty), so comparing revenue's FY19-FY24 CAGR against profit's
    FY22-FY24 CAGR pits two different eras against each other -- it could read a recent profit
    recovery as a full-period margin expansion, both on the page and in the suggestion score.
    Restricting BOTH to their shared years makes the two growth rates cover one identical period
    (equal span => the annualized comparison is monotonic with the true end-to-end margin change),
    while interior gaps are harmless (only the shared endpoints drive an endpoint CAGR). Needs a
    >=3-year overlap with positive endpoints (see cagr); otherwise there is no honest comparison
    and this abstains rather than guess."""
    common = set(revenue_series) & set(profit_series)
    rev = cagr({y: revenue_series[y] for y in common})
    prof = cagr({y: profit_series[y] for y in common})
    if rev is None or prof is None:
        return None
    if prof[0] > rev[0] + _MARGIN_MIN:
        return True
    if prof[0] < rev[0] - _MARGIN_MIN:
        return False
    return None


def trend_improving(revenue_series: dict[int, float],
                    profit_series: dict[int, float]) -> bool:
    """Structured multi-year signal for the ranker: True iff the BOTTOM LINE is compounding above
    the growth floor, margins have been improving, or sales are growing WITHOUT profit shrinking.
    WHY (real money): the suggestion score must read this from the numbers, not by substring-
    matching the plain-language insight prose, so a wording change can never silently flip a scoring
    input. Shares the thresholds AND the common-window margin comparison with trend_points so the
    flag and the words agree. Needs >=3 cross-verified years (see cagr) or returns False — no
    history, no claimed trend.

    WHY revenue growth alone no longer rescues a falling bottom line (value trap): sales compounding
    while PROFIT actually shrinks is the textbook unprofitable-growth trap (top line up, earnings
    down, margins collapsing). Crediting it as "improving" would flatly contradict the "profit
    shrinking / margins under pressure" lines trend_points prints for the very same numbers, and a
    value investor treats it as deterioration, not progress. Profit growth or a genuine margin
    expansion still stands on its own; a merely-flat bottom line still lets sales growth count (a
    judgment call trend_points leaves open, not a contradiction)."""
    rev = cagr(revenue_series)
    prof = cagr(profit_series)
    profit_growing = prof is not None and prof[0] > _GROWTH_MIN
    profit_shrinking = prof is not None and prof[0] < -_GROWTH_MIN
    sales_growing = rev is not None and rev[0] > _GROWTH_MIN
    if profit_growing or margins_improving(revenue_series, profit_series) is True:
        return True
    return bool(sales_growing and not profit_shrinking)


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
    # WHY (real money, honesty): a year-over-year growth % is ill-defined through a SIGN CHANGE --
    # a near-zero base (a small loss between profit years) explodes it into an absurd, alarming
    # figure a parent would read as a typo (live-repro: "a 20201-percentage-point range"). When
    # profit crossed between losses and profits in the window, fire the SAME earning-power caveat
    # but phrase it qualitatively rather than quoting a nonsensical percentage.
    values = list(profit_series.values())
    if any(v < 0 for v in values) and any(v > 0 for v in values):
        return (f"Profit has swung between losses and profits over the last {n_years} years -- a "
                "single year's ROE/margin may not represent its long-term earning power, so weigh "
                "the multi-year trend, not just the latest year.")
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


def limited_history_note(years: int) -> str | None:
    """A caveat when only a SHORT cross-verified financial history is available (fewer than the 3
    years a trend needs), so a single strong or weak year is not over-read as a proven track record.

    WHY (real money, rigor): a recently-listed company shows only latest-year ratios, and the
    verdict's confidence reflects how many METRICS cross-verified, not how many YEARS -- so it can
    read HIGH-confidence FAVORABLE on a single year. A value investor is wary of a short track record
    (an unproven IPO can post one flattering year right after listing); state the limitation plainly.
    Silent at >=3 years (a real multi-year trend is available) and at 0 (nothing cross-verified --
    that is handled as no-data, not a short-history caveat)."""
    if years < 1 or years >= 3:
        return None
    label = "year" if years == 1 else "years"
    return (f"Heads up: only {years} {label} of cross-verified financial history is available here, "
            "so this rests on a short track record, not a proven multi-year one -- weigh a single "
            "strong or weak year with that in mind.")


def trend_points(revenue_series: dict[int, float],
                 profit_series: dict[int, float]) -> list[str]:
    """Plain-language multi-year track-record points, from cross-verified yearly series."""
    points: list[str] = []
    rev = cagr(revenue_series)
    prof = cagr(profit_series)
    # WHY abs(rate) (real money, clarity): the direction is already carried by _word
    # (growing/shrinking/roughly flat), so a negative CAGR must show its MAGNITUDE -- "shrinking
    # about 13% a year", never the double-negative "shrinking about -13% a year".
    if rev:
        rate, span = rev
        points.append(f"Track record: sales have been {_word(rate)} about {abs(rate):.0f}% a year "
                      f"over the last {span} years (cross-verified).")
    if prof:
        rate, span = prof
        points.append(f"Track record: profit has been {_word(rate)} about {abs(rate):.0f}% a year "
                      f"over the last {span} years.")
    # Margin direction compares the two growth rates over their COMMON window (see
    # margins_improving) so a revenue/profit year-coverage mismatch can't read two different eras
    # against each other. The per-series CAGR lines above keep each figure's own full window.
    # WHY "outpaced/lagged" not "grown faster/slower" (clarity/correctness): when BOTH sales and
    # profit are FALLING, "profit grew slower than sales" is wrong -- profit shrank faster. Outpaced/
    # lagged is true whether the business grew or shrank (it's about the ratio, i.e. the margin).
    margin = margins_improving(revenue_series, profit_series)
    if margin is True:
        points.append("Profit has outpaced sales, so margins have been improving.")
    elif margin is False:
        points.append("Profit has lagged sales, so margins have been under pressure.")
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
