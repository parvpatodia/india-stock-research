"""Company report pipeline: the end-to-end chain.

figures (per-source values) -> cross-verify each -> run the analysis framework on ONLY the
cross-verified values -> assemble a caveated verdict -> return a DRAFT report. The report is
DRAFT and not trusted until the human expert approves it (see report.py). This module has no
LLM and no network; sources are passed in, so it is fully testable.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .analysis.framework import (
    REAL_ESTATE_LEVERAGE_CAVEAT,
    assemble_verdict,
    earnings_quality,
    leverage_health,
    promoter_pledge,
    valuation_vs_history,
    value_if_trustworthy,
)
from .data.figure_sources import POINT_FIGURES, YEAR_FIGURES, FigureSource
from .research.claims import Claim
from .research.report import Report, ReviewStatus
from .research.verification import SourcedValue, verify_figure


def build_company_report(company: str,
                         figures: dict[str, list[SourcedValue]],
                         claims: tuple[Claim, ...] = (),
                         median_pe: float | None = None,
                         is_bank: bool = False,
                         is_real_estate: bool = False,
                         trend_insights: tuple[str, ...] = (),
                         trend_improving: bool = False,
                         prior_year_figures: dict[str, float] | None = None) -> Report:
    # WHY: some cross-source figures legitimately differ by more than the 2% default because the
    # two sources compute them from different windows/bases, not because either is a parse/scale
    # error -- those figures get a wider band, while a genuinely gross disagreement still
    # CONFLICTs (see the per-figure tests).
    #  - dividend_yield_pct (0.25): yfinance vs Screener differ 2-17% (different trailing-dividend
    #    windows/methodology); a large special-dividend gap (TCS, 48%) still correctly conflicts.
    #  - current_pe (0.15): live-verified that yfinance trailingPE vs Screener's Stock P/E differ
    #    ~3-9% for major stocks (RELIANCE 3.4%, TCS 6.8%, HDFCBANK 9.3%) from different trailing-EPS
    #    windows / consolidated-vs-standalone basis / price snapshot -- NOT the "clean ratio" the
    #    old 2% band assumed. At 2% the current P/E was CONFLICTing for these names, withholding the
    #    ENTIRE valuation tier (the core margin-of-safety signal); a scale/parse error (~10x) still
    #    conflicts well within 0.15.
    _TOLERANCE = {"dividend_yield_pct": 0.25, "current_pe": 0.15}
    verified = {name: verify_figure(name, values, rel_tolerance=_TOLERANCE.get(name, 0.02))
               for name, values in figures.items()}

    def tv(name: str):
        return value_if_trustworthy(verified.get(name))

    # median_pe is a computed baseline for the (opinion) valuation tier; fall back to a
    # cross-verified median_pe figure if one was supplied, else None -> valuation unknown.
    median = median_pe if median_pe is not None else tv("median_pe")
    valuation = valuation_vs_history(tv("current_pe"), median)
    if is_bank:
        # WHY: the industrial lenses (D/E, coverage, EBIT) do not apply to banks; use ROA.
        from .analysis.bank_framework import assemble_bank_verdict, return_on_assets
        roa = return_on_assets(tv("net_profit"), tv("total_assets"))
        verdict = assemble_bank_verdict(valuation, roa)
    else:
        leverage = leverage_health(tv("total_debt"), tv("equity"), tv("ebit"),
                                   tv("interest_expense"))
        quality_signals = [
            earnings_quality(tv("operating_cash_flow"), tv("net_profit")),
            leverage,
            promoter_pledge(tv("promoter_pledge_pct")),
        ]
        # WHY: only attach the sector caveat when leverage actually reads "stretched" -- a
        # real-estate name with low/comfortable debt (e.g. DLF, live D/E 0.01) has nothing to
        # caveat, so attaching it unconditionally to every real-estate report would be clutter
        # dressed up as diligence, not honesty.
        extra_caveats = ((REAL_ESTATE_LEVERAGE_CAVEAT,)
                        if is_real_estate and leverage.verdict == "stretched" else ())
        verdict = assemble_verdict(valuation, quality_signals, sector_caveats=extra_caveats)

    # Plain-language "why" points from the ratio suite + core figures (cross-verified only).
    from .analysis.deep_metrics import compute_deep_metrics, plain_points
    tvals = {name: tv(name) for name in (
        "current_pe", "operating_cash_flow", "net_profit", "total_debt",
        "equity", "ebit", "interest_expense", "total_assets", "revenue",
        "dividend_yield_pct")}
    # WHY: use the computed median actually used for the valuation tier, not the (unfetched)
    # median_pe figure, so the price/valuation reason renders whenever valuation was assessed.
    tvals["median_pe"] = median
    # Opening (prior-year) balances for the CA-standard average-denominator return ratios (ROE,
    # ROCE, ROA). Only cross-verified prior-year values are passed; compute_deep_metrics falls
    # back to the closing value for any that are absent, so nothing is fabricated.
    for key in ("equity", "total_debt", "total_assets"):
        if prior_year_figures and prior_year_figures.get(key) is not None:
            tvals[f"prior_{key}"] = prior_year_figures[key]
    insights = plain_points(tvals, compute_deep_metrics(tvals, is_bank=is_bank),
                            is_real_estate=is_real_estate, is_bank=is_bank)

    return Report(
        company=company,
        claims=tuple(claims),
        figures=tuple(verified.values()),
        verdict=verdict,
        status=ReviewStatus.DRAFT,
        insights=tuple(insights) + tuple(trend_insights),
        trend_improving=trend_improving,
    )


def gather_figures(symbol: str,
                   sources: list[FigureSource]) -> dict[str, list[SourcedValue]]:
    """Collect each figure's value from every source, tagged by source id. A figure with
    values from >=2 sources can cross-verify; one source stays single-source (not trusted)."""
    merged: dict[str, list[SourcedValue]] = defaultdict(list)
    for source in sources:
        for name, value in source.figures(symbol).items():
            merged[name].append(SourcedValue(value=value, source_id=source.source_id))
    return dict(merged)


def _latest_common_year(per_source: dict[str, dict[int, float]]) -> int | None:
    """The most recent fiscal year present in >=2 sources; else the latest year available."""
    if not per_source:
        return None
    counts = Counter(year for yearmap in per_source.values() for year in yearmap)
    common = [year for year, c in counts.items() if c >= 2]
    if common:
        return max(common)
    all_years = [year for yearmap in per_source.values() for year in yearmap]
    return max(all_years) if all_years else None


def gather_aligned_figures(symbol: str,
                           sources: list[FigureSource]) -> dict[str, list[SourcedValue]]:
    """Cross-source figures with fiscal-year alignment.

    For each statement figure, compare the SAME latest common fiscal year across sources (so a
    source that reports one year ahead of another does not spuriously conflict). Point figures
    (current P/E, pledge) are compared as-is. Scalar-only sources (e.g. annual-report extraction)
    fill in a figure only when fewer than two year-aligned sources cover it.
    """
    scalar_by_source = {src.source_id: src.figures(symbol) for src in sources}
    series_by_figure: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for src in sources:
        for figure, yearmap in src.figures_by_year(symbol).items():
            if yearmap:
                series_by_figure[figure][src.source_id] = yearmap

    merged: dict[str, list[SourcedValue]] = defaultdict(list)

    for figure in YEAR_FIGURES:
        per_source = series_by_figure.get(figure, {})
        year = _latest_common_year(per_source)
        used = set()
        if year is not None:
            for sid, yearmap in per_source.items():
                if yearmap.get(year) is not None:
                    merged[figure].append(SourcedValue(yearmap[year], sid, locator=f"FY{year}"))
                    used.add(sid)
        for sid, scalar in scalar_by_source.items():  # fallback for scalar-only sources
            if len(merged[figure]) >= 2:
                break
            if sid not in used and scalar.get(figure) is not None:
                merged[figure].append(SourcedValue(scalar[figure], sid, locator="reported"))

    for figure in POINT_FIGURES:
        for sid, scalar in scalar_by_source.items():
            if scalar.get(figure) is not None:
                merged[figure].append(SourcedValue(scalar[figure], sid))

    return dict(merged)


def gather_series(symbol: str,
                  sources: list[FigureSource]) -> dict[str, dict[str, dict[int, float]]]:
    """Per-figure, per-source multi-year series: {figure: {source_id: {year: value}}}. Used to
    cross-verify each year for trend analysis."""
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for src in sources:
        for figure, yearmap in src.figures_by_year(symbol).items():
            if yearmap:
                out[figure][src.source_id] = yearmap
    return dict(out)


def build_report_for_symbol(symbol: str, sources: list[FigureSource],
                            claims: tuple[Claim, ...] = ()) -> Report:
    """Real-data entry point: gather fiscal-year-aligned figures across sources, compute the
    historical median P/E for the valuation baseline, add cross-verified multi-year trends, then
    run the pipeline. A single source stays single-source (low confidence); agreeing sources verify."""
    from .analysis.bank_framework import sector_category
    from .analysis.trends import (
        cash_conversion_quality_point,
        leverage_trend_point,
        trend_improving,
        trend_points,
        verified_series,
    )
    from .analysis.valuation import compute_median_pe
    series = gather_series(symbol, sources)
    rev_series = verified_series(series.get("revenue", {}))
    prof_series = verified_series(series.get("net_profit", {}))
    # WHY (sector-aware analysis): an NBFC borrows to lend, just like a bank, so its leverage is
    # the business model, not a risk signal, penalizing it under the industrial D/E lens is a
    # real analytical error. Route it through the same ROA-based framework as a bank. A real-estate
    # developer stays on the industrial D/E lens (it is not a borrow-to-lend business) but gets an
    # added leverage caveat -- see framework.REAL_ESTATE_LEVERAGE_CAVEAT for why. sector_category
    # fetches yfinance's industry string ONCE and classifies it, rather than the three separate
    # fetches an earlier version made (one each for is_bank/is_nbfc/is_real_estate).
    category = sector_category(symbol)
    trend_insights = list(trend_points(rev_series, prof_series))
    # WHY (CA-level rigor): add the multi-year debt/equity trend for industrials/real-estate --
    # is the balance sheet getting more or less leveraged over time? Skipped for banks/NBFCs,
    # whose leverage is their business model, not a risk signal (same reason they use the ROA
    # framework, not the D/E lens). Built from cross-verified debt & equity, so it sits with the
    # other cross-verified insights, not the single-source Screener context signals.
    if category not in ("bank", "nbfc"):
        lev = leverage_trend_point(verified_series(series.get("total_debt", {})),
                                   verified_series(series.get("equity", {})))
        if lev:
            trend_insights.append(lev)
        # WHY (CA-level quality of earnings): the MULTI-YEAR cumulative view of whether reported
        # profit actually converts to cash -- a chronic gap is a far stronger red flag than any
        # one lumpy year. Skipped for banks/NBFCs, whose operating cash flow is dominated by
        # lending/deposit flows, not the industrial profit-to-cash relationship this measures.
        cash = cash_conversion_quality_point(verified_series(series.get("operating_cash_flow", {})),
                                             verified_series(series.get("net_profit", {})))
        if cash:
            trend_insights.append(cash)

    # Opening (prior-year) balances for the CA-standard average-denominator return ratios (ROE,
    # ROCE, ROA). Only the exact year before the latest cross-verified year counts, so we never
    # average across a gap; absent -> the ratio falls back to the closing value. Computed for all
    # sectors (banks use average total assets for ROA too).
    def _series(figure: str) -> dict[int, float]:
        return verified_series(series.get(figure, {}))

    def _opening(vseries: dict[int, float]) -> float | None:
        return vseries.get(max(vseries) - 1) if len(vseries) >= 2 else None

    eq_series, debt_series = _series("equity"), _series("total_debt")
    prior_year_figures: dict[str, float | None] = {
        "equity": _opening(eq_series),
        "total_assets": _opening(_series("total_assets")),
    }
    # WHY (found by adversarial review): ROCE averages a TWO-item sum (equity + debt), and each
    # leg's opening is anchored to its own series' latest year. yfinance/Screener often cover
    # equity and Total Debt through DIFFERENT latest years (yfinance frequently leaves the newest
    # Total Debt cell empty), which would blend equity's (Y, Y-1) window with debt's (Y-1, Y-2)
    # window yet label it a clean "average capital". Only pair the debt opening when both series
    # share the same latest cross-verified year (which also makes the closings a clean Y+Y per
    # gather_aligned_figures); otherwise ROCE keeps point capital, no misleading "average" label.
    if eq_series and debt_series and max(eq_series) == max(debt_series):
        prior_year_figures["total_debt"] = _opening(debt_series)
    return build_company_report(symbol, gather_aligned_figures(symbol, sources),
                                claims=claims, median_pe=compute_median_pe(symbol),
                                is_bank=category in ("bank", "nbfc"),
                                is_real_estate=category == "real_estate",
                                trend_insights=tuple(trend_insights),
                                trend_improving=trend_improving(rev_series, prof_series),
                                prior_year_figures=prior_year_figures)
