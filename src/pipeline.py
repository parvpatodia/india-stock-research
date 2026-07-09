"""Company report pipeline: the end-to-end chain.

figures (per-source values) -> cross-verify each -> run the analysis framework on ONLY the
cross-verified values -> assemble a caveated verdict -> return a DRAFT report. The report is
DRAFT and not trusted until the human expert approves it (see report.py). This module has no
LLM and no network; sources are passed in, so it is fully testable.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .analysis.framework import (
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
                         trend_insights: tuple[str, ...] = (),
                         trend_improving: bool = False) -> Report:
    # WHY: dividend_yield_pct gets a wider cross-verification tolerance than the 2% default.
    # Live-verified across 6 real stocks: yfinance vs Screener typically differ 2-17% (different
    # trailing-dividend windows/methodology), not the parsing/scale errors the tight default
    # guards against elsewhere; a stock with a large recent special dividend (confirmed on TCS,
    # a 48% gap) still correctly stays a CONFLICT even at this wider band.
    _TOLERANCE = {"dividend_yield_pct": 0.25}
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
        quality_signals = [
            earnings_quality(tv("operating_cash_flow"), tv("net_profit")),
            leverage_health(tv("total_debt"), tv("equity"), tv("ebit"), tv("interest_expense")),
            promoter_pledge(tv("promoter_pledge_pct")),
        ]
        verdict = assemble_verdict(valuation, quality_signals)

    # Plain-language "why" points from the ratio suite + core figures (cross-verified only).
    from .analysis.deep_metrics import compute_deep_metrics, plain_points
    tvals = {name: tv(name) for name in (
        "current_pe", "operating_cash_flow", "net_profit", "total_debt",
        "equity", "ebit", "interest_expense", "total_assets", "revenue",
        "dividend_yield_pct")}
    # WHY: use the computed median actually used for the valuation tier, not the (unfetched)
    # median_pe figure, so the price/valuation reason renders whenever valuation was assessed.
    tvals["median_pe"] = median
    insights = plain_points(tvals, compute_deep_metrics(tvals, is_bank=is_bank))

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
    from .analysis.bank_framework import is_bank, is_nbfc
    from .analysis.trends import trend_improving, trend_points, verified_series
    from .analysis.valuation import compute_median_pe
    series = gather_series(symbol, sources)
    rev_series = verified_series(series.get("revenue", {}))
    prof_series = verified_series(series.get("net_profit", {}))
    # WHY (sector-aware analysis): an NBFC borrows to lend, just like a bank, so its leverage is
    # the business model, not a risk signal, penalizing it under the industrial D/E lens is a
    # real analytical error. Route it through the same ROA-based framework as a bank.
    return build_company_report(symbol, gather_aligned_figures(symbol, sources),
                                claims=claims, median_pe=compute_median_pe(symbol),
                                is_bank=is_bank(symbol) or is_nbfc(symbol),
                                trend_insights=tuple(trend_points(rev_series, prof_series)),
                                trend_improving=trend_improving(rev_series, prof_series))
