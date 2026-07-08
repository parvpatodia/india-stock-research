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
                         median_pe: float | None = None) -> Report:
    verified = {name: verify_figure(name, values) for name, values in figures.items()}

    def tv(name: str):
        return value_if_trustworthy(verified.get(name))

    # median_pe is a computed baseline for the (opinion) valuation tier; fall back to a
    # cross-verified median_pe figure if one was supplied, else None -> valuation unknown.
    median = median_pe if median_pe is not None else tv("median_pe")
    valuation = valuation_vs_history(tv("current_pe"), median)
    quality_signals = [
        earnings_quality(tv("operating_cash_flow"), tv("net_profit")),
        leverage_health(tv("total_debt"), tv("equity"), tv("ebit"), tv("interest_expense")),
        promoter_pledge(tv("promoter_pledge_pct")),
    ]
    verdict = assemble_verdict(valuation, quality_signals)

    return Report(
        company=company,
        claims=tuple(claims),
        figures=tuple(verified.values()),
        verdict=verdict,
        status=ReviewStatus.DRAFT,
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


def build_report_for_symbol(symbol: str, sources: list[FigureSource],
                            claims: tuple[Claim, ...] = ()) -> Report:
    """Real-data entry point: gather fiscal-year-aligned figures across sources, compute the
    historical median P/E for the valuation baseline, then run the pipeline. A single source
    stays single-source (low confidence); agreeing sources verify."""
    from .analysis.valuation import compute_median_pe
    return build_company_report(symbol, gather_aligned_figures(symbol, sources),
                                claims=claims, median_pe=compute_median_pe(symbol))
