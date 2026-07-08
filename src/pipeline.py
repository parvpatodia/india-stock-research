"""Company report pipeline: the end-to-end chain.

figures (per-source values) -> cross-verify each -> run the analysis framework on ONLY the
cross-verified values -> assemble a caveated verdict -> return a DRAFT report. The report is
DRAFT and not trusted until the human expert approves it (see report.py). This module has no
LLM and no network; sources are passed in, so it is fully testable.
"""
from __future__ import annotations

from collections import defaultdict

from .analysis.framework import (
    assemble_verdict,
    earnings_quality,
    leverage_health,
    promoter_pledge,
    valuation_vs_history,
    value_if_trustworthy,
)
from .data.figure_sources import FigureSource
from .research.claims import Claim
from .research.report import Report, ReviewStatus
from .research.verification import SourcedValue, verify_figure


def build_company_report(company: str,
                         figures: dict[str, list[SourcedValue]],
                         claims: tuple[Claim, ...] = ()) -> Report:
    verified = {name: verify_figure(name, values) for name, values in figures.items()}

    def tv(name: str):
        return value_if_trustworthy(verified.get(name))

    valuation = valuation_vs_history(tv("current_pe"), tv("median_pe"))
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


def build_report_for_symbol(symbol: str, sources: list[FigureSource],
                            claims: tuple[Claim, ...] = ()) -> Report:
    """Real-data entry point: pull figures from the given sources, then run the same pipeline.
    With a single source, figures are single-source and the verdict stays low-confidence."""
    return build_company_report(symbol, gather_figures(symbol, sources), claims=claims)
