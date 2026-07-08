"""Company report pipeline: the end-to-end chain.

figures (per-source values) -> cross-verify each -> run the analysis framework on ONLY the
cross-verified values -> assemble a caveated verdict -> return a DRAFT report. The report is
DRAFT and not trusted until the human expert approves it (see report.py). This module has no
LLM and no network; sources are passed in, so it is fully testable.
"""
from __future__ import annotations

from .analysis.framework import (
    assemble_verdict,
    earnings_quality,
    leverage_health,
    promoter_pledge,
    valuation_vs_history,
    value_if_trustworthy,
)
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
