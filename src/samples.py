"""Synthetic sample companies for building and demoing the report pipeline (NOT real figures).

Each figure has two independent sources so it cross-verifies. "Risky Corp" carries a deliberate
revenue conflict (sources disagree) to demonstrate the conflict gate blocking approval.
"""
from __future__ import annotations

from .research.verification import SourcedValue


def _sv(value: float, source: str) -> SourcedValue:
    return SourcedValue(value=value, source_id=source)


SAMPLE_COMPANIES: dict[str, dict[str, list[SourcedValue]]] = {
    "Acme Industries (SAMPLE)": {
        "current_pe": [_sv(18.0, "screener"), _sv(18.1, "tickertape")],
        "median_pe": [_sv(24.0, "screener"), _sv(24.0, "annual_report")],       # cheap vs history
        "operating_cash_flow": [_sv(88000, "annual_report"), _sv(88200, "screener")],
        "net_profit": [_sv(79000, "annual_report"), _sv(79010, "screener")],     # OCF/NP ~1.1 strong
        "total_debt": [_sv(30000, "annual_report"), _sv(30050, "screener")],
        "equity": [_sv(200000, "annual_report"), _sv(200000, "screener")],       # D/E 0.15 healthy
        "ebit": [_sv(110000, "annual_report"), _sv(110000, "screener")],
        "interest_expense": [_sv(5000, "annual_report"), _sv(5000, "screener")], # cover 22x
        "promoter_pledge_pct": [_sv(0.0, "nsdl"), _sv(0.0, "bse")],
    },
    "Risky Corp (SAMPLE)": {
        "current_pe": [_sv(45.0, "screener"), _sv(45.0, "tickertape")],
        "median_pe": [_sv(20.0, "screener"), _sv(20.0, "annual_report")],        # expensive
        "operating_cash_flow": [_sv(20000, "annual_report"), _sv(20000, "screener")],
        "net_profit": [_sv(60000, "annual_report"), _sv(60000, "screener")],     # OCF/NP 0.33 weak
        "total_debt": [_sv(300000, "annual_report"), _sv(300000, "screener")],
        "equity": [_sv(150000, "annual_report"), _sv(150000, "screener")],       # D/E 2.0 stretched
        "ebit": [_sv(40000, "annual_report"), _sv(40000, "screener")],
        "interest_expense": [_sv(20000, "annual_report"), _sv(20000, "screener")],  # cover 2x
        "promoter_pledge_pct": [_sv(60.0, "nsdl"), _sv(60.0, "bse")],            # high pledge
        "revenue": [_sv(500000, "annual_report"), _sv(650000, "moneycontrol")],  # CONFLICT on purpose
    },
}
