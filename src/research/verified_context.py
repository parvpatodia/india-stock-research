"""Feed the Ask tab's grounding with numbers that already passed cross-verification.

The Ask tab previously only had curated documents + live news to ground answers, so it had
no access to the same rigorously cross-verified figures the Research tab computes and shows.
This turns an already-built Report into a citable context document, but ONLY from figures that
are is_trustworthy (VERIFIED, i.e. >=2 independent sources agreed) plus the plain-language
insights (themselves derived only from cross-verified figures, see pipeline.py). A single-source
or conflicting figure is never included, so the Ask tab can never ground an answer in a number
that has not already cleared the same bar as everything else in this app.
"""
from __future__ import annotations

from ..data.figure_sources import PERCENT_FIGURES, RATIO_FIGURES
from ..sources.adapters import FetchedDocument
from .report import Report

VERIFIED_FIGURES_SOURCE_ID = "verified_figures"

# Human-readable labels for the framework's snake_case figure keys (src/data/figure_sources.py).
_LABELS = {
    "current_pe": "Current P/E",
    "median_pe": "Historical median P/E",
    "operating_cash_flow": "Operating cash flow",
    "net_profit": "Net profit",
    "total_debt": "Total debt",
    "equity": "Shareholders' equity",
    "ebit": "EBIT (operating profit before interest & tax)",
    "interest_expense": "Interest expense",
    "promoter_pledge_pct": "Promoter pledge",
    "total_assets": "Total assets",
    "revenue": "Revenue",
    "dividend_yield_pct": "Dividend yield",
}


def _format(name: str, value: float) -> str:
    if name in RATIO_FIGURES:
        return f"{value:.1f}x"
    if name in PERCENT_FIGURES:
        return f"{value:.1f}%"
    return f"₹{value:,.0f}"


def verified_figures_document(symbol: str, report: Report | None) -> FetchedDocument | None:
    """A citable document of ONLY cross-verified figures + derived insights for `symbol`, or
    None if there is no report or nothing on it cleared cross-verification. Registered at
    PRIMARY tier by the caller: every number here already passed the same >=2-independent-source
    bar the rest of the app requires before a figure counts as fact, so it is safe to cite."""
    if report is None:
        return None
    lines: list[str] = list(report.insights)
    for fig in report.figures:
        if fig.is_trustworthy:
            label = _LABELS.get(fig.name, fig.name)
            lines.append(f"{label}: {_format(fig.name, fig.value)} (cross-verified: {fig.note}).")
    if not lines:
        return None
    text = (f"Cross-verified research on {symbol} (each figure independently agreed by "
            f">=2 public sources):\n" + "\n".join(lines))
    return FetchedDocument(VERIFIED_FIGURES_SOURCE_ID, text, url="",
                           locator=f"{symbol} verified figures")
