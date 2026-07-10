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

from ..data.figure_sources import format_figure_value
from ..sources.adapters import FetchedDocument
from .report import Report

VERIFIED_FIGURES_SOURCE_ID = "verified_figures"
PROMOTER_TREND_SOURCE_ID = "promoter_trend"

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
            lines.append(f"{label}: {format_figure_value(fig.name, fig.value)} "
                        f"(cross-verified: {fig.note}).")
    if not lines:
        return None
    # WHY (real money, honesty): the Ask tab stamps every citation's as_of with the CURRENT time
    # (when the question is asked), not when these figures were actually fetched -- a stock
    # researched hours earlier in the same session would otherwise look freshly-verified when
    # asked about later. Self-disclose the real fetch time in the text itself, so the freshness
    # signal travels with the content regardless of the citation metadata (same pattern already
    # used for news and the annual-report reader).
    text = (f"Cross-verified research on {symbol}, fetched {report.created_at} (each figure "
            f"independently agreed by >=2 public sources):\n" + "\n".join(lines))
    return FetchedDocument(VERIFIED_FIGURES_SOURCE_ID, text, url="",
                           locator=f"{symbol} verified figures")


def promoter_trend_document(symbol: str, trend: str | None) -> FetchedDocument | None:
    """Wrap the Research tab's promoter-shareholding-trend sentence (see
    screener_source.promoter_holding_trend, Screener-only, single-source) as a citable document
    for the Ask tab, or None if no trend is available. Registered at ANALYST tier by the caller
    (never PRIMARY/citable_as_fact) so it can only ever surface as reported context, matching the
    "not cross-verified" caveat already embedded in the sentence itself. WHY: promoter behavior is
    a core Indian-investor signal the Research tab already fetches, but until now the Ask tab had
    no access to it at all, so a direct question ("has the promoter been selling?") could never
    be grounded even when the app already had the answer sitting in cache."""
    if not trend:
        return None
    return FetchedDocument(PROMOTER_TREND_SOURCE_ID, f"Promoter shareholding for {symbol}: {trend}",
                           url="", locator=f"{symbol} promoter shareholding trend")
