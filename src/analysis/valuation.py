"""Historical median P/E, the baseline for the valuation opinion.

The current P/E is a cross-verified figure (a fact). The median P/E is a COMPUTED reference:
for each fiscal year, P/E = (year-end price) / (EPS), where EPS = net profit / shares; the
median across years is the "own history" baseline. It is used only for the caveated valuation
tier (an opinion), never presented as a verified fact, so a best-effort single-source
computation is acceptable here.
"""
from __future__ import annotations

import statistics

import pandas as pd


def median_pe_from_annuals(net_profit_by_year: dict[int, float],
                           price_by_year: dict[int, float],
                           shares: float | None) -> float | None:
    """Median of (year-end price / EPS) across years. Needs >= 2 years and positive earnings."""
    if not shares or shares <= 0:
        return None
    pes: list[float] = []
    for year, net_profit in net_profit_by_year.items():
        price = price_by_year.get(year)
        if price is None or net_profit is None or net_profit <= 0 or price <= 0:
            continue
        eps = net_profit / shares
        if eps <= 0:
            continue
        pes.append(price / eps)
    return statistics.median(pes) if len(pes) >= 2 else None


def _price_by_fiscal_year(history, years: list[int]) -> dict[int, float]:
    """Close on/just before 31 March of each fiscal year (Indian FY end)."""
    out: dict[int, float] = {}
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return out
    close = history["Close"].dropna()
    try:
        close.index = close.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    for year in years:
        upto = close[close.index <= pd.Timestamp(f"{year}-03-31")]
        if not upto.empty:
            out[year] = float(upto.iloc[-1])
    return out


def compute_median_pe(symbol: str) -> float | None:
    """Best-effort historical median P/E from yfinance (income statement + price history)."""
    import yfinance as yf

    from ..data.figure_sources import _num, _safe, _series_from_statement
    from ..data.yfinance_provider import to_yahoo_symbol
    ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
    if ticker is None:
        return None
    income = _safe(lambda: ticker.income_stmt)
    net_profit = _series_from_statement(income, ["Net Income", "Net Income Common Stockholders"])
    if not net_profit:
        return None
    info = _safe(lambda: ticker.info) or {}
    shares = _num(info.get("sharesOutstanding"))
    history = _safe(lambda: ticker.history(period="6y"))
    prices = _price_by_fiscal_year(history, list(net_profit))
    return median_pe_from_annuals(net_profit, prices, shares)
