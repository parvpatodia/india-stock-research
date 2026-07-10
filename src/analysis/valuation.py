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


def median_pe_from_eps(price_by_year: dict[int, float],
                       eps_by_year: dict[int, float]) -> float | None:
    """Median of (year-end price / period EPS). Uses each year's own EPS, so it is correct for
    companies that issued shares over time (unlike dividing by today's share count)."""
    pes: list[float] = []
    for year, eps in eps_by_year.items():
        price = price_by_year.get(year)
        if price is None or eps is None or eps <= 0 or price <= 0:
            continue
        pes.append(price / eps)
    return statistics.median(pes) if len(pes) >= 2 else None


def median_pe_from_annuals(net_profit_by_year: dict[int, float],
                           price_by_year: dict[int, float],
                           shares: float | None) -> float | None:
    """Fallback when per-year EPS isn't available: EPS = net profit / CURRENT shares. WHY caveat:
    using today's share count understates past EPS for a company that diluted, so the median P/E
    reads a bit high (stock looks cheaper). Only used when the income statement has no EPS row."""
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


def _price_by_fiscal_year(history, year_ends: dict[int, "pd.Timestamp"]) -> dict[int, float]:
    """Close on/just before each fiscal year's ACTUAL statement period-end date. WHY: pricing
    every company at a hardcoded 31 March mis-pairs a Dec-FY-end company's December EPS with a
    March price (~9 months off), distorting its median P/E; use the real period-end per year."""
    out: dict[int, float] = {}
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return out
    close = history["Close"].dropna()
    try:
        close.index = close.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    for year, end in year_ends.items():
        try:
            cutoff = pd.Timestamp(end)
            if cutoff.tz is not None:
                cutoff = cutoff.tz_localize(None)
        except (TypeError, ValueError):
            continue
        upto = close[close.index <= cutoff]
        if not upto.empty:
            out[year] = float(upto.iloc[-1])
    return out


def compute_median_pe(symbol: str) -> float | None:
    """Best-effort historical median P/E from yfinance (income statement + price history)."""
    import yfinance as yf

    from ..data.figure_sources import _num, _safe, _series_from_statement, _year_of
    from ..data.yfinance_provider import to_yahoo_symbol
    ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
    if ticker is None:
        return None
    income = _safe(lambda: ticker.income_stmt)
    net_profit = _series_from_statement(income, ["Net Income", "Net Income Common Stockholders"])
    eps_series = _series_from_statement(income, ["Diluted EPS", "Basic EPS"])
    if not net_profit and not eps_series:
        return None
    # Map each fiscal year to its actual statement period-end date (yfinance columns are period
    # ends), so prices are paired at the right time for a March- OR December-FY-end company.
    year_ends: dict[int, pd.Timestamp] = {}
    if income is not None and not getattr(income, "empty", True):
        for col in income.columns:
            y = _year_of(col)
            if y is not None and y not in year_ends:
                try:
                    year_ends[y] = pd.Timestamp(col)
                except (TypeError, ValueError):
                    pass
    target_years = list(eps_series) or list(net_profit)
    # WHY auto_adjust=False (real money, data quality; live-verified): the historical median P/E
    # pairs prices with yfinance's income_stmt EPS, which is retroactively restated to the CURRENT
    # split-adjusted share basis (confirmed on NESTLEIND: pre-split FY2021/22 EPS both report the
    # current ~192.8cr shares). yfinance's Close is ALWAYS split-adjusted in BOTH modes -- the
    # auto_adjust flag only toggles the DIVIDEND adjustment (confirmed: RELIANCE Mar-2022 Close is
    # 1216 either way = ~2400 pre-bonus / 2). The DEFAULT (auto_adjust=True) additionally applies
    # dividends, understating past prices (~11% over 3y on a ~3%-yield name like ITC) and biasing
    # the "cheap vs its own history" margin-of-safety call toward "expensive". auto_adjust=False
    # gives split-adjusted-but-NOT-dividend-adjusted Close -- exactly the basis the EPS is on.
    history = _safe(lambda: ticker.history(period="6y", auto_adjust=False))
    prices = _price_by_fiscal_year(history, {y: year_ends[y] for y in target_years
                                             if y in year_ends})
    # Prefer period-correct EPS (no dilution error); fall back to net profit / current shares.
    if eps_series:
        m = median_pe_from_eps(prices, eps_series)
        if m is not None:
            return m
    info = _safe(lambda: ticker.info) or {}
    return median_pe_from_annuals(net_profit, prices, _num(info.get("sharesOutstanding")))
