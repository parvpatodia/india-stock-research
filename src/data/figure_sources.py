"""Figure sources: pull the analysis framework's inputs from real providers.

A FigureSource returns, for a symbol, the framework's figures (net profit, cash flow, debt,
etc.) as plain numbers tagged with its source id. The pipeline gathers figures from several
sources and cross-verifies them. With only ONE source wired (yfinance), figures come back
single-source and are NOT trustworthy, so the verdict stays low-confidence by design; adding
a second independent source (an owner API adapter) makes agreeing figures cross-verify
automatically. Nothing here fakes a second source to manufacture confidence.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .yfinance_provider import to_yahoo_symbol

# The figure names the analysis framework (src/analysis/framework.py) consumes.
FRAMEWORK_FIGURES = (
    "current_pe", "median_pe", "operating_cash_flow", "net_profit",
    "total_debt", "equity", "ebit", "interest_expense", "promoter_pledge_pct",
)
# Figures that come from an annual fiscal-year statement (align by year across sources).
YEAR_FIGURES = ("net_profit", "operating_cash_flow", "total_debt", "equity", "ebit",
                "interest_expense")
# Figures that are point-in-time (current), not tied to a fiscal year.
POINT_FIGURES = ("current_pe", "median_pe", "promoter_pledge_pct")

_YEAR = re.compile(r"(\d{4})")


class FigureSource(ABC):
    source_id: str

    @abstractmethod
    def figures(self, symbol: str) -> dict[str, float | None]:
        """Return {figure_name: value_or_None} for every FRAMEWORK_FIGURES name (latest)."""

    def figures_by_year(self, symbol: str) -> dict[str, dict[int, float]]:
        """Optional per-fiscal-year series {figure: {year: value}} for YEAR_FIGURES, used to
        cross-verify the SAME year across sources. Default: none (source is scalar-only)."""
        return {}


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return f if f == f else None  # reject NaN
    except (TypeError, ValueError):
        return None


def _safe(getter):
    try:
        return getter()
    except Exception:
        return None


def _latest(df, candidates: list[str]) -> float | None:
    """Most-recent non-NaN value for the first matching row label in a yfinance statement."""
    if df is None or getattr(df, "empty", True):
        return None
    for label in candidates:
        if label in df.index:
            for value in df.loc[label]:  # columns are periods, most recent first
                n = _num(value)
                if n is not None:
                    return n
    return None


def _year_of(col) -> int | None:
    year = getattr(col, "year", None)
    if isinstance(year, int):
        return year
    m = _YEAR.search(str(col))
    return int(m.group(1)) if m else None


def _series_from_statement(df, candidates: list[str]) -> dict[int, float]:
    """Build {fiscal_year: value} for the first matching row label across all period columns."""
    out: dict[int, float] = {}
    if df is None or getattr(df, "empty", True):
        return out
    for label in candidates:
        if label in df.index:
            for col, value in df.loc[label].items():
                year, n = _year_of(col), _num(value)
                if year is not None and n is not None and year not in out:
                    out[year] = n
            break
    return out


class YFinanceFigureSource(FigureSource):
    """One real, free source. Provides what yfinance actually exposes; leaves the rest None
    (e.g. historical median P/E and promoter pledge are not available here)."""

    source_id = "yfinance"

    def figures(self, symbol: str) -> dict[str, float | None]:
        import yfinance as yf
        out: dict[str, float | None] = {name: None for name in FRAMEWORK_FIGURES}
        ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
        if ticker is None:
            return out
        info = _safe(lambda: ticker.info) or {}
        income = _safe(lambda: ticker.income_stmt)
        balance = _safe(lambda: ticker.balance_sheet)
        cash = _safe(lambda: ticker.cashflow)

        out["current_pe"] = _num(info.get("trailingPE"))
        out["net_profit"] = _latest(income, ["Net Income", "Net Income Common Stockholders"])
        interest = _latest(income, ["Interest Expense", "Interest Expense Non Operating"])
        out["interest_expense"] = abs(interest) if interest is not None else None
        # WHY: define EBIT as pre-tax income + interest, matching Screener's "PBT + interest",
        # so the two sources compare the same concept. Fall back to Operating Income.
        pretax = _latest(income, ["Pretax Income", "Income Before Tax", "Pre Tax Income"])
        if pretax is not None and interest is not None:
            out["ebit"] = pretax + abs(interest)
        else:
            out["ebit"] = _latest(income, ["EBIT", "Operating Income"])
        out["operating_cash_flow"] = _latest(
            cash, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        out["total_debt"] = _latest(balance, ["Total Debt"]) or _num(info.get("totalDebt"))
        out["equity"] = _latest(
            balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
        return out

    def figures_by_year(self, symbol: str) -> dict[str, dict[int, float]]:
        import yfinance as yf
        ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
        if ticker is None:
            return {}
        income = _safe(lambda: ticker.income_stmt)
        balance = _safe(lambda: ticker.balance_sheet)
        cash = _safe(lambda: ticker.cashflow)
        interest = _series_from_statement(income, ["Interest Expense", "Interest Expense Non Operating"])
        pretax = _series_from_statement(income, ["Pretax Income", "Income Before Tax", "Pre Tax Income"])
        ebit_series = {y: pretax[y] + abs(interest[y]) for y in pretax if y in interest} \
            or _series_from_statement(income, ["EBIT", "Operating Income"])
        series = {
            "net_profit": _series_from_statement(income, ["Net Income", "Net Income Common Stockholders"]),
            "ebit": ebit_series,
            "interest_expense": {y: abs(v) for y, v in interest.items()},
            "operating_cash_flow": _series_from_statement(
                cash, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
            "total_debt": _series_from_statement(balance, ["Total Debt"]),
            "equity": _series_from_statement(
                balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"]),
        }
        return {name: yearmap for name, yearmap in series.items() if yearmap}
