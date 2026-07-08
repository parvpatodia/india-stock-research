"""Figure sources: pull the analysis framework's inputs from real providers.

A FigureSource returns, for a symbol, the framework's figures (net profit, cash flow, debt,
etc.) as plain numbers tagged with its source id. The pipeline gathers figures from several
sources and cross-verifies them. With only ONE source wired (yfinance), figures come back
single-source and are NOT trustworthy, so the verdict stays low-confidence by design; adding
a second independent source (an owner API adapter) makes agreeing figures cross-verify
automatically. Nothing here fakes a second source to manufacture confidence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .yfinance_provider import to_yahoo_symbol

# The figure names the analysis framework (src/analysis/framework.py) consumes.
FRAMEWORK_FIGURES = (
    "current_pe", "median_pe", "operating_cash_flow", "net_profit",
    "total_debt", "equity", "ebit", "interest_expense", "promoter_pledge_pct",
)


class FigureSource(ABC):
    source_id: str

    @abstractmethod
    def figures(self, symbol: str) -> dict[str, float | None]:
        """Return {figure_name: value_or_None} for every FRAMEWORK_FIGURES name."""


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
        out["ebit"] = _latest(income, ["EBIT", "Operating Income"])
        interest = _latest(income, ["Interest Expense", "Interest Expense Non Operating"])
        out["interest_expense"] = abs(interest) if interest is not None else None
        out["operating_cash_flow"] = _latest(
            cash, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        out["total_debt"] = _latest(balance, ["Total Debt"]) or _num(info.get("totalDebt"))
        out["equity"] = _latest(
            balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
        return out
