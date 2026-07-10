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
# WHY: "promoter_pledge_pct" is declared here but NOT populated by either real source below
# (YFinanceFigureSource or screener_source.parse_screener_figures) — see the note on
# analysis/framework.py:promoter_pledge for the live investigation confirming this. It always
# reads unavailable in production today; do not assume it works without checking that first.
FRAMEWORK_FIGURES = (
    "current_pe", "median_pe", "operating_cash_flow", "net_profit",
    "total_debt", "equity", "ebit", "interest_expense", "promoter_pledge_pct",
    "total_assets", "revenue", "dividend_yield_pct",
)
# Figures that come from an annual fiscal-year statement (align by year across sources).
YEAR_FIGURES = ("net_profit", "operating_cash_flow", "total_debt", "equity", "ebit",
                "interest_expense", "total_assets", "revenue")
# Figures that are point-in-time (current), not tied to a fiscal year.
POINT_FIGURES = ("current_pe", "median_pe", "promoter_pledge_pct", "dividend_yield_pct")
# Unit classification, shared by every place that formats or labels a figure by name (the Ask
# tab's verified-figures document, the expert-correction UI, ...), so "is this rupees, a ratio,
# or a percentage" is defined ONCE and can't drift out of sync between call sites.
RATIO_FIGURES = frozenset({"current_pe", "median_pe"})                       # e.g. "18.2x"
PERCENT_FIGURES = frozenset({"promoter_pledge_pct", "dividend_yield_pct"})    # e.g. "0.5%"


def _indian_group(int_digits: str) -> str:
    """Group an integer digit-string in the Indian convention: the last 3 digits, then 2 at a
    time (e.g. 1056499 -> '10,56,499'). WHY: the reader is an Indian investor; Western thousands
    grouping of a crore figure reads wrong to them."""
    if len(int_digits) <= 3:
        return int_digits
    head, last3 = int_digits[:-3], int_digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + last3


def _format_money(value: float) -> str:
    """Rupees in the Indian crore/lakh convention. WHY (real money, UI honesty, Ask answer
    quality): figures are stored in ABSOLUTE rupees, so a real net profit rendered raw is
    '₹790,000,000,000.00' -- a 12-digit string a parent has to count zeros on. Indian investors
    read financials in crore (1e7) / lakh (1e5); showing them that way is how the reader, and the
    Ask model quoting the grounding document, naturally states them. Trailing zeros are stripped
    so a whole-crore figure reads clean and its digits line up with a '79,000 crore' phrasing
    under numbers_grounded."""
    a = abs(value)
    if a >= 1e7:
        num, unit = value / 1e7, " crore"
    elif a >= 1e5:
        num, unit = value / 1e5, " lakh"
    else:
        num, unit = value, ""
    s = f"{num:.2f}".rstrip("0").rstrip(".")   # up to 2 decimals, no trailing zeros
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    int_part, _, dec_part = s.partition(".")
    grouped = _indian_group(int_part) + (f".{dec_part}" if dec_part else "")
    return f"{'-' if neg else ''}₹{grouped}{unit}"


def format_figure_value(name: str, value: float) -> str:
    """Render a figure's value in ITS actual unit (ratio / percent / rupees), not a bare number.
    WHY: a bare '25.00' is genuinely ambiguous between a 25% pledge and Rs.25 -- every place that
    displays a figure by name (the Research tab's evidence table, the PDF export, the Ask tab's
    verified-figures document) must agree on this, so unit awareness is defined here ONCE."""
    if name in RATIO_FIGURES:
        return f"{value:.1f}x"
    if name in PERCENT_FIGURES:
        return f"{value:.1f}%"
    return _format_money(value)


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


def _has_any_data(df, label: str) -> bool:
    return label in df.index and any(_num(v) is not None for v in df.loc[label])


def _latest_pair(df, primary_candidates: list[str],
                  secondary_candidates: list[str]) -> tuple[float | None, float | None]:
    """The most recent SINGLE period where a primary and a secondary statement row are BOTH
    non-NaN, returned as (primary_value, secondary_value) -- or (None, None) if no such period
    exists. WHY: pulling each row's own most-recent non-null value independently (two separate
    _latest() calls) risks silently pairing values from DIFFERENT fiscal periods if one row has
    a gap the other doesn't (e.g. this year's Interest Expense + last year's Pretax Income) --
    a combined figure like EBIT would be meaningless even though each half looks fine alone.
    Matches figures_by_year()'s existing period-aligned pairing for the same figure.

    WHY skip an entirely-empty candidate label (not just an absent one): matches _latest()'s own
    fallback behaviour, which tries every candidate row and only gives up on one if it has NO
    data at all. An earlier version of this function locked onto the first candidate label merely
    PRESENT in the index, even when every value in that row was NaN, silently losing a good
    figure whenever a company's statement uses the first-listed row name but leaves it empty
    (found by adversarial review) -- exactly the gappy-statement case these synonym lists exist
    to handle."""
    if df is None or getattr(df, "empty", True):
        return None, None
    primary_label = next((label for label in primary_candidates if _has_any_data(df, label)), None)
    secondary_label = next(
        (label for label in secondary_candidates if _has_any_data(df, label)), None)
    if primary_label is None or secondary_label is None:
        return None, None
    for col in df.columns:  # most recent first
        p, s = _num(df.loc[primary_label, col]), _num(df.loc[secondary_label, col])
        if p is not None and s is not None:
            return p, s
    return None, None


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
        # Use _latest_pair (not two independent _latest() calls) so pretax and interest come from
        # the SAME fiscal period -- see _latest_pair's docstring for why pairing independently
        # found "latest" values can silently mix two different years.
        pretax, interest_for_ebit = _latest_pair(
            income, ["Pretax Income", "Income Before Tax", "Pre Tax Income"],
            ["Interest Expense", "Interest Expense Non Operating"])
        if pretax is not None and interest_for_ebit is not None:
            out["ebit"] = pretax + abs(interest_for_ebit)
        else:
            out["ebit"] = _latest(income, ["EBIT", "Operating Income"])
        out["operating_cash_flow"] = _latest(
            cash, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        # WHY: explicit None check, not `or`. A debt-free company's 0.0 is falsy and would wrongly
        # fall through to info's (possibly missing/stale) value, hiding a legitimate "no debt".
        td = _latest(balance, ["Total Debt"])
        out["total_debt"] = td if td is not None else _num(info.get("totalDebt"))
        out["equity"] = _latest(
            balance, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
        ta = _latest(balance, ["Total Assets"])
        out["total_assets"] = ta if ta is not None else _num(info.get("totalAssets"))
        out["revenue"] = _latest(income, ["Total Revenue", "Operating Revenue",
                                          "Total Revenue As Reported"])
        # WHY: yfinance's "dividendYield" is already a percentage number (e.g. 0.47 meaning
        # 0.47%), matching Screener's displayed "Dividend Yield X%" directly -- confirmed live
        # across several stocks. Do NOT use "trailingAnnualDividendYield", which is a FRACTION
        # (0.0047) and would silently create a 100x scale mismatch against Screener's parse.
        out["dividend_yield_pct"] = _num(info.get("dividendYield"))
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
            "total_assets": _series_from_statement(balance, ["Total Assets"]),
            "revenue": _series_from_statement(income, ["Total Revenue", "Operating Revenue",
                                                       "Total Revenue As Reported"]),
        }
        return {name: yearmap for name, yearmap in series.items() if yearmap}
