"""Screener.in figure source: a free, independent, structured second source.

Screener publishes each company's annual P&L, balance sheet, and cash flow (in Rs crore) on a
public page with no login. This parses those tables into the framework's figures and converts
crore -> absolute rupees IN CODE (Screener is always in crore), so the values cross-check
against yfinance's absolute figures. It is an unofficial source (scraping, ToS grey, can break
when the page changes), which is exactly why every figure still has to agree with yfinance
before it is trusted; a parse error or a page change shows up as a conflict, not a bad fact.
"""
from __future__ import annotations

import io
import re
from typing import Callable

import pandas as pd

from .figure_sources import FRAMEWORK_FIGURES, FigureSource

_CRORE = 1e7
_PE_RE = re.compile(r"Stock P/E.*?([\d,]+\.?\d+)", re.IGNORECASE | re.DOTALL)
_YEAR_RE = re.compile(r"(\d{4})\s*$")
_MONTH_YEAR_RE = re.compile(r"([A-Za-z]{3})\s+(\d{4})\s*$")


def _clean_label(value) -> str:
    return str(value).replace("\xa0", " ").replace("+", "").strip().lower()


def _num(value) -> float | None:
    try:
        s = str(value).replace(",", "").replace("%", "").strip()
        if s in ("", "nan", "-", "none"):
            return None
        f = float(s)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _is_annual(df) -> bool:
    # WHY (resilience): Screener's quarterly results table precedes the annual P&L and shares its
    # row labels, so "distinct years in the first few columns" misfires on a quarter that straddles
    # a calendar year (e.g. Dec 2023 / Mar 2024 / Jun 2024). Annual columns instead all share ONE
    # fiscal-year-end month across >=2 distinct years; a quarterly table cycles through months. Test
    # every dated column so a truncated view can't fool it.
    months: set[str] = set()
    years: set[str] = set()
    for col in list(df.columns)[1:]:
        m = _MONTH_YEAR_RE.search(str(col).strip())
        if m:
            months.add(m.group(1).lower())
            years.add(m.group(2))
    return len(years) >= 2 and len(months) <= 1


def _latest_annual(df, label: str) -> float | None:
    if df is None:
        return None
    # WHY: take the rightmost column whose header is a fiscal year (e.g. "Mar 2024"), skipping
    # a "TTM" column. Otherwise a P&L row would be read TTM while yfinance reports the latest
    # full year, a period mismatch that would (correctly but unhelpfully) show as a conflict.
    year_idx = None
    for i, col in enumerate(df.columns):
        if i > 0 and _YEAR_RE.search(str(col)):
            year_idx = i
    if year_idx is None:
        return None
    for _, row in df.iterrows():
        lab = _clean_label(row.iloc[0])
        if lab == label or lab.startswith(label):
            return _num(row.iloc[year_idx])
    return None


def _find_tables(html: str):
    """Return (pnl, balance, cash) DataFrames from Screener HTML, or Nones."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    pnl = balance = cash = None
    for df in tables:
        if df.shape[1] < 2:
            continue
        labels = {_clean_label(v) for v in df.iloc[:, 0]}
        if pnl is None and any("net profit" in l for l in labels) \
                and any("operating profit" in l for l in labels) and _is_annual(df):
            pnl = df
        if balance is None and any("borrowings" in l for l in labels) \
                and any("equity capital" in l for l in labels):
            balance = df
        if cash is None and any("cash from operating activity" in l for l in labels):
            cash = df
    return pnl, balance, cash


def _annual_series(df, label: str) -> dict[int, float]:
    """{fiscal_year: value} across every year column for the first matching row."""
    out: dict[int, float] = {}
    if df is None:
        return out
    for _, row in df.iterrows():
        lab = _clean_label(row.iloc[0])
        if lab == label or lab.startswith(label):
            for col in list(df.columns)[1:]:
                m = _YEAR_RE.search(str(col))
                n = _num(row[col])
                if m and n is not None:
                    out[int(m.group(1))] = n
            break
    return out


def parse_screener_series(html: str) -> dict[str, dict[int, float]]:
    """Pure: per-fiscal-year series for the statement figures (absolute rupees)."""
    pnl, balance, cash = _find_tables(html)
    net = _annual_series(pnl, "net profit")
    pbt = _annual_series(pnl, "profit before tax")
    interest = _annual_series(pnl, "interest")
    sales = _annual_series(pnl, "sales")
    ocf = _annual_series(cash, "cash from operating activity")
    debt = _annual_series(balance, "borrowings")
    eqcap = _annual_series(balance, "equity capital")
    reserves = _annual_series(balance, "reserves")
    tassets = _annual_series(balance, "total liabilities")
    series = {
        "net_profit": {y: v * _CRORE for y, v in net.items()},
        "operating_cash_flow": {y: v * _CRORE for y, v in ocf.items()},
        "total_debt": {y: v * _CRORE for y, v in debt.items()},
        "interest_expense": {y: abs(v) * _CRORE for y, v in interest.items()},
        "ebit": {y: (pbt[y] + interest[y]) * _CRORE for y in pbt if y in interest},
        "equity": {y: (eqcap[y] + reserves[y]) * _CRORE for y in eqcap if y in reserves},
        "total_assets": {y: v * _CRORE for y, v in tassets.items()},
        "revenue": {y: v * _CRORE for y, v in sales.items()},
    }
    return {name: yearmap for name, yearmap in series.items() if yearmap}


def parse_screener_figures(html: str) -> dict[str, float | None]:
    """Pure: parse Screener HTML into framework figures (absolute rupees). Missing -> None."""
    out: dict[str, float | None] = {name: None for name in FRAMEWORK_FIGURES}
    pnl, balance, cash = _find_tables(html)

    net_profit = _latest_annual(pnl, "net profit")
    pbt = _latest_annual(pnl, "profit before tax")
    interest = _latest_annual(pnl, "interest")
    ocf = _latest_annual(cash, "cash from operating activity")
    debt = _latest_annual(balance, "borrowings")
    equity_capital = _latest_annual(balance, "equity capital")
    reserves = _latest_annual(balance, "reserves")

    if net_profit is not None:
        out["net_profit"] = net_profit * _CRORE
    if ocf is not None:
        out["operating_cash_flow"] = ocf * _CRORE
    if debt is not None:
        out["total_debt"] = debt * _CRORE
    if interest is not None:
        out["interest_expense"] = abs(interest) * _CRORE
    if pbt is not None and interest is not None:
        out["ebit"] = (pbt + interest) * _CRORE  # EBIT = profit before tax + interest
    if equity_capital is not None and reserves is not None:
        out["equity"] = (equity_capital + reserves) * _CRORE  # shareholders' equity
    total_assets = _latest_annual(balance, "total liabilities")  # Screener's grand total = assets
    if total_assets is not None:
        out["total_assets"] = total_assets * _CRORE
    revenue = _latest_annual(pnl, "sales")
    if revenue is not None:
        out["revenue"] = revenue * _CRORE
    m = _PE_RE.search(html)
    if m:
        out["current_pe"] = _num(m.group(1))
    return out


class ScreenerFigureSource(FigureSource):
    source_id = "screener"

    def __init__(self, fetcher: Callable[[str], str | None] | None = None):
        self._fetcher = fetcher or self._http_fetch
        # WHY: memoize the page per symbol. build_company_report calls figures() AND
        # figures_by_year() (and the series gather calls it again) for the same symbol, so without
        # this each stock hit Screener ~3x -> a 32-stock batch made ~96 requests and tripped
        # Cloudflare's rate limit. One fetch per symbol keeps the batch to ~32.
        self._cache: dict[str, str | None] = {}

    def _fetch_cached(self, symbol: str) -> str | None:
        key = symbol.strip().upper()
        if key not in self._cache:
            self._cache[key] = self._fetcher(key)
        return self._cache[key]

    @staticmethod
    def _http_fetch(symbol: str) -> str | None:
        # WHY: browser-like headers + retry-with-backoff. Screener sits behind Cloudflare, which
        # rate-limits bursts from datacenter IPs (Streamlit Cloud), starving cross-verification to
        # a single source. Backoff gives a transient rate-limit time to clear; best-effort, not
        # guaranteed past a hard JS challenge.
        import time

        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                      "image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.screener.in/",
            "Connection": "keep-alive",
        }
        session = requests.Session()
        for attempt in range(3):
            for path in (f"https://www.screener.in/company/{symbol}/consolidated/",
                         f"https://www.screener.in/company/{symbol}/"):
                try:
                    resp = session.get(path, headers=headers, timeout=30)
                    if resp.status_code == 200 and resp.text:
                        return resp.text
                except Exception:
                    continue
            if attempt < 2:
                time.sleep(2 * (attempt + 1))   # 2s, 4s backoff before retrying
        return None

    def figures(self, symbol: str) -> dict[str, float | None]:
        html = self._fetch_cached(symbol)
        if not html:
            return {name: None for name in FRAMEWORK_FIGURES}
        return parse_screener_figures(html)

    def figures_by_year(self, symbol: str) -> dict[str, dict[int, float]]:
        html = self._fetch_cached(symbol)
        if not html:
            return {}
        return parse_screener_series(html)
