"""yfinance adapter for Indian listings. v1 data source.

Every method degrades to a None/empty result on failure rather than raising, so one bad
ticker never takes down the dashboard. The cost is that callers must treat missing data
as missing (the analysis layer already does).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..constants import BSE_SUFFIX, DEFAULT_HISTORY_PERIOD, NSE_SUFFIX
from .provider import MarketDataProvider


def to_yahoo_symbol(symbol: str) -> str:
    """Map a bare Indian symbol to its Yahoo Finance ticker.

    Indices (start with ^) and already-suffixed symbols pass through. A purely numeric
    symbol is treated as a BSE scrip code; everything else defaults to NSE.
    """
    s = symbol.strip().upper()
    if s.startswith("^") or s.endswith(NSE_SUFFIX) or s.endswith(BSE_SUFFIX):
        return s
    if s.isdigit():
        return s + BSE_SUFFIX
    return s + NSE_SUFFIX


class YFinanceProvider(MarketDataProvider):
    def current_prices(self, symbols: list[str]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for sym in symbols:
            out[sym] = self._one_price(sym)
        return out

    def _one_price(self, symbol: str) -> float | None:
        try:
            ticker = yf.Ticker(to_yahoo_symbol(symbol))
            fast = getattr(ticker, "fast_info", None)
            if fast is not None:
                price = fast.get("last_price") or fast.get("lastPrice")
                if price:
                    return float(price)
            hist = ticker.history(period="5d")
            closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
            if not closes.empty:
                return float(closes.iloc[-1])
        except Exception:
            return None
        return None

    def fundamentals(self, symbol: str) -> dict:
        try:
            info = yf.Ticker(to_yahoo_symbol(symbol)).info or {}
        except Exception:
            info = {}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "currency": info.get("currency"),
        }

    def history(self, symbol: str, period: str = DEFAULT_HISTORY_PERIOD) -> pd.DataFrame:
        try:
            return yf.Ticker(to_yahoo_symbol(symbol)).history(period=period)
        except Exception:
            return pd.DataFrame()

    def index_quote(self, index_symbol: str) -> dict:
        try:
            hist = yf.Ticker(index_symbol).history(period="5d")
            closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
            if closes.empty:
                return {"symbol": index_symbol, "price": None, "change_pct": None}
            price = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else price
            change_pct = ((price - prev) / prev * 100.0) if prev else 0.0
            return {"symbol": index_symbol, "price": price, "change_pct": change_pct}
        except Exception:
            return {"symbol": index_symbol, "price": None, "change_pct": None}
