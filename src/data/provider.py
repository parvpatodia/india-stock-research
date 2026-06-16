"""The single network boundary. Analysis code depends on this interface, not on yfinance,
so swapping to a broker API (Upstox, Zerodha Kite) means adding one adapter and nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def current_prices(self, symbols: list[str]) -> dict[str, float | None]:
        """Map each symbol to its latest price, or None if unavailable."""

    @abstractmethod
    def fundamentals(self, symbol: str) -> dict:
        """Best-effort fundamentals. Missing fields are None, never fabricated."""

    @abstractmethod
    def history(self, symbol: str, period: str) -> pd.DataFrame:
        """OHLCV history with a 'Close' column. Empty DataFrame if unavailable."""

    @abstractmethod
    def index_quote(self, index_symbol: str) -> dict:
        """{'symbol', 'price', 'change_pct'} for an index. None values if unavailable."""
