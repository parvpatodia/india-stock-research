"""Portfolio data models. Plain dataclasses, no I/O, no network."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Holding:
    """One line of the uploaded portfolio: what the user owns and what they paid."""
    symbol: str
    quantity: float
    avg_cost: float
    sector: str = "Unknown"


@dataclass(frozen=True)
class PositionAnalysis:
    """A holding priced at the current market, with derived figures."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    sector: str
    weight: float  # fraction of total portfolio market value, 0..1

    @property
    def invested(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_abs(self) -> float:
        return self.market_value - self.invested

    @property
    def pnl_pct(self) -> float:
        # WHY: guard zero-cost lots (bonus/IPO allotment at 0) against div-by-zero.
        return (self.pnl_abs / self.invested * 100.0) if self.invested else 0.0


@dataclass(frozen=True)
class PortfolioAnalysis:
    """Whole-portfolio rollup. Missing-priced symbols are reported, not silently dropped."""
    positions: list[PositionAnalysis]
    total_invested: float
    total_value: float
    missing_symbols: list[str]
    hhi: float                       # Herfindahl index of weights, 0..1
    effective_holdings: float        # 1/HHI: how many "equal" names the book behaves like
    top_holding_weight: float
    sector_weights: dict[str, float]

    @property
    def total_pnl_abs(self) -> float:
        return self.total_value - self.total_invested

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl_abs / self.total_invested * 100.0) if self.total_invested else 0.0
