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
    def pnl_pct(self) -> float | None:
        # WHY None (not 0.0) for a zero-cost lot (real money, honesty): a bonus/IPO lot at 0 cost is
        # ALL gain, so its PERCENT return is undefined, not 0% -- "0.0%" reads as break-even and hides
        # the gain. Return None (the holdings table shows a blank); pnl_abs still carries the real
        # rupee gain. Also guards the div-by-zero the old code prevented.
        return (self.pnl_abs / self.invested * 100.0) if self.invested else None


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
    def total_pnl_pct(self) -> float | None:
        # WHY None (not 0.0) when total cost is 0 (real money, honesty; mirrors PositionAnalysis.
        # pnl_pct): an all-zero-cost book (only bonus/IPO lots, or blank/0 cost cells) is ALL gain,
        # so its PERCENT return is UNDEFINED -- a portfolio "0.0%" reads as break-even and HIDES the
        # gain, exactly the misread the per-position guard prevents. total_pnl_abs still carries the
        # real rupee gain; the caller shows the metric without a percentage delta.
        return (self.total_pnl_abs / self.total_invested * 100.0) if self.total_invested else None
