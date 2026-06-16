"""Portfolio analytics. Pure functions: data is injected, nothing here touches the network.

Keeping this layer free of I/O is what lets the math be unit-tested against known inputs,
which matters because a silent bug in P&L or weights costs real money.
"""
from __future__ import annotations

import math

import pandas as pd

from ..constants import TRADING_DAYS_PER_YEAR
from .models import Holding, PortfolioAnalysis, PositionAnalysis


def analyze_portfolio(holdings: list[Holding],
                      prices: dict[str, float]) -> PortfolioAnalysis:
    """Price the book and roll up. prices maps symbol -> current price.

    A symbol with no price (None or missing) is reported in missing_symbols and excluded
    from every total, so it cannot distort value, P&L, or weights.
    """
    priced = [(h, prices.get(h.symbol)) for h in holdings]
    usable = [(h, p) for h, p in priced if p is not None]
    missing = [h.symbol for h, p in priced if p is None]

    total_value = sum(h.quantity * p for h, p in usable)
    total_invested = sum(h.quantity * h.avg_cost for h, p in usable)

    positions: list[PositionAnalysis] = []
    for h, p in usable:
        market_value = h.quantity * p
        weight = (market_value / total_value) if total_value else 0.0
        positions.append(PositionAnalysis(
            symbol=h.symbol, quantity=h.quantity, avg_cost=h.avg_cost,
            current_price=p, sector=h.sector, weight=weight,
        ))

    hhi = sum(pos.weight ** 2 for pos in positions)
    effective_holdings = (1.0 / hhi) if hhi else 0.0
    top_holding_weight = max((pos.weight for pos in positions), default=0.0)

    sector_weights: dict[str, float] = {}
    for pos in positions:
        sector_weights[pos.sector] = sector_weights.get(pos.sector, 0.0) + pos.weight

    return PortfolioAnalysis(
        positions=positions,
        total_invested=total_invested,
        total_value=total_value,
        missing_symbols=missing,
        hhi=hhi,
        effective_holdings=effective_holdings,
        top_holding_weight=top_holding_weight,
        sector_weights=sector_weights,
    )


# --- Risk metrics. Inputs are price/return series, injected by the caller. ---

def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    return float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity of the asset to the benchmark: cov(a, b) / var(b)."""
    df = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    var_b = b.var()
    if var_b == 0:
        return 0.0
    return float(a.cov(b) / var_b)


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough drop as a negative fraction, e.g. -0.32 = down 32%."""
    if close is None or close.empty:
        return 0.0
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(drawdown.min())


def portfolio_daily_returns(close_by_symbol: dict[str, pd.Series],
                            weights: dict[str, float]) -> pd.Series:
    """Weighted daily return of the book, over dates where all series overlap.

    Weights are renormalized across the symbols that actually have history, so a name
    missing price history does not silently count as zero return.
    """
    frames = {sym: daily_returns(close) for sym, close in close_by_symbol.items()
              if close is not None and not close.empty}
    if not frames:
        return pd.Series(dtype=float)
    df = pd.DataFrame(frames).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    w = pd.Series({s: weights.get(s, 0.0) for s in df.columns})
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    return df.mul(w, axis=1).sum(axis=1)
