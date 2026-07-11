"""Portfolio analytics. Pure functions: data is injected, nothing here touches the network.

Keeping this layer free of I/O is what lets the math be unit-tested against known inputs,
which matters because a silent bug in P&L or weights costs real money.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import pandas as pd

from ..constants import TRADING_DAYS_PER_YEAR
from .models import Holding, PortfolioAnalysis, PositionAnalysis


def enrich_sectors(holdings: list[Holding],
                   fundamentals_fetcher: Callable[[str], dict]) -> list[Holding]:
    """Backfill a blank ("Unknown") sector from a fundamentals lookup.

    WHY: portfolio CSVs usually leave Sector empty, so the holdings table and the sector
    allocation chart collapse into one "Unknown" bucket. yfinance already exposes
    info["sector"]; use it, but only for holdings missing one (never refetch a set sector),
    fall back to industry, and never let one failed lookup crash the page (mirrors the
    provider's degrade-to-missing contract). Pure: the fetcher is injected so tests run offline.
    """
    out: list[Holding] = []
    for h in holdings:
        if h.sector and h.sector != "Unknown":
            out.append(h)
            continue
        sector = "Unknown"
        try:
            info = fundamentals_fetcher(h.symbol) or {}
            candidate = info.get("sector") or info.get("industry")
            if candidate and str(candidate).strip():
                sector = str(candidate).strip()
        except Exception:
            sector = "Unknown"
        out.append(replace(h, sector=sector))
    return out


def _merge_lots(usable: list[tuple[Holding, float]]) -> list[tuple[Holding, float]]:
    """Consolidate multiple lots of the same symbol into one Holding (summed quantity,
    quantity-weighted average cost), so weight/concentration math treats repeat purchases of
    the same stock as ONE position, not one per lot.

    WHY (real money, HIGH severity): a stock bought in more than one lot (periodic buying, or a
    broker export that lists each purchase as its own row) is a very common real scenario. Left
    unmerged, top_holding_weight and HHI are computed per LOT, not per symbol -- live-verified:
    two Reliance lots at 37% each understated top_holding_weight as 37% instead of the true 74%,
    and HHI as 0.34 instead of the true ~0.62, silently weakening the over-concentration warning
    for exactly the user who bought more of a stock they already hold a lot of. Order of first
    appearance is preserved; a non-'Unknown' sector wins over 'Unknown' if lots disagree.
    """
    qty_by_symbol: dict[str, float] = {}
    cost_sum_by_symbol: dict[str, float] = {}
    sector_by_symbol: dict[str, str] = {}
    price_by_symbol: dict[str, float] = {}
    order: list[str] = []
    for h, p in usable:
        if h.symbol not in qty_by_symbol:
            order.append(h.symbol)
            qty_by_symbol[h.symbol] = 0.0
            cost_sum_by_symbol[h.symbol] = 0.0
            sector_by_symbol[h.symbol] = h.sector
        elif sector_by_symbol[h.symbol] == "Unknown" and h.sector != "Unknown":
            sector_by_symbol[h.symbol] = h.sector
        qty_by_symbol[h.symbol] += h.quantity
        cost_sum_by_symbol[h.symbol] += h.quantity * h.avg_cost
        price_by_symbol[h.symbol] = p
    return [
        (Holding(symbol=sym, quantity=qty_by_symbol[sym],
                avg_cost=(cost_sum_by_symbol[sym] / qty_by_symbol[sym]
                         if qty_by_symbol[sym] else 0.0),
                sector=sector_by_symbol[sym]),
         price_by_symbol[sym])
        for sym in order
    ]


def analyze_portfolio(holdings: list[Holding],
                      prices: dict[str, float]) -> PortfolioAnalysis:
    """Price the book and roll up. prices maps symbol -> current price.

    A symbol with no price (None or missing) is reported in missing_symbols and excluded
    from every total, so it cannot distort value, P&L, or weights. Multiple lots of the same
    symbol are merged into one position before weights/concentration are computed (see
    _merge_lots).
    """
    priced = [(h, prices.get(h.symbol)) for h in holdings]
    usable = _merge_lots([(h, p) for h, p in priced if p is not None])
    missing = list(dict.fromkeys(h.symbol for h, p in priced if p is None))

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


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Sensitivity of the asset to the benchmark: cov(a, b) / var(b).

    None when beta cannot be computed -- fewer than 2 overlapping return points (e.g. the benchmark
    index history failed to load while the stock's did) or a zero-variance benchmark. WHY (real
    money, "never a fabricated number"): returning 0.0 there is indistinguishable from a REAL
    market-neutral 0.00 beta and would render as one in the risk panel; None lets the caller show
    'n/a' honestly, mirroring historical_cagr's None contract for the same not-computable situation.
    """
    df = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return None
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    var_b = b.var()
    if var_b == 0:
        return None
    return float(a.cov(b) / var_b)


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough drop as a negative fraction, e.g. -0.32 = down 32%."""
    if close is None or close.empty:
        return 0.0
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(drawdown.min())


def historical_cagr(close: pd.Series) -> tuple[float, float] | None:
    """Annualized (CAGR) return from a price series' first close to its last, and the actual
    span in years the data covers. None if there isn't at least ~3 years of history or the
    series is unusable (empty, non-positive prices) -- a shorter window is not a meaningful
    long-term reference.

    WHY: used to show a REAL, live, dated benchmark return (e.g. the Sensex) next to a SIP
    calculator's assumed-return input, rather than asserting a "typical equity return" figure
    from memory. The years-of-data actually used is returned alongside the number, so the UI can
    be honest about how far back it goes rather than implying a universal truth.
    """
    if close is None or close.empty or len(close) < 2:
        return None
    first, last = close.iloc[0], close.iloc[-1]
    if first is None or last is None or first <= 0 or last <= 0:
        return None
    years = (close.index[-1] - close.index[0]).days / 365.25
    if years < 3:
        return None
    cagr_pct = ((last / first) ** (1 / years) - 1) * 100
    return float(cagr_pct), float(years)


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
