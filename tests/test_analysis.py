import numpy as np
import pandas as pd

from src.portfolio.analysis import (
    analyze_portfolio,
    annualized_volatility,
    beta,
    max_drawdown,
    portfolio_daily_returns,
)
from src.portfolio.models import Holding


def test_analyze_basic_value_pnl_weights():
    holdings = [Holding("A", 10, 100.0, "Tech"), Holding("B", 5, 200.0, "Bank")]
    prices = {"A": 150.0, "B": 200.0}
    a = analyze_portfolio(holdings, prices)

    assert a.total_invested == 2000.0
    assert a.total_value == 2500.0
    assert a.total_pnl_abs == 500.0
    assert abs(a.total_pnl_pct - 25.0) < 1e-9

    weight_a = next(p.weight for p in a.positions if p.symbol == "A")
    assert abs(weight_a - 0.6) < 1e-9
    assert abs(sum(p.weight for p in a.positions) - 1.0) < 1e-9


def test_concentration_metrics():
    holdings = [Holding("A", 10, 100.0, "Tech"), Holding("B", 5, 200.0, "Bank")]
    prices = {"A": 150.0, "B": 200.0}  # weights 0.6 / 0.4
    a = analyze_portfolio(holdings, prices)
    assert abs(a.hhi - 0.52) < 1e-9          # 0.6^2 + 0.4^2
    assert abs(a.effective_holdings - 1 / 0.52) < 1e-9
    assert abs(a.top_holding_weight - 0.6) < 1e-9


def test_sector_weights_aggregate():
    holdings = [Holding("A", 10, 100, "Tech"), Holding("B", 10, 100, "Tech"),
                Holding("C", 10, 100, "Bank")]
    prices = {"A": 100, "B": 100, "C": 100}
    a = analyze_portfolio(holdings, prices)
    assert abs(a.sector_weights["Tech"] - 2 / 3) < 1e-9
    assert abs(a.sector_weights["Bank"] - 1 / 3) < 1e-9


def test_missing_price_is_excluded_not_dropped_silently():
    holdings = [Holding("A", 10, 100, "Tech"), Holding("B", 10, 100, "Bank")]
    prices = {"A": 120.0}  # B has no price
    a = analyze_portfolio(holdings, prices)
    assert a.missing_symbols == ["B"]
    assert a.total_value == 1200.0
    assert len(a.positions) == 1


def test_zero_cost_lot_does_not_divide_by_zero():
    holdings = [Holding("A", 10, 0.0, "Tech")]
    prices = {"A": 50.0}
    a = analyze_portfolio(holdings, prices)
    assert a.positions[0].pnl_pct == 0.0
    assert a.total_pnl_pct == 0.0


def test_max_drawdown():
    close = pd.Series([100, 120, 90, 95, 60, 100])  # peak 120 -> trough 60
    assert abs(max_drawdown(close) - (-0.5)) < 1e-9


def test_beta_of_2x_series_is_2():
    idx = pd.Series(np.random.RandomState(0).normal(0, 0.01, 200))
    asset = 2 * idx
    assert abs(beta(asset, idx) - 2.0) < 1e-6


def test_volatility_and_empty_guards():
    assert annualized_volatility(pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])) > 0
    assert annualized_volatility(pd.Series(dtype=float)) == 0.0
    assert max_drawdown(pd.Series(dtype=float)) == 0.0
    assert beta(pd.Series(dtype=float), pd.Series(dtype=float)) == 0.0


def test_portfolio_daily_returns_equal_weight_cancel():
    close_a = pd.Series([100, 110, 121])  # +10%, +10%
    close_b = pd.Series([100, 90, 81])    # -10%, -10%
    pr = portfolio_daily_returns({"A": close_a, "B": close_b}, {"A": 0.5, "B": 0.5})
    assert all(abs(x) < 1e-9 for x in pr)
