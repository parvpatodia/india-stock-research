import numpy as np
import pandas as pd

from src.portfolio.analysis import (
    analyze_portfolio,
    annualized_volatility,
    beta,
    enrich_sectors,
    historical_cagr,
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


def test_multiple_lots_of_the_same_stock_merge_into_one_position():
    # WHY (real money, HIGH severity): a stock bought in more than one lot (a very common real
    # scenario -- periodic/SIP-style buying, or a broker export that lists each purchase as its
    # own row) must be treated as ONE position for concentration purposes. Before this fix, two
    # RELIANCE lots (10 @ 2000, 10 @ 2400) each showed as their OWN 37% position instead of one
    # combined 74% position -- top_holding_weight reported 37% (the largest SINGLE lot) instead
    # of the true 74%, so the Concentration tab's over-concentration warning could silently fail
    # to trigger for the exact user this feature exists to protect: someone who bought more of a
    # stock they already hold a lot of.
    holdings = [
        Holding("RELIANCE", 10, 2000.0, "Energy"),
        Holding("RELIANCE", 10, 2400.0, "Energy"),
        Holding("TCS", 5, 3000.0, "IT"),
    ]
    prices = {"RELIANCE": 2500.0, "TCS": 3500.0}
    a = analyze_portfolio(holdings, prices)

    assert len(a.positions) == 2                      # merged into one RELIANCE position
    reliance = next(p for p in a.positions if p.symbol == "RELIANCE")
    assert reliance.quantity == 20
    assert abs(reliance.avg_cost - 2200.0) < 1e-9      # quantity-weighted average cost
    assert abs(reliance.weight - (20 * 2500.0) / a.total_value) < 1e-9
    assert abs(a.top_holding_weight - reliance.weight) < 1e-9  # the TRUE merged weight, not 37%


def test_multiple_lots_weighted_average_cost_with_uneven_quantities():
    holdings = [Holding("A", 10, 100.0, "Tech"), Holding("A", 30, 200.0, "Tech")]
    prices = {"A": 250.0}
    a = analyze_portfolio(holdings, prices)
    assert len(a.positions) == 1
    pos = a.positions[0]
    assert pos.quantity == 40
    # WHY: weighted by quantity, not a plain average of the two costs (would wrongly read 150).
    assert abs(pos.avg_cost - ((10 * 100.0 + 30 * 200.0) / 40)) < 1e-9
    assert abs(pos.invested - (10 * 100.0 + 30 * 200.0)) < 1e-9


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


def test_missing_price_symbol_listed_once_even_with_multiple_unpriced_lots():
    holdings = [Holding("A", 10, 100, "Tech"), Holding("B", 5, 100, "Bank"),
                Holding("B", 5, 110, "Bank")]
    prices = {"A": 120.0}  # B has no price, in either lot
    a = analyze_portfolio(holdings, prices)
    assert a.missing_symbols == ["B"]                  # not ["B", "B"]


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


def test_historical_cagr_doubles_over_ten_years_is_about_seven_pct():
    idx = pd.date_range("2016-01-01", "2026-01-01", freq="365D")
    close = pd.Series([100.0, 200.0], index=[idx[0], idx[-1]])
    result = historical_cagr(close)
    assert result is not None
    cagr_pct, years = result
    assert abs(years - 10.0) < 0.1
    assert abs(cagr_pct - 7.18) < 0.1                # 2x over 10y ~ 7.18%/yr


def test_historical_cagr_none_when_span_too_short():
    # WHY: a <3-year window is not a meaningful "long-term" reference point.
    idx = pd.date_range("2024-01-01", "2026-01-01", freq="365D")
    close = pd.Series([100.0, 110.0], index=[idx[0], idx[-1]])
    assert historical_cagr(close) is None


def test_historical_cagr_none_on_empty_or_bad_data():
    assert historical_cagr(pd.Series(dtype=float)) is None
    assert historical_cagr(pd.Series([100.0])) is None                   # single point
    idx = pd.date_range("2016-01-01", "2026-01-01", freq="365D")
    assert historical_cagr(pd.Series([0.0, 100.0], index=[idx[0], idx[-1]])) is None  # bad first
    assert beta(pd.Series(dtype=float), pd.Series(dtype=float)) == 0.0


def test_portfolio_daily_returns_equal_weight_cancel():
    close_a = pd.Series([100, 110, 121])  # +10%, +10%
    close_b = pd.Series([100, 90, 81])    # -10%, -10%
    pr = portfolio_daily_returns({"A": close_a, "B": close_b}, {"A": 0.5, "B": 0.5})
    assert all(abs(x) < 1e-9 for x in pr)


# --- sector backfill ---

def test_enrich_sectors_backfills_blank_and_keeps_existing():
    holdings = [Holding("A", 10, 100.0), Holding("B", 5, 200.0, "Energy")]
    calls: list[str] = []

    def fetcher(sym: str) -> dict:
        calls.append(sym)
        return {"A": {"sector": "Financial Services", "industry": "Banks"}}.get(sym, {})

    out = enrich_sectors(holdings, fetcher)
    assert out[0].sector == "Financial Services"   # blank ("Unknown") backfilled
    assert out[1].sector == "Energy"               # existing kept
    assert calls == ["A"]                           # not refetched for the one already set


def test_enrich_sectors_falls_back_to_industry_then_unknown():
    holdings = [Holding("A", 10, 100.0), Holding("B", 5, 200.0)]

    def fetcher(sym: str) -> dict:
        return {
            "A": {"sector": None, "industry": "Auto Components"},
            "B": {"sector": None, "industry": None},
        }[sym]

    out = enrich_sectors(holdings, fetcher)
    assert out[0].sector == "Auto Components"       # sector missing -> industry
    assert out[1].sector == "Unknown"               # neither known -> stays Unknown


def test_enrich_sectors_survives_fetcher_error():
    # WHY: one bad ticker must never crash the page (provider degrade-to-missing contract).
    def fetcher(sym: str) -> dict:
        raise RuntimeError("network down")

    out = enrich_sectors([Holding("A", 10, 100.0)], fetcher)
    assert out[0].sector == "Unknown"
