import pandas as pd
import pytest

from src.portfolio.loader import load_holdings, normalize_symbol


def test_normalize_symbol():
    assert normalize_symbol("RELIANCE.NS") == "RELIANCE"
    assert normalize_symbol("NSE:TCS") == "TCS"
    assert normalize_symbol("INFY-EQ") == "INFY"
    assert normalize_symbol(" hdfcbank ") == "HDFCBANK"
    assert normalize_symbol("500325.BO") == "500325"


def test_generic_format_with_commas_and_rupee():
    df = pd.DataFrame({
        "Symbol": ["RELIANCE", "TCS"],
        "Quantity": [10, 5],
        "Avg Cost": ["2,450.50", "₹3380"],
        "Sector": ["Energy", "IT"],
    })
    holdings = load_holdings(df)
    assert len(holdings) == 2
    assert holdings[0].symbol == "RELIANCE"
    assert holdings[0].quantity == 10
    assert holdings[0].avg_cost == 2450.50
    assert holdings[0].sector == "Energy"
    assert holdings[1].avg_cost == 3380.0


def test_zerodha_like_headers_and_default_sector():
    df = pd.DataFrame({"Instrument": ["INFY-EQ"], "Qty.": [20], "Avg. cost": [1410.25]})
    holdings = load_holdings(df)
    assert len(holdings) == 1
    assert holdings[0].symbol == "INFY"
    assert holdings[0].sector == "Unknown"


def test_skips_unparseable_and_empty_rows():
    df = pd.DataFrame({
        "Symbol": ["RELIANCE", "", "BADQTY"],
        "Quantity": [10, 5, "abc"],
        "Avg Cost": [2450, 100, 100],
    })
    holdings = load_holdings(df)
    assert [h.symbol for h in holdings] == ["RELIANCE"]


def test_missing_required_column_raises():
    df = pd.DataFrame({"Foo": [1], "Bar": [2]})
    with pytest.raises(ValueError):
        load_holdings(df)
