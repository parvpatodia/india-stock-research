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


def test_nan_quantity_row_skipped_not_poisoning_totals():
    # WHY (regression): float('nan') parses but slips past `<= 0`, poisoning every total.
    df = pd.DataFrame({"Symbol": ["A", "B"], "Quantity": [10, "nan"], "Avg Cost": [100, 200]})
    holdings = load_holdings(df)
    assert [h.symbol for h in holdings] == ["A"]


def test_negative_cost_row_skipped():
    df = pd.DataFrame({"Symbol": ["A", "B"], "Quantity": [10, 5], "Avg Cost": [100, -50]})
    holdings = load_holdings(df)
    assert [h.symbol for h in holdings] == ["A"]


def test_avg_cost_column_wins_over_total_cost_column():
    # WHY (regression): set-ordered matching could pick total "Cost" as per-share avg cost.
    df = pd.DataFrame({"Symbol": ["A"], "Quantity": [10], "Avg Cost": [100], "Cost": [1000]})
    holdings = load_holdings(df)
    assert holdings[0].avg_cost == 100.0


def test_symbol_column_wins_over_generic_name_column():
    df = pd.DataFrame({"Symbol": ["RELIANCE"], "Name": ["WRONG"],
                       "Quantity": [10], "Avg Cost": [100]})
    holdings = load_holdings(df)
    assert holdings[0].symbol == "RELIANCE"


def test_load_holdings_from_csv_text_filelike():
    # WHY: the deployed app reads a published Google Sheet CSV link as text -> StringIO.
    import io
    text = "Symbol,Quantity,Avg Cost,Sector\nRELIANCE,10,2400,Energy\nSBIN,5,600,\n"
    holdings = load_holdings(io.StringIO(text))
    assert [h.symbol for h in holdings] == ["RELIANCE", "SBIN"]
    assert holdings[0].sector == "Energy"
    assert holdings[1].sector == "Unknown"
