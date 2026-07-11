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


def test_accepts_rs_and_inr_currency_prefixes_so_rows_are_not_silently_dropped():
    # WHY (real money, data quality): Indian portfolio CSVs / manual entries commonly write costs as
    # "Rs 1,520" / "Rs.1,520" / "INR 1520", not the ₹ symbol (harder to type). _to_float stripped only
    # "₹", so an "Rs"-priced cell raised ValueError and load_holdings SILENTLY DROPPED that holding --
    # understating the parent's portfolio value, P&L and weights with no error shown. The common rupee
    # prefixes must parse like a plain number; scientific notation (1.2e3) must stay untouched.
    from src.portfolio.loader import _to_float
    assert _to_float("Rs 1,520.75") == 1520.75
    assert _to_float("Rs.1,520") == 1520.0
    assert _to_float("INR 1520") == 1520.0
    assert _to_float("1520 rs") == 1520.0
    assert _to_float("₹ 1,520.75") == 1520.75      # still handled
    assert _to_float("1,00,000") == 100000.0        # Indian digit grouping still handled
    assert _to_float("1.2e3") == 1200.0             # scientific notation NOT corrupted (no 'e' strip)
    df = pd.DataFrame({
        "Symbol": ["RELIANCE", "TCS", "HDFCBANK"],
        "Quantity": [10, 5, 8],
        "Avg Cost": ["Rs 2,000", "Rs.3,000", "INR 1500"],
    })
    got = {h.symbol: h.avg_cost for h in load_holdings(df)}
    assert got == {"RELIANCE": 2000.0, "TCS": 3000.0, "HDFCBANK": 1500.0}   # none silently dropped


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


def test_maps_google_finance_sheet_ticker_and_per_unit_cost():
    # WHY (real sheet): ticker lives in 'Stock Formula Name' (not the display 'Stock Name'), the
    # avg buy price in 'Per Unit Cost' (not the 'Investment Cost' total), and a TOTAL row trails.
    df = pd.DataFrame({
        "Stock Name": ["Adani Power Ltd.", "BLS International", "TOTAL"],
        "Stock Formula Name": ["ADANIPOWER", "BLS", ""],
        "Quantity": [1840, 1250, ""],
        "Per Unit Cost": [226, 400, ""],
        "Investment Cost": [415840, 500000, ""],
    })
    holdings = load_holdings(df)
    assert [h.symbol for h in holdings] == ["ADANIPOWER", "BLS"]   # tickers, TOTAL skipped
    assert holdings[0].avg_cost == 226.0                           # per-unit, not the total
    assert holdings[0].quantity == 1840


def test_load_holdings_from_csv_text_filelike():
    # WHY: the deployed app reads a published Google Sheet CSV link as text -> StringIO.
    import io
    text = "Symbol,Quantity,Avg Cost,Sector\nRELIANCE,10,2400,Energy\nSBIN,5,600,\n"
    holdings = load_holdings(io.StringIO(text))
    assert [h.symbol for h in holdings] == ["RELIANCE", "SBIN"]
    assert holdings[0].sector == "Energy"
    assert holdings[1].sector == "Unknown"
