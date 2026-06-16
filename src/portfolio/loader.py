"""Load and normalize a portfolio CSV into [Holding].

Handles Zerodha/Groww-style exports and a generic Symbol/Quantity/Avg Cost format by
matching column names loosely. No network. Raises a clear error if required columns are
missing, because guessing a column wrong would corrupt every downstream number.
"""
from __future__ import annotations

import pandas as pd

from .models import Holding

# Candidate header names per role, matched case-insensitively after stripping.
_SYMBOL_HEADERS = {"symbol", "instrument", "stock name", "stock", "ticker", "scrip",
                   "scrip name", "name", "company"}
_QTY_HEADERS = {"quantity", "qty", "qty.", "shares", "units", "holding qty", "net qty"}
_COST_HEADERS = {"avg cost", "avg. cost", "average cost", "average buy price",
                 "avg buy price", "buy avg", "avg price", "average price", "cost"}
_SECTOR_HEADERS = {"sector", "industry"}


def normalize_symbol(raw: object) -> str:
    """RELIANCE.NS -> RELIANCE, NSE:RELIANCE -> RELIANCE, INFY-EQ -> INFY."""
    s = str(raw).strip().upper()
    if ":" in s:                      # "NSE:RELIANCE"
        s = s.split(":")[-1]
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    if s.endswith("-EQ"):             # NSE equity series tag
        s = s[: -len("-EQ")]
    return s.strip()


def _to_float(value: object) -> float:
    """Parse '1,520.75' or '₹ 1,520.75' to float."""
    s = str(value).replace(",", "").replace("₹", "").strip()
    return float(s)


def _resolve_columns(columns) -> dict[str, str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for role, candidates in (
        ("symbol", _SYMBOL_HEADERS),
        ("quantity", _QTY_HEADERS),
        ("avg_cost", _COST_HEADERS),
        ("sector", _SECTOR_HEADERS),
    ):
        for cand in candidates:
            if cand in lookup:
                resolved[role] = lookup[cand]
                break
    missing = [r for r in ("symbol", "quantity", "avg_cost") if r not in resolved]
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in CSV headers {list(columns)}. "
            "Expected something like Symbol, Quantity, Avg Cost."
        )
    return resolved


def _read(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source)


def load_holdings(source) -> list[Holding]:
    """source: a path, a file-like object (e.g. Streamlit upload), or a DataFrame."""
    df = _read(source)
    col = _resolve_columns(df.columns)
    holdings: list[Holding] = []
    for _, row in df.iterrows():
        symbol = normalize_symbol(row[col["symbol"]])
        if not symbol or symbol == "NAN":
            continue
        try:
            quantity = _to_float(row[col["quantity"]])
            avg_cost = _to_float(row[col["avg_cost"]])
        except (ValueError, TypeError):
            continue  # skip unparseable rows rather than poison the totals
        if quantity <= 0:
            continue
        sector = "Unknown"
        if "sector" in col:
            raw_sector = str(row[col["sector"]]).strip()
            if raw_sector and raw_sector.lower() != "nan":
                sector = raw_sector
        holdings.append(Holding(symbol=symbol, quantity=quantity,
                                avg_cost=avg_cost, sector=sector))
    return holdings
