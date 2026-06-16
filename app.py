"""India Equity Research - Streamlit app.

Upload a portfolio CSV, get grounded analysis and research. Research-only: no trades,
no buy/sell calls. Every figure is fetched and timestamped; the LLM only summarizes
fetched facts.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# WHY: ensure `src` imports work regardless of the cwd Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.constants import (  # noqa: E402
    CONCENTRATION_HHI_WARN,
    CONCENTRATION_TOP_HOLDING_WARN,
    CURRENCY_SYMBOL,
    DEFAULT_BENCHMARK,
    DISCLAIMER,
    INDEX_DISPLAY_NAMES,
    NIFTY50_SYMBOL,
    SENSEX_SYMBOL,
)
from src.data.yfinance_provider import YFinanceProvider  # noqa: E402
from src.portfolio.analysis import (  # noqa: E402
    analyze_portfolio,
    annualized_volatility,
    beta,
    daily_returns,
    max_drawdown,
    portfolio_daily_returns,
)
from src.portfolio.loader import load_holdings  # noqa: E402
from src.research.analyst import ResearchAnalyst  # noqa: E402

load_dotenv()

st.set_page_config(page_title="India Equity Research", layout="wide", page_icon="📊")

SAMPLE_CSV = Path(__file__).resolve().parent / "sample_data" / "sample_portfolio.csv"


# --- cached data access (the provider is the only network boundary) ---

@st.cache_resource
def get_provider() -> YFinanceProvider:
    return YFinanceProvider()


@st.cache_resource
def get_analyst() -> ResearchAnalyst:
    return ResearchAnalyst()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(symbols: tuple[str, ...]) -> dict[str, float | None]:
    return get_provider().current_prices(list(symbols))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_index(symbol: str) -> dict:
    return get_provider().index_quote(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_close(symbol: str) -> pd.Series:
    hist = get_provider().history(symbol, period="1y")
    if "Close" in hist:
        return hist["Close"].dropna()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    return get_provider().fundamentals(symbol)


def money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{CURRENCY_SYMBOL}{value:,.0f}"


# --- header ---

st.title("📊 India Equity Research")
st.caption("Research and analysis for your NSE/BSE portfolio. Decision support, not advice.")
st.warning(DISCLAIMER, icon="⚠️")

# --- sidebar: input + status ---

with st.sidebar:
    st.header("Portfolio")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    use_sample = st.checkbox("Use sample portfolio", value=not uploaded)
    st.caption("Columns matched loosely: Symbol, Quantity, Avg Cost, (optional) Sector. "
               "Zerodha/Groww exports work too.")

    st.divider()
    analyst = get_analyst()
    if analyst.available:
        st.success(f"AI research: on ({analyst.model})")
    else:
        st.info("AI research: off. Set ANTHROPIC_API_KEY in .env to enable research notes. "
                "Analysis below works without it.")

# --- resolve the data source ---

source = None
if uploaded is not None:
    source = uploaded
elif use_sample:
    source = SAMPLE_CSV

if source is None:
    st.info("Upload a portfolio CSV or tick 'Use sample portfolio' in the sidebar to begin.")
    st.stop()

try:
    holdings = load_holdings(source)
except Exception as exc:
    st.error(f"Could not read that CSV: {exc}")
    st.stop()

if not holdings:
    st.error("No valid holdings found in that file.")
    st.stop()

symbols = tuple(h.symbol for h in holdings)
as_of = datetime.now().strftime("%Y-%m-%d %H:%M")

# --- market context ---

st.subheader("Market context")
ctx_cols = st.columns(2)
for col, idx_symbol in zip(ctx_cols, (NIFTY50_SYMBOL, SENSEX_SYMBOL)):
    quote = fetch_index(idx_symbol)
    name = INDEX_DISPLAY_NAMES.get(idx_symbol, idx_symbol)
    price = quote.get("price")
    change = quote.get("change_pct")
    col.metric(
        name,
        f"{price:,.2f}" if price is not None else "n/a",
        f"{change:+.2f}%" if change is not None else None,
    )

# --- price + analyze ---

with st.spinner("Fetching live prices..."):
    prices = fetch_prices(symbols)
analysis = analyze_portfolio(holdings, prices)

st.subheader("Portfolio summary")
st.caption(f"Prices as of {as_of}. Source: yfinance / Yahoo Finance.")

m = st.columns(4)
m[0].metric("Invested", money(analysis.total_invested))
m[1].metric("Market value", money(analysis.total_value))
m[2].metric("Unrealized P&L", money(analysis.total_pnl_abs),
            f"{analysis.total_pnl_pct:+.2f}%")
m[3].metric("Holdings priced", f"{len(analysis.positions)} / {len(holdings)}")

if analysis.missing_symbols:
    st.warning(
        "No price found for: " + ", ".join(analysis.missing_symbols)
        + ". These are excluded from every total above. Check the symbol spelling "
          "or exchange (NSE assumed; numeric codes treated as BSE)."
    )

# --- holdings table ---

rows = [{
    "Symbol": p.symbol,
    "Sector": p.sector,
    "Qty": p.quantity,
    "Avg cost": round(p.avg_cost, 2),
    "Price": round(p.current_price, 2),
    "Invested": round(p.invested, 2),
    "Value": round(p.market_value, 2),
    "P&L": round(p.pnl_abs, 2),
    "P&L %": round(p.pnl_pct, 2),
    "Weight %": round(p.weight * 100, 2),
} for p in sorted(analysis.positions, key=lambda x: -x.market_value)]
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --- allocation charts ---

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Allocation by holding**")
    alloc_df = pd.DataFrame({
        "Symbol": [p.symbol for p in analysis.positions],
        "Weight": [p.weight * 100 for p in analysis.positions],
    })
    st.plotly_chart(px.pie(alloc_df, names="Symbol", values="Weight", hole=0.4),
                    width="stretch")
with c2:
    st.markdown("**Allocation by sector**")
    sector_df = pd.DataFrame({
        "Sector": list(analysis.sector_weights.keys()),
        "Weight %": [w * 100 for w in analysis.sector_weights.values()],
    }).sort_values("Weight %", ascending=False)
    st.plotly_chart(px.bar(sector_df, x="Sector", y="Weight %"),
                    width="stretch")

# --- concentration flags (advisory display, not advice) ---

st.subheader("Concentration")
cc = st.columns(3)
cc[0].metric("Largest holding", f"{analysis.top_holding_weight * 100:.1f}%")
cc[1].metric("HHI", f"{analysis.hhi:.3f}")
cc[2].metric("Effective # holdings", f"{analysis.effective_holdings:.1f}")
flags = []
if analysis.top_holding_weight > CONCENTRATION_TOP_HOLDING_WARN:
    flags.append(f"One name is over {CONCENTRATION_TOP_HOLDING_WARN * 100:.0f}% of the book.")
if analysis.hhi > CONCENTRATION_HHI_WARN:
    flags.append("HHI reads as concentrated (few names drive the book).")
if flags:
    st.warning(" ".join(flags) + " This is an observation about structure, not advice.")
else:
    st.success("No single-name or HHI concentration flag triggered.")

# --- risk (gated: fetches 1y history per holding + benchmark) ---

st.subheader("Risk (1-year history)")
if st.button("Compute risk metrics"):
    with st.spinner("Fetching 1-year history..."):
        bench_close = fetch_history_close(DEFAULT_BENCHMARK)
        bench_returns = daily_returns(bench_close)
        close_by_symbol = {p.symbol: fetch_history_close(p.symbol)
                           for p in analysis.positions}
        weights = {p.symbol: p.weight for p in analysis.positions}
        port_returns = portfolio_daily_returns(close_by_symbol, weights)

    risk_cols = st.columns(3)
    risk_cols[0].metric("Annualized volatility",
                        f"{annualized_volatility(port_returns) * 100:.1f}%")
    risk_cols[1].metric(f"Beta vs {INDEX_DISPLAY_NAMES[DEFAULT_BENCHMARK]}",
                        f"{beta(port_returns, bench_returns):.2f}")
    worst = min((max_drawdown(c) for c in close_by_symbol.values() if not c.empty),
                default=0.0)
    risk_cols[2].metric("Worst single-name drawdown", f"{worst * 100:.1f}%")
    st.caption("Volatility and beta are for the book at current weights. Drawdown is the "
               "worst peak-to-trough of any single holding over the last year.")

# --- AI research (gated per call; grounded in fetched facts) ---

st.subheader("AI research notes")
analyst = get_analyst()
if not analyst.available:
    st.info("Set ANTHROPIC_API_KEY in your .env to generate research notes.")
else:
    if "notes" not in st.session_state:
        st.session_state.notes = {}

    if st.button("Generate portfolio overview"):
        with st.spinner("Writing overview..."):
            st.session_state.notes["__overview__"] = analyst.portfolio_overview(analysis)
    if "__overview__" in st.session_state.notes:
        st.markdown(st.session_state.notes["__overview__"])

    st.markdown("**Per-holding notes**")
    for p in sorted(analysis.positions, key=lambda x: -x.market_value):
        with st.expander(f"{p.symbol} — {p.weight * 100:.1f}% of book"):
            if st.button("Generate note", key=f"note_{p.symbol}"):
                with st.spinner(f"Researching {p.symbol}..."):
                    fundamentals = fetch_fundamentals(p.symbol)
                    st.session_state.notes[p.symbol] = analyst.research_note(p, fundamentals)
            if p.symbol in st.session_state.notes:
                st.markdown(st.session_state.notes[p.symbol])

st.divider()
st.caption(DISCLAIMER)
