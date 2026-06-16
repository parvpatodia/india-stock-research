# India Equity Research

Research and analysis for your Indian (NSE/BSE) stock portfolio. Upload your holdings, get
live valuation, P&L, concentration and sector breakdown, risk metrics, and grounded
research notes.

**This is decision support, not advice.** It never places trades and never tells you what
to buy or sell. Every figure is fetched from a data source and timestamped. The AI research
layer only summarizes the fetched facts. It is told never to introduce a number from memory
and never to make a recommendation. Verify every figure before you act on it.

## What it does (v1)

- **Portfolio analysis**: current value, invested, unrealized P&L per holding and total,
  position weights.
- **Concentration**: largest-holding weight, Herfindahl index (HHI), effective number of
  holdings, with advisory flags.
- **Sector breakdown**: allocation by sector.
- **Risk** (1-year history): annualized volatility and beta of the book vs NIFTY 50, plus
  the worst single-name drawdown.
- **Market context**: live NIFTY 50 and SENSEX levels.
- **AI research notes**: a portfolio overview and per-holding notes, each grounded in the
  fetched data and sourced. Requires an Anthropic API key; everything else works without one.

Not in v1: screening the full market universe, trade execution, recommendations.

## Data source

v1 uses **yfinance** (free, no key) with NSE (`.NS`) and BSE (`.BO`) suffixes. A symbol
Yahoo cannot price is excluded from all totals and flagged, never guessed. The data layer
sits behind `MarketDataProvider` (`src/data/provider.py`), so moving to a broker feed
(Upstox free API, or Zerodha Kite) later is one new adapter and no change to the analysis.

## Setup

```bash
cd india-stock-research
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Optional, for AI research notes only:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...
```

## Run

```bash
./.venv/bin/streamlit run app.py
```

Then open the URL it prints (default http://localhost:8501). Tick "Use sample portfolio"
to try it immediately, or upload your own CSV.

## Portfolio CSV format

Columns are matched loosely, so Zerodha and Groww holdings exports work as-is. Minimum:

```csv
Symbol,Quantity,Avg Cost,Sector
RELIANCE,15,1180.00,Energy
TCS,10,2350.00,IT
```

`Sector` is optional. Symbols may be bare (`RELIANCE`), suffixed (`RELIANCE.NS`),
prefixed (`NSE:RELIANCE`), or carry the equity series tag (`RELIANCE-EQ`). A purely numeric
symbol is treated as a BSE scrip code.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

Covers the loader (CSV normalization) and the analysis math (P&L, weights, concentration,
risk), which is the part where a silent bug would cost real money.

## Layout

```
app.py                      Streamlit UI
src/constants.py            domain constants (one source of truth)
src/portfolio/              models, CSV loader, analysis math (pure, tested)
src/data/                   MarketDataProvider interface + yfinance adapter
src/research/               LLM research notes (grounded, graceful no-key degrade)
tests/                      loader + analysis unit tests
sample_data/                sample_portfolio.csv
```
