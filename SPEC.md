# SPEC: india-stock-research (v1)

## Purpose
A personal tool that ingests a user's Indian equity portfolio and produces grounded
research and analysis to support the user's own decisions. It does not decide, recommend,
or trade.

## Users
One user (the owner), running locally or on their own Streamlit Cloud instance, analyzing
their real holdings.

## Non-goals (hard boundaries)
- No trade execution. No broker order placement.
- No buy / sell / hold recommendations. No price targets. No "should I" answers.
- No promise of returns. No backtested "strategy" sold as predictive.
- Not a registered investment advisory service. Personal research use only.
- v1 does not screen the full ~5000-stock universe. That is v2.

## Real-money guardrails (the reason the design exists)
1. Every figure shown comes from fetched data, with a visible "as of" timestamp.
2. The LLM research layer only summarizes data passed to it. It is instructed never to
   introduce numbers from memory. Prompts pin the model to the supplied facts.
3. Missing or stale data is shown as missing, never silently dropped or guessed.
4. A standing disclaimer is visible in the app and every export.
5. The analysis math (P&L, weights, concentration) is covered by unit tests, because a
   silent arithmetic bug here costs real money.

## Data sources
- v1 backbone: **yfinance** with NSE (`.NS`) / BSE (`.BO`) suffixes. Free, no key.
  Gives current price, 1y history, and a fundamentals subset for Indian listings.
- Portfolio: uploaded CSV. Supports Zerodha/Groww-style exports and a generic format.
- Upgrade path (no analysis changes): add a `MarketDataProvider` adapter for Upstox
  (free API, live data + holdings) or Zerodha Kite (paid, best quality).

## Architecture
```
app.py                      Streamlit UI (upload -> dashboard -> research)
src/
  constants.py              all domain constants in one place
  portfolio/
    models.py               Holding, PositionAnalysis, PortfolioAnalysis (dataclasses)
    loader.py               CSV parse + normalize (Zerodha/Groww/generic) -> [Holding]
    analysis.py             pure functions: value, P&L, weights, concentration, risk
  data/
    provider.py             MarketDataProvider interface (ABC)
    yfinance_provider.py    yfinance adapter
  research/
    analyst.py              LLM research notes, grounded + sourced, graceful degrade
tests/
  test_loader.py            CSV normalization
  test_analysis.py          portfolio math (no network; data injected)
```
SOLID: analysis depends on injected price/history data, not on the provider. The provider
is the only network boundary. Swapping data sources touches one file.

## Feature checklist (each gated by a runnable check)
- [ ] F1 loader: parse Zerodha/Groww/generic CSV into normalized Holdings. (test_loader)
- [ ] F2 analysis: value, invested, P&L abs/pct, weights, totals. (test_analysis)
- [ ] F3 analysis: concentration (HHI, effective N, top-holding %), sector exposure. (test_analysis)
- [ ] F4 risk: annualized volatility, beta vs NIFTY, max drawdown from history. (test_analysis)
- [ ] F5 provider: yfinance adapter returns price/fundamentals/history. (smoke, network)
- [ ] F6 research: grounded per-holding notes; degrades to "set key" without API key. (smoke)
- [ ] F7 app: upload CSV -> dashboard (tables + charts) + research + disclaimer. (run + screenshot)

## Success criteria (v1 done)
- `pytest` green on F1-F4.
- App launches, accepts the sample CSV, renders the dashboard with live prices.
- Research notes render when a key is set; a clear prompt shows when it is not.
- README documents the demo path. No secrets committed.
