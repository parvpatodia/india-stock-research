# india-stock-research

MODE: BUILD

AI-assisted research and analysis for Indian equities (NSE/BSE).

This is **research-only decision support**. The system never places trades and never
issues buy/sell calls. Every number it shows is traceable to a fetched source, freshness
is timestamped, and the human stays in the loop on anything that moves money. This rule is
load-bearing because real capital is involved: an LLM that states a wrong figure with
confidence is the main failure mode the design guards against.

- Architecture, scope, and feature checklist: see `SPEC.md`.
- Build lessons (read at session start): see `LESSONS.md`.
- Progress / handoff: see `PROGRESS.md`.

Data layer sits behind `MarketDataProvider` (src/data/provider.py). v1 uses yfinance;
swap to a broker API by adding one adapter, no analysis code changes.
