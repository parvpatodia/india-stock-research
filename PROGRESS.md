# PROGRESS

## 2026-06-15 (session 1, BUILD)
- Scoped to research-only decision support (no trades, no recommendations). See SPEC.md.
- Data source decision: yfinance v1 behind MarketDataProvider; Upstox/Kite later.
- Building feature by feature: F1 loader -> F2/F3/F4 analysis -> F5 provider -> F6 research -> F7 app.

## 2026-06-15 (session 1 result)
- v1 SHIPPED + verified. F1-F7 all met. 18 unit tests green; streamlit AppTest runs the
  full app end-to-end with live data, sample 7/7 priced, no exception.
- Adversarial review found 4 real correctness bugs (nan qty poisoning totals, set-ordered
  column matching, stale "as of" timestamp on cache hit, negative cost). All fixed with
  regression tests. See LESSONS.md.
- Demo path: `./.venv/bin/streamlit run app.py`, tick "Use sample portfolio".

## Improvement metrics (session 1, baseline)
- parv_corrections: 0 (not reviewed by Parv yet) | repeat_mistakes: 0
- bugs_found: 5 (1 dep miss + 4 review findings) | shipped_first_try: false | rework_commits: 1

## Next (v2 candidates, not started)
- Market-wide screener (the full NSE/BSE universe), per SPEC out-of-scope-for-v1.
- Broker-API provider adapter (Upstox free / Zerodha Kite) for live data + holdings import.
- News ingestion to feed dated headlines into research notes.
- Portfolio-level max drawdown (currently worst single-name only).
