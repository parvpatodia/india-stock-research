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

## 2026-06-18 (session 2, BUILD) — v2 pivot: parent-facing research mentor
- Owner reframed: build for his PARENTS to use with real money. Confirmed advisory level =
  "research mentor, they decide" (no recommendations/guarantees), influencers = context-only
  attributed, language = English, instruments = stocks + MF/SIP + IPO + other.
- Built the v2 SAFETY SPINE (SPEC v2, features G1-G7): credibility-tiered SourceRegistry,
  offline document grounding + abstention, Claim/Citation contract that downgrades any
  unsourced "fact", grounded analyst that never trusts model output as-is, AMFI MF NAV
  provider (Tier-1, free), instrument taxonomy, sources.yaml template.
- 34 tests green. Adversarial review of the spine found 7 issues (2 high: mixed-tier fact,
  bad NAV); all fixed with regression tests. See LESSONS.md.

## Improvement metrics (session 2)
- parv_corrections: 0 | repeat_mistakes: 0 (different bug class than session 1)
- bugs_found: 7 (all by adversarial review) | shipped_first_try: false | rework_commits: 1

## Next (sequenced)
- G8 parent-facing UI: plain-English research view over the engine (English, large/readable),
  wired to upload + the existing stock dashboard. THE next user-visible step.
- Wire grounded analyst into the app: ingest owner-supplied documents into DocumentStore
  under registered sources; show cited claims + abstentions; live smoke with a real key.
- G9 IPO data adapter + DRHP/RHP analysis (needs a source/feed from owner).
- MF/SIP views over AMFIProvider (NAV history, SIP return math).
- Owner to provide: source list (fill config/sources.yaml), ANTHROPIC_API_KEY for the writer.
- Carryover from v1: market-wide screener; Upstox/Kite live-data adapter; portfolio-level drawdown.
