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
- Owner to provide: source list (fill config/sources.yaml), and ONE LLM option in .env
  (NVIDIA NIM free key, or local Ollama). No paid key required.
- Carryover from v1: market-wide screener; Upstox/Kite live-data adapter; portfolio-level drawdown.

## 2026-06-18 (session 5, /loop) — mutual funds & SIPs in the UI
- Reviewed + hardened session-4 ingestion/render (corrupt-PDF degrade, fail-safe render,
  fingerprint-keyed source cache) -- all committed.
- Added src/sip.py: pure SIP future-value math (annuity-due), framed as arithmetic on an
  assumption, NOT a prediction. Tested (zero-rate, invested=monthly*months, growth bound).
- Added "Mutual funds & SIPs" section to app.py: live AMFI NAV lookup by fund name (gated
  behind a search so page load needs no network), and a SIP projection helper with an
  explicit "not a prediction, returns can be negative" caveat.
- VERIFIED: 48 tests green (+4 SIP). AppTest default load clean (no network), SIP metrics
  render (₹12L invested on 10k/10y). Live AppTest: fund search "bluechip" -> 15 real schemes
  with live NAVs through the UI, no exception.
- Next logical step: a SIP/MF "what this means" plain-English helper + ingest fund factsheets
  into the grounded library so the mentor can answer fund questions with citations. IPO path
  still blocked on an owner-provided data source.

## 2026-06-18 (session 4, /loop) — document ingestion + parent research surface
- Built src/research/library.py: ingests txt/md/pdf from a documents dir into a registry-
  bound DocumentStore, matching filename stem -> source id; unregistered/untiered files
  skipped and reported (never ingested). Added DocumentStore.source_ids(). pypdf for PDFs.
- Added the "Ask the research mentor" section to app.py: loads the source library (real
  config/sources.yaml + documents/, else bundled sample_data), shows loaded sources by tier,
  answers a typed question with cited verified facts / opinion / unverified badges, or a
  clear "No verified answer" abstention.
- Sample library bundled (sample_data/sources.yaml + documents/*, synthetic, labeled).
- VERIFIED: 43 tests green (+4 ingestion). AppTest with no LLM -> research surface renders,
  degrades cleanly. LIVE AppTest with ollama_chat/qwen2.5:7b -> typed question returned a
  green VERIFIED FACT cited to its source through the UI, no exception. This is G8's first
  working version (parent-facing grounded Q&A). Remaining G8 polish: readability pass, MF/SIP
  and per-holding entry points into the same surface.

## 2026-06-18 (session 3) — provider-agnostic LLM
- Owner directed: no paid Anthropic key; use free/open models. Built `src/llm/client.py`
  (LLMClient + LiteLLMClient); analysts now take an injected client. Default config via
  LLM_MODEL env -> NVIDIA NIM free / Ollama local / any LiteLLM provider. Removed the hard
  Anthropic dependency.
- Verified offline: 39 tests green incl. full grounded path with a FakeClient; app AppTest
  clean, shows the LLM-off hint, no Anthropic leftovers.
- LIVE VERIFIED (2026-06-18): installed Ollama (brew), ran scripts/live_smoke.py against a
  real local model (qwen2.5:7b via ollama_chat/, no key, data local). Answerable question ->
  verified fact cited to its primary source; figure-not-in-source -> abstained, no fabricated
  number; unrelated question -> abstained. The free-open-model + grounding spine works e2e.
- Ollama models available locally now: qwen2.5:7b, llama3.1. Recommended LLM_MODEL prefix is
  `ollama_chat/` (sends the system prompt). Server: `ollama serve` (or `brew services start ollama`).
