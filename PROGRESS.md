# PROGRESS

## 2026-07-27 — Professional UI overhaul (top-tier investing-app look, all tabs)
Raised the whole deployed app to a cohesive, premium design language in light + dark + mobile,
without changing any behaviour (research-only, every feature intact; every custom render
try/except-guarded with a fallback to the native widget). Shipped as gated increments, each
adversarially reviewed, all verified LIVE in the local preview (identical code to Cloud):
- Design system (`_IER_CSS`): `--ier-*` tokens, elevated cards with real shadows, hover-lift
  buttons, a clean underline tab bar (emoji labels removed), card-style expanders, tighter type.
- Portfolio: replaced the raw `st.dataframe` with a premium color-coded holdings table
  (`holdings_table_html`) — symbol+sector stacked, right-aligned tabular numbers, P&L% green/red,
  weight micro-bars, sorted by value; phones show the essential trio (Holding/Value/P&L%) so
  nothing overflows 375px. `format_rupees_precise` keeps paise on per-share prices (₹15.62, not a
  rounded ₹16); integer quantities. Allocation charts themed (`style_chart`, transparent bg for
  light/dark, app palette).
- Research: `verdict_rating_html` rating strip (Valuation/Quality/Leaning/Confidence at a glance,
  colour-toned by favourability; Confidence stays neutral — it's data coverage, not good/bad news)
  + `figures_table_html` evidence table with a verification-status chip (Verified green / 1-source
  amber / Conflict red).
- Invest: `allocation_table_html` — the suggested spread as Stock / Add (₹) / a share bar.
- Ask: `claim_card_html` — each answer claim as one unified trust card with a left accent
  (green verified fact / red unverified primary-only / blue reported/opinion/estimate).
- All builders are PURE + unit-tested (holdings, formatter, rating strip, claim card, figures,
  allocation). One self-caught bug: a `price()` helper collided with a `price` script var (a float
  shadowed the function) — the builder unit test surfaced it before it shipped.
- DEPLOY note: a code-only Cloud push that adds a NEW `src` symbol imported by app.py can
  stale-import (Cloud reruns without reimporting the changed module). The `format_rupees_precise`
  push hit this LIVE; forced a full rebuild via a requirements.txt touch and logged the rule
  (pair any new top-level src import with a requirements bump). Later UI pushes added no new src
  import, so they deployed clean.
- VERIFIED: `./verify.sh` 818 -> 877 tests; `run_eval.py` 4-gate PASS at every merge; deploy
  confirmed healthy after each push (gate renders, zero console errors).

## 2026-07-26 — BSE announcements wired into the freshness ingest (H4 fix + wiring)
Live verification of H4/H5 (on the residential-IP Mac) found the BSE source used a DEAD endpoint:
`AnnGetData/w` with a capital-S `strScrip` returned "No Record Found!" for every query. Fixed to
the working `AnnSubCategoryGetData/w` + `subcategory=-1` + lowercase `strscrip` (per the maintained
BseIndiaApi client) — live-verified 15 real RELIANCE announcements. Then wired BSE INTO the ingest:
- `src/data/bse_scrip_codes.py`: `BseScripResolver` maps an NSE symbol -> BSE scrip code via a
  live-verified static seed of the 31 holdings + a live PeerSmartSearch fallback that resolves ONLY
  on an EXACT ticker match (never guesses a scrip — a wrong code would ingest another company's
  filings; unresolvable -> None, NSE still covers it). Adversarial review found + fixed a real
  hazard: a span-less `<li>` could pair a scrip with a different row's span; now parsed per-`<li>`
  with a scrip cross-check.
- `run_ingest` gained an optional BSE pass (into the SAME log); the snapshot count is `max(NSE,BSE)`
  not the sum (the same filing is disclosed to both exchanges). Live-verified: RELIANCE +15,
  ICICIBANK +32 real BSE announcements alongside NSE.
- H5 finding (recorded, not a code change): pdfplumber table extraction RUNS on a real 187-page RIL
  AR and produces typed records, but quality is rough on borderless tables — a best-effort
  improvement over pypdf, not a reliable precise-figure extractor; the cross-verify gate stays the
  real guard.

## 2026-07-26 — H7: freshness snapshot reaches the DEPLOYED app (closes the biggest residual)
The W1 freshness engine only ever saw live data on the owner's Mac (NSE/BSE block Streamlit
Cloud's datacenter IP; Cloud can't run the scheduler), so the parents never saw freshness on the
deployed app. H7 bridges that gap through the EXISTING Sheets backend — no Apps Script change (the
bridge already does generic tab read/write):
- `src/freshness/snapshot.py`: `SymbolSnapshot` + `snapshot_for` project each symbol's ingest-run
  summaries (news/announcements/AR) into a compact row; `as_row`/`from_row` round-trip through the
  gateway's `{header: value}` shape (from_row tolerant — Sheets returns strings); `parse_snapshot`
  groups rows by symbol. Built from the RUN summaries (symbol is unambiguous at ingest time), never
  reverse-engineered from a log key; never fabricates a date (AR sentinels -1 / "").
- `scripts/ingest_freshness.py`: `--publish` flag → after ingest, `build_snapshot` +
  `publish_snapshot` write the `Freshness` tab via `gateway_from_env` (APPS_SCRIPT_URL/TOKEN).
  Best-effort: a Sheet blip or missing config prints `publish: skipped/failed`, never aborts the
  run. `load_dotenv()` in `main()` (CLI only) so the launchd job picks up `.env`.
- `app.py`: `load_freshness_snapshot()` (cached 30 min, degrades to `{}` on any backend failure) +
  the pure `freshness_snapshot_line()` → a "🔄 Data last refreshed …; tracking N news and M filings
  from the last 120 days; latest annual report FY2026 (…)." banner in the Research tab, right after
  the H6 live-AR block (which shows nothing on Cloud). Flags a 🔄 "refresh looks overdue" after 3
  days so a stalled Mac pipeline is visible, not silent. Every new access try/except-guarded.
- DEPLOY.md §4d documents the launchd `--publish` cron.
- VERIFIED: `./verify.sh` ALL GREEN 818 → 840 tests (+22, incl. an AppTest that renders the banner
  through Streamlit's own runtime with the Sheets read injected at the src-level gateway);
  `run_eval.py` 4-gate PASS. Additive: with no `--publish` / no snapshot, ingest + app behaviour
  byte-unchanged. HONEST residual: the live round-trip (Mac cron → real Sheet → Cloud read) is
  offline-unverified here — needs the owner's Sheet URL/token + the launchd install.

## 2026-07-25 — SPEC v4 W7 UI, increment 1 (functional integration + trust UI)
First user-visible wiring of the W3/W4/W6 spine into the deployed app (app.py). Ask tab only;
research-only, additive, degrade-safe at every new boundary.
- Ask tab now routes the grounded-answer path through the W4 `ResearchOrchestrator`
  (PLAN -> RETRIEVE -> COMPUTE -> VERIFY[gate] -> WRITE), reusing the SAME analyst/retrieval/pins/
  as-of/hint the direct path used, so a growth/CAGR figure is pre-computed in Python and the model
  only phrases it (compute-don't-generate, end to end). DEGRADE-SAFE: on any orchestrator error it
  falls back to the proven `grounded.answer`, so the new layer can never take the parents' page down.
- Trust UI on each answer claim, each rendered inside its own try/except: claim-type BADGE
  (`st.badge`, green ONLY for a verified fact); click-through SOURCE SPANS (`Claim.spans()` ->
  source · locator + exact supporting quote); FRESHNESS banner (`describe_freshness` off the news
  locator date -> stale/undated visibly flagged, never shown as current); SHOW-THE-COMPUTATION
  expander for any `ComputedFigure` the orchestrator produced (label + value + inputs + formula).
- Preserved: password gate, all four tabs, MF/SIP, disclaimer, Research/Invest/Portfolio unchanged.
- Pure, testable helpers (`claim_badge`, `claim_freshness_lines`, `format_computed_figure`) with
  5 new tests incl. an AppTest that renders the badge through Streamlit's own runtime.
- VERIFIED: `./verify.sh` ALL GREEN 715 -> 720 tests, app smoke clean; `run_eval.py` EVAL GATE PASS.

## 2026-07-25 — SPEC v4 W1 freshness engine, increment 2 (filings coverage + scheduled entrypoint)
Built on increment 1's `src/freshness/` event log. Extended freshness COVERAGE beyond news:
- `src/data/announcements_source.py`: NSE/BSE corporate-announcement source (results, board
  meetings, dividends, allotments, AGM notices) behind an injectable fetcher; parser mirrors the
  real `/api/corporate-announcements` JSON; tier PRIMARY; degrades to [] on fetch failure. ToS
  reality documented (personal-use; licensed feed swaps in behind the seam).
- `src/data/nse_annual_reports.py`: added `AnnualReportRef` + `latest_report()` (URL + fiscal
  year + FY-end as-of); `latest_report_url()` now delegates to it. Conservative as-of = FY end
  (31 Mar), so staleness errs older, never fresher-than-reality.
- `src/freshness/filings_ingest.py`: `ingest_announcements` + `ingest_annual_report` record into
  the SAME log; one logical AR record per symbol so next year's report supersedes; reject-hard at
  the core, degrade-per-item at the batch.
- `scripts/ingest_freshness.py`: runnable scheduled entrypoint (mirrors daily_suggestions.py) —
  news + announcements + AR per symbol into the log, per-symbol summary. Thin, tested
  orchestration (`run_ingest`/`parse_symbol_args`/`format_summary`). launchd/cron documented, NOT
  installed. Log at `data/freshness/events.jsonl` (gitignored).
- Coverage: added a regression exercising `news_ingest`'s `errors` counter across BOTH bad-input
  classes (no-key cluster + core-rejected record) with the batch surviving.
- VERIFIED: `./verify.sh` ALL GREEN — 582 -> 611 tests (+29), app smoke clean. Offline (no live
  endpoints hit).

## 2026-07 — v3 expert-grade platform (V1-V6 complete)
Built the SPEC v3 platform for the parents' real-money use, all FREE (no paid keys):
- V1 cross-verification engine (consensus rule, 2% tolerance, computed-identity checks).
- V2 report + hard expert-review lifecycle (DRAFT until approved; conflict-blocked approval;
  audit trail; caveated verdict, never certainty).
- V3 analysis framework (valuation/earnings-quality/leverage/pledge) on verified figures only.
- V4a source-adapter seam + HttpDocumentAdapter (live-verified on a real NSE annual report).
- V4b yfinance figure source; V4c free independent sources (Screener + annual-report LLM
  extraction, grounded by verbatim quote); V4d fiscal-year alignment; V4e annual report as a
  year-tagged tiebreaker + consensus verification (INFY net profit verified 3-source live).
- V5 company-search UI: pick a company or live NSE symbol -> draft report -> expert review
  (approve/reject) -> approved. Optional annual-report URL adds cross-verification.
- V6 self-improvement loop: expert corrections -> ground-truth cases -> replayed every run;
  trusted-but-wrong must stay 0; scored in-app + scripts/run_eval.py gate.
- 113 tests green. All commits free of AI attribution. Verify with ./verify.sh.
- Open (V4f): more free sources (BSE, Tickertape via bharat-sm-data), per-symbol AR URL
  auto-resolution, browser-MCP fallback for hosts blocking plain HTTP.
- Owner still to provide: the real holdings Google Sheet (only after the platform is trusted).

## 2026-06-25 (loop paused) — autonomous build complete
- Owner chose to pause the /loop (cron job 9ff18873 cancelled). The build is complete and
  verified for everything buildable without owner data: Streamlit app (portfolio, MF/SIP,
  grounded research mentor, glossary, first-time guide), provider-agnostic LLM (live-verified
  on local Ollama qwen2.5:7b), verify.sh gate, build-tested Dockerfile, Streamlit Cloud deploy
  guide, private data gitignored. 51 tests green.
- Blocked on owner inputs: real sources (config/sources.yaml + documents/), parents' holdings
  CSV + the funds they hold, an IPO data source choice.
- PROCESS FIX: loop commits had carried a Co-Authored-By: Claude trailer (harness default)
  that violates the repo's no-AI-attribution rule. Repo was unpushed; local history scrubbed
  of the trailer this session. Going forward: no AI attribution in commits.

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

## 2026-06-18 (session 6, /loop) — plain-English readability for non-expert parents
- src/glossary.py: curated one-line definitions (no LLM). Wired as help= tooltips on P&L,
  HHI, beta, volatility, drawdown, SIP, plus a glossary expander. 51 tests green (+3); AppTest
  confirms help + glossary render.
- README fully refreshed to the current feature set (portfolio, MF/SIP, grounded mentor,
  sources/credibility contract, provider-agnostic LLM, glossary, layout); all referenced
  paths verified to exist. Next: quick correctness pass on SIP/analysis edge cases; then
  fund-factsheet ingestion (needs owner docs); IPO path (needs owner data source).
- STILL the highest-value lever: owner's real sources (config/sources.yaml + documents/) and
  which funds/stocks the parents hold. Autonomous marginal value is tapering.

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

## 2026-07-25 — SPEC v4 platform upgrade (W1-W8) — LOCAL, awaiting push
Alinea-inspired research-grade upgrade, built on local `main` (NOT pushed; the deployed
parents' app is untouched). 754 tests green; 4 eval gates pass (ground-truth / red-team 4/4 /
numeric-exact 7/7 / compliance-lint 0); live-smoked end to end (Ask-tab trust UI renders on a
real Ollama answer; premium redesign verified desktop/dark/mobile).
- W1 freshness engine: event log + content-hash dedup + near-dup news clustering + as-of/staleness;
  NSE/BSE announcement + annual-report/AGM ingestion; scripts/ingest_freshness.py entrypoint.
- W2 retrieval: element-aware chunking + typed numeric records with provenance. (Dense-embedding
  retrieval + rerank + real PDF-table extraction DEFERRED to protect the free Cloud deploy.)
- W3 compute-don't-generate: record-backed numeric grounding + deterministic compute seam.
- W4 orchestrator: plain-Python plan->retrieve->compute->verify[gate]->write (no framework dep).
- W5 eval: red-team suite + unit-normalized numeric-exact-match; found+fixed the crore/lakh unit
  trap (numbers_unit_consistent). 4-gate run_eval.py.
- W6 citations: span-level click-through (source . locator . exact quote).
- W7 UI: orchestrator wiring + trust UI (claim badges, source spans, freshness banners, show-the-
  computation) + premium visual redesign (native light/dark theme, mobile-first).
- W8 compliance: AI-usage disclosure (footer/answer/PDF) + self-voice advice/return/win-rate lint
  (ast-scoped literals, negation guard) as the 4th eval gate.
- Fixed a real prod bug in passing: lxml missing from requirements (Screener silently broken on Cloud).
NEXT / DEFERRED: dense retrieval + rerank + PDF-table extraction (W2 inc2); residential-IP live smoke
of the NSE/BSE announcement + AR fetchers (they degrade safely but real coverage is unconfirmed);
period-mixing claim-layer guard; more sources (BSE, Tickertape). Awaiting owner go to push.
