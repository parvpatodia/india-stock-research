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

---

# SPEC v2: parent-facing research mentor

## What changed
v1 was a personal portfolio analyzer. v2 makes the system a research mentor the owner's
parents use directly to make better-informed decisions with real money. It explains, in
plain English, what is verified vs opinion vs unknown, with every claim sourced. It still
does not recommend, predict, promise returns, or trade. (Confirmed with owner 2026-06-18:
advisory level = "research mentor, they decide"; influencers = context-only/attributed;
language = English; instruments = stocks + mutual funds/SIPs + IPOs + other research-backed.)

## Users (v2)
The owner's parents: non-expert investors. Output must be readable, plain-language, English,
no unexplained jargon. The owner configures the sources and reviews before they act.

## The credibility + grounding contract (the core of v2)
A claim is shown as **fact only if it is traceable to a primary source**. Otherwise it is
labeled opinion, estimate, or marked unverified. There is no unsourced fact. Mechanism:
- **Source tiers.** Tier 1 PRIMARY (annual reports, AGM, SEBI filings, exchange/AMFI data,
  audited financials) = citable as fact. Tier 2 ANALYST (registered analyst/broker research,
  reputable press) = attributed opinion. Tier 3 CREATOR (YouTube/Instagram finfluencers) =
  context only, attributed, dated, never a basis for a fact or a number.
- **Retrieval-grounded.** The LLM answers only from retrieved chunks of the actual documents.
  No retrieved chunk -> no claim. Every claim cites the chunk(s) it came from.
- **Citation enforcement.** A claim labeled "fact" with no Tier-1 citation is downgraded to
  unverified before display. Validation runs on every result.
- **Calibrated abstention.** If retrieval finds nothing relevant, the system returns "no
  verified answer," never a guess.
- **Human in the loop.** Output informs; the owner and parents decide. No auto-trading.

## Sources (owner-supplied, pluggable)
The owner will supply the source list (sites, AGMs, annual reports, credible creators).
They go in `config/sources.yaml` keyed by tier. Free data adapters built in:
- yfinance (equities) — v1.
- AMFI NAVAll (mutual fund NAVs) — Tier 1, free, no key.
Pending owner input / keys: specific filings, news, creator handles, IPO data feed.

## Architecture additions (v2)
```
config/sources.example.yaml     tier-keyed source list template (owner fills sources.yaml)
src/instruments.py              InstrumentType taxonomy (stock, mutual_fund, sip, ipo, other)
src/sources/registry.py         Source, CredibilityTier, SourceRegistry (+ from_config)
src/research/grounding.py       DocumentStore: chunk + offline TF-IDF retrieve + abstention
src/research/claims.py          Citation, Claim, ResearchResult + validate/enforce citations
src/research/grounded_analyst.py grounded Q&A: retrieve -> LLM(JSON) -> assemble -> validate
src/data/amfi_provider.py       mutual fund NAV lookup (AMFI)
src/llm/client.py               provider-agnostic LLMClient via LiteLLM (NIM/Ollama/etc.)
```

## LLM provider (decided 2026-06-18)
Provider-agnostic via LiteLLM, configured by `LLM_MODEL` env (e.g. `nvidia_nim/...` free
hosted, or `ollama/...` local). No vendor lock-in, no mandatory paid key. Rationale: the
grounding spine validates output regardless of model, so the model is a config line and the
choice is low-stakes; a fast instruct model fits the structured-extraction task better than
a heavy reasoning model. Caveats: free hosted tiers are eval-grade (rate-limited), and
hosted routing sends data off-machine (NIM = US, Ollama = local).

## Feature checklist v2 (each gated by a runnable check)
- [ ] G1 sources: tiers + registry + from_config(yaml); citable_as_fact only for Tier 1. (test)
- [ ] G2 grounding: chunk a doc, retrieve relevant chunks, abstain on no match. (test)
- [ ] G3 claims: Claim/Citation/ResearchResult; enforce_citations downgrades unsourced facts. (test)
- [ ] G4 assemble: model JSON -> ResearchResult with correct tiers; bad/empty -> abstain. (test)
- [ ] G5 grounded analyst: end-to-end with key; clean abstain/degrade without key. (smoke)
- [ ] G6 instruments: taxonomy covers stock/MF/SIP/IPO/other. (test)
- [ ] G7 AMFI: parse NAVAll, look up a scheme NAV. (test on sample + live smoke)
- [ ] G8 parent UI: plain-English, English, readable research view over the engine. (later)
- [ ] G9 IPO data adapter + analysis from DRHP/RHP. (later, needs source)

## Success criteria (v2 increment this session)
- `pytest` green on G1-G4, G6, G7 (deterministic parts).
- Grounded analyst abstains cleanly with no key and no matching source.
- `config/sources.example.yaml` documents how the owner plugs in his sources.
- No unsourced claim can render as fact (enforced in code, covered by a test).

---

# SPEC v3: expert-grade research platform (owner's parents, real money)

## What changed and why
The owner's parents described how real investors work (primary docs first, aggregators
second, a skeptical human checking everything) and asked for a platform that researches a
company end to end and gives an actionable report. Two asks were rejected as impossible/
dangerous and reframed (confirmed with owner 2026-06-30):
- REJECTED: guaranteed buy/sell/timing calls, and "zero errors / never wrong / foolproof."
  No one can time markets or guarantee a call; an LLM cannot be promised zero hallucination.
  Promising either would be the dangerous lie. SEBI also regulates buy/sell advice as a service.
- REFRAMED INTO: a seasoned-analyst-grade research + decision-support platform where a human
  expert signs off before anything is trusted, every number is cross-verified, and the system
  abstains when unsure. "No mistake twice" (regression on every caught error), not "no mistake
  ever." "Foolproof" = traceable + cross-checked + abstaining + expert-gated.

## Locked decisions (owner, 2026-06-30)
1. **Output level = labeled verdict.** Each report has the full analysis PLUS a structured,
   caveated verdict: valuation tier (cheap/fair/expensive vs own history), quality tier
   (strong/weak), and a leaning, each with a confidence level and reasons tied to cited,
   cross-verified figures. Shown as opinion, never as certainty. No naked buy/sell order.
2. **Hard expert sign-off per report.** A report is DRAFT until the human expert reviews and
   approves it. Parents see approved reports; a draft is clearly labeled "not yet reviewed."
   Expert corrections feed the eval/regression loop.
3. **Source APIs owner-provided.** Owner will supply APIs for sites hosting AGM transcripts
   and annual reports for all companies. These plug in behind a source-adapter interface;
   build the interface + free public sources now, slot the APIs in when credentials arrive.
   Use them "accurately and honestly": tier them, cite them, cross-check them.

## Verification protocol (the owner's #1 rule: check twice or thrice)
Every figure: (1) extracted from a primary document with page/locator cited; (2) cross-checked
against >=1 independent source OR a computed identity (balance sheet balances, segment
revenues sum to total, YoY math); (3) shown as fact ONLY if independent sources agree within
tolerance. Disagreement -> status CONFLICT, value withheld, flagged. One source -> SINGLE_SOURCE,
shown but marked not cross-verified. Implemented in `src/research/verification.py`.

## Analysis framework (what a seasoned Indian investor actually checks)
- Business: what it does, segments, moat, sector context (from AR MD&A, filings).
- Financials: revenue/profit trend, margins, ROE/ROCE, debt and interest coverage, and
  operating cash flow vs reported profit (quality of earnings).
- Valuation: P/E, P/B, EV/EBITDA vs the company's OWN history and sector, dividend yield.
- Governance red flags: promoter pledge, related-party transactions, auditor changes/
  qualifications, contingent liabilities, receivables/working-capital blowups, dilution.
- Every metric is a cross-verified figure with a citation, or it is shown as unverified.

## Review lifecycle (safety gate)
Report status: DRAFT -> (expert) APPROVED | REJECTED(+corrections). Only APPROVED is trusted.
Rejections capture the correction as a test case. Audit trail on every report.

## Self-improvement loop (no mistake twice, not no mistake ever)
Expert corrections -> regression test cases; every caught error becomes a test so it cannot
recur; adversarial review of each report before display; a running eval scores figure-accuracy
and verdict-calibration over time. This is measurable and honest.

## Architecture additions (v3)
```
src/research/verification.py    cross-source + computed-identity figure verification
src/research/report.py          Report, Verdict, ReviewStatus, review workflow, audit trail
src/analysis/framework.py       seasoned-investor metric computations (valuation/health/flags)
src/sources/adapters/           source-adapter interface + owner-provided API adapters (later)
src/eval/                       regression cases from expert corrections + accuracy eval
```

## Feature checklist v3 (each gated by a runnable check)
- [ ] V1 verification: cross-source agree->VERIFIED, disagree->CONFLICT, one->SINGLE_SOURCE;
      computed-identity (sum/balance) check. Only VERIFIED is trustworthy. (test)
- [ ] V2 report+review: DRAFT/APPROVED/REJECTED lifecycle; only approved trusted; audit trail;
      caveated Verdict (valuation/quality/leaning + confidence + reasons). (test)
- [ ] V3 analysis framework: valuation/health/governance metrics from verified figures. (test)
- [x] V4a source-adapter interface + HttpDocumentAdapter (PDF/HTML/text); ingest only into
      registered sources. Live-verified fetch of a real NSE annual report. See SOURCES.md. (test+live)
- [x] V4b figure-source layer: FigureSource interface + YFinanceFigureSource (real data);
      pipeline gathers figures across sources and cross-verifies. Single source -> not trusted,
      verdict stays low-confidence (honest); a 2nd source makes agreeing figures VERIFIED.
      Live-verified on RELIANCE via the UI. (test + live)
- [x] V4c independent sources for cross-verification, all FREE (no paid keys, owner's constraint):
      (1) AnnualReportFigureSource (LLM extraction, grounded by verbatim quote); (2) ScreenerFigureSource
      (parses Screener's public tables, crore->absolute). Both cross-check against yfinance; only
      agreeing figures are trusted. verify tolerance = 2%. Live-verified on RELIANCE (ocf/debt/equity
      cross-verify; P&L conflicts are genuine source/period differences, withheld for the expert).
- [x] V4d fiscal-year alignment: cross-verify the same latest common FY across sources (removes
      period-mismatch conflicts); EBIT defined consistently (PBT + interest). Live on RELIANCE:
      OCF/debt/equity verify; net-profit/EBIT/interest conflicts CONFIRMED as genuine source
      definition differences (consolidation/minority), correctly withheld for the expert.
- [x] V4e annual report as a year-tagged 3rd source + consensus verification (largest agreeing
      cluster >=2 sources wins, outliers named/withheld). A two-source conflict is resolved when
      the primary filing confirms one side; a wrong extraction stays a withheld outlier. Live on
      INFY: net profit VERIFIED via 3-source consensus. Tolerance stays 2% (not loosened).
- [x] V4f per-symbol annual-report auto-resolution: NseAnnualReportResolver (cookie-primed NSE
      listing) + nse_annual_report_source chains resolve->fetch->extract; app auto-includes the AR
      per symbol when an LLM is set. Live-verified (RELIANCE/INFY/TCS URLs; INFY net profit
      VERIFIED 3-source auto). Still open: BSE/Tickertape sources, browser-MCP fallback for blocked hosts.
- [x] V6 self-improvement loop: expert correction -> GroundTruth (value + figure snapshot) ->
      replayed on every run. Outcomes MATCH / WITHHELD / TRUSTED_WRONG; trusted-wrong must stay 0.
      Captured in the review panel; scored in-app + via scripts/run_eval.py (gate). "No mistake twice."
- [x] V5 company-search -> draft report -> expert review panel -> approved report UI. Real
      portfolio (holdings.csv) loads by default; research picks any holding or searches any symbol.
- [x] V6 eval loop (done above).
- [x] V7 median-P/E valuation: computed historical median P/E baseline so the valuation tier
      populates for every stock (current P/E stays a cross-verified fact; median is an opinion
      baseline). Live-verified: Adani Power expensive, Brigade/BLS cheap, Shakti Pumps fair.
- [x] V8 bank framework: is_bank routing; ROA (net profit / total assets) instead of the
      industrial lenses; valuation applies; verdict always carries a "GNPA/CASA/CRAR not in free
      feeds, check the filing" caveat. Banks now get valuation + caveat (was blank "unknown").
      Honest ceiling: ROA often stays unknown because net profit / total assets do not
      cross-verify for banks (consolidated vs standalone differ); the system withholds rather
      than trust a mismatch. Deeper fix (diminishing returns): normalize consolidation basis.

## Success criteria (v3)
- Reports are never treated as trusted without expert approval (enforced in code + test).
- No figure is shown as fact unless cross-verified; conflicts are flagged, not hidden. (test)
- The verdict is always caveated and carries the disclaimer + review status.
- Built and proven on sample companies before any real holdings are connected.
