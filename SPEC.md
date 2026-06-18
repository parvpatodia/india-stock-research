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
