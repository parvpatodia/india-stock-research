# SPEC v4: research-grade platform upgrade (Alinea-inspired, evidence-driven)

> Derived from a deep teardown of Alinea Invest + the global AI-investing landscape
> (consumer, institutional, Indian), financial-RAG/anti-hallucination SOTA, and the Indian
> data-source + SEBI regulatory reality. Research dossiers summarized inline below.
> MODE: BUILD. Research-only decision support. No buy/sell calls, no return/accuracy marketing.

## 0. The one-paragraph thesis
The viral apps win on **distribution + trust-through-UX**, not AI depth (Alinea's "Allie" is a
single RAG assistant over its own content; its edge is 550M TikTok views + values-based
"playlists", and its weaknesses are billing/withdrawal/support, not model quality — AUM is
only ~$66M on ~$575 average balances). The *serious* research tools converge on ONE accuracy
architecture: **RAG over a curated/verified corpus -> cite every claim to source -> decompose
hard questions -> an independent verification/critic pass -> abstain when unsure**
(AlphaSense, Fiscal.ai click-through citations, Hebbia decomposition, Brightwave LLM-judge,
Rogo numeric-grounding+audit, Bloomberg summary-vs-source checks + human SME). Our v3 already
implements most of this spine, so we are *ahead of Alinea on accuracy* and behind on
**data freshness, retrieval quality, agentic orchestration, eval breadth, and UI**. v4 closes
those gaps without ever crossing the SEBI research-only line.

## 1. Failure modes to design AGAINST (explicit "never do this" list)
- **Bespoke pretrained finance model that goes stale** (BloombergGPT: ConvFinQA 43% vs GPT-4 69-76%; costly to retrain). -> Use RAG over fresh data, not a frozen model's memory.
- **Black-box single score + backtested win-rate marketed as edge** (Kavout K-Score, Danelfin 70% *backtested*, Univest "86% accuracy"). -> No score we can't fully explain; NO win-rate/return claims anywhere (also SEBI enforcement bait).
- **Overfitting sold as alpha; self-reported performance, no audit** (Numerai 17% 2023 drawdown; Bridgewater regime risk). -> Publish methodology + limits, not performance claims.
- **Dark-pattern billing / broken exit** (Alinea + Univest: the dominant real complaints, F BBB rating). -> Not our model (personal tool), but the lesson: trust dies at the billing/withdraw/support layer, so any user-facing edge must be transparent and reversible.
- **Confident wrong number** (our own SPEC's named failure mode). -> compute-don't-generate + cross-verify-or-withhold, enforced in code + eval.

## 2. Highest-leverage technical decisions (from the RAG/anti-hallucination SOTA)
1. **Compute-don't-generate** for every number: extract raw figures -> do all arithmetic (ratios, growth, CAGR) in deterministic Python -> hand computed values back to the LLM only to phrase. (FinQA/ConvFinQA show LLM arithmetic fails; this removes most numeric hallucination.)
2. **Element-aware chunking + typed numeric records with provenance**: parse to structural elements (never split a table from its header/units/period); store each number as `{value, raw, unit, scale, currency, period, company, doc, page, table_id, cell}` with metadata filtering pre-retrieval (stops FY23/FY24 mixing). (arXiv:2402.05131)
3. **A hard verification GATE that abstains** rather than answers on thin evidence. (FinanceBench: GPT-4+retrieval wrong-or-refused on 81% — an honest abstain is the feature.)

## 3. Gap analysis vs current v3 (honest)
| Layer | v3 today | v4 target |
|---|---|---|
| Safety spine | trust tiers, grounding+abstain, claim-typing, cross-verify (consensus+identity), DRAFT->APPROVED review, regression eval | keep; harden numeric grounding to full compute-don't-generate |
| Data freshness | on-demand fetch per request | **scheduled ingestion** of filings/announcements + news + AR/AGM; content-hash dedup; as-of dating; staleness flags |
| Retrieval | offline TF-IDF | hybrid dense+BM25 + cross-encoder rerank; element-aware chunks; Docling/Camelot table extraction |
| Orchestration | procedural pipeline.py | LangGraph: plan -> retrieve -> compute -> verify[gate] -> write; critic/judge pass |
| Eval | regression cases + accuracy eval | + DeepEval CI gate + RAGAS faithfulness + India golden set (exact-numeric-match, unit-normalized) + red-team set (unit traps, period-mixing, phantom figures, as-of violations) |
| Citations | tier + numbers_grounded downgrade | span-level click-through (Claude Citations API or deterministic block IDs) |
| LLM | local Ollama qwen2.5:7b | provider-agnostic; add a citations-capable path; keep free/local default |
| UI | functional Streamlit | YC-grade: onboarding, plain-language, as-of banners, staleness flags, claim-type badges, "show the computation" expanders, conflicting-sources side-by-side |

## 4. Prioritized workstreams (each gated by a runnable check; TDD)
- **W1 Freshness engine** (highest leverage — the explicit ask "keep up to date with news/AGM/annual reports"): scheduler -> per-source fetchers (NSE/BSE announcements, RSS+Google News, AR/AGM PDFs) -> content-hash dedup + near-dup news clustering -> append-only ingestion event log (JSONL) -> as-of/staleness on every record. Gate: ingest a fixed fixture set, assert dedup + timestamps + staleness flags.
- **W2 Retrieval upgrade**: element-aware chunking + typed numeric records; hybrid retrieval + rerank; Docling/Camelot table extraction with confidence gate. Gate: retrieval unit tests on fixture filings; numeric-record extraction exactness.
- **W3 Numeric integrity**: compute-don't-generate layer; strengthen the verify gate. Gate: property tests that no LLM-emitted number lacks a typed source record; ratio math unit-tested.
- **W4 Orchestration**: LangGraph state machine (plan/retrieve/compute/verify/write) + critic pass + abstain-on-fail; keep it debuggable (explicit state, checkpoints). Gate: end-to-end fixture run produces cited memo or clean abstention.
- **W5 Eval harness**: DeepEval CI gate + RAGAS faithfulness + India golden set + red-team set; block a change on regression. Gate: `scripts/run_eval.py` extended; CI-style thresholds.
- **W6 Citations**: span-level click-through to source+page for every fact. Gate: every rendered FACT resolves to a locatable span.
- **W7 UI**: YC-grade research surface (see table). Gate: run + screenshot; mobile 375px; dark mode.
- **W8 Compliance surface**: visible research-only framing, AI-usage disclosure, no win-rate claims, human-in-loop preserved, disclaimer on every export. Gate: presence tests + a lint that fails on forbidden advice/return phrasing.

## 5. Data stack (India, ToS-aware)
Free foundation: **bhavcopy + yfinance** (prices, EOD warehouse), **Screener export + parsed annual reports** (fundamentals), **NSE/BSE corporate-announcement feeds** (filings, freshest free), **RSS (ET/Mint/BS/MC) + Google News** (news, dedup by canonical URL + title embedding), **AMFI/mftool + mfdata.in** (MF NAV/holdings), exchange **bulk/block-deal + shareholding** reports. Keep every source behind the existing adapter seam so a free scraper swaps for a licensed API (GDFL/TrueData/indianapi) per source without touching analysis. **ToS reality:** exchange/broker data is personal-use, non-redistributable — fine for a personal/parents tool; a public multi-user product needs a licensed real-time path. Architect for that line now.

## 6. Compliance posture (SEBI, load-bearing)
Research-only, never personalized buy/sell/allocation, never a return or win-rate claim. Disclose AI usage; the operator owns AI output (Reg 16C). Human-in-the-loop stays (expert approves before parents act). Every figure sourced+dated or withheld. This is not optional polish — it is the difference between a research tool and an enforcement target.

## 7. Build discipline (the loop)
Spec-first (this doc) -> workstream-by-workstream, TDD, gate each commit on `./verify.sh` + eval, commit-on-green, **adversarial review before "done"**, append every caught error to `LESSONS.md` as a regression ("no mistake twice"), one metrics line per build session. Parallelize independent research/impl with subagents. Never mark done on red.
