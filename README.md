# India Equity Research

A research and analysis tool for the Indian market (NSE/BSE), built to be usable by
non-expert investors. It analyzes your stock portfolio, looks up mutual funds, projects
SIPs, and answers questions about companies and funds, but only from sources you trust,
with every claim cited.

**This is decision support, not advice.** It never places trades, never recommends what to
buy or sell, and never promises returns. Every number is fetched and timestamped. Research
answers come only from a tiered library of sources you configure: a claim is shown as a
verified fact only when a primary source backs it; otherwise it is labeled opinion or
unverified, or the system says it has no verified answer. Verify every figure before acting.

## What it does

**Portfolio**
- Upload a holdings CSV (Zerodha/Groww exports work). Value, invested, unrealized P&L per
  holding and total, and position weights.
- Concentration: largest-holding weight, Herfindahl index (HHI), effective number of
  holdings, with advisory flags.
- Sector breakdown, and live NIFTY 50 / SENSEX context.
- Risk (1-year history): annualized volatility, beta vs NIFTY 50, worst single-name drawdown.

**Mutual funds & SIPs**
- Look up any fund's NAV by name, live from AMFI (free, no key).
- SIP projection: compound-interest arithmetic on a return you assume, framed clearly as not
  a prediction (real returns vary and can be negative).

**Research mentor (grounded)**
- Ask a plain-English question. The answer comes only from your source library, with
  citations, or it abstains. It never guesses, recommends, or predicts.

**Readability**
- A plain-English glossary and hover-help on every metric, for non-expert readers.

Not included: trade execution, buy/sell recommendations, return guarantees, full-market
screening, IPO data (pending a chosen data source).

## The credibility and grounding contract (why a number can be trusted)

The one property the system enforces: it never presents an unverified claim as a fact.

- **Source tiers.** Primary (annual reports, AGM/SEBI filings, exchange and AMFI data) can
  back a stated fact. Analyst/press is attributed opinion. Social creators are context only,
  attributed and dated, never a fact or a number.
- **Retrieval-grounded.** The model answers only from retrieved chunks of your documents. No
  matching source means no claim.
- **Citation enforcement.** A claim is a verified fact only if every citation backing it is
  primary; anything else is downgraded to "unverified" before it can display.
- **Abstention.** When sources do not answer, the system says so instead of guessing.
- **Human in the loop.** It informs; you decide.

## Sources: bring your own

```bash
cp config/sources.example.yaml config/sources.yaml   # list your sources under primary/analyst/creator
mkdir -p documents                                    # put files named by source id here:
#   documents/<source_id>.pdf   (also .txt, .md)
```

With no `config/sources.yaml`, the app falls back to a bundled synthetic sample so you can
try the research mentor immediately. A file whose stem is not a registered source is skipped
(never ingested as if trusted); an unreadable file is reported, not fatal.

`documents/` and `config/sources.yaml` are gitignored, so your real (possibly private, paid,
or copyrighted) sources are never committed, even when you push to GitHub. Only the synthetic
`sample_data/` and the `config/sources.example.yaml` template are tracked.

## Data sources

- Equities: **yfinance** (free, no key), behind `MarketDataProvider` so a broker feed
  (Upstox/Zerodha) is one new adapter with no change to the analysis.
- Mutual funds: **AMFI NAVAll** (free, no key).
- Research: your tiered document library (above).

## LLM (provider-agnostic, optional)

Only the written research answers/notes need an LLM. All analysis, lookups, and SIP math
work without one. The model is selected by `LLM_MODEL` and routed by LiteLLM, so it is a
config choice, not a code change.

```bash
cp .env.example .env
# pick ONE:
#   A) NVIDIA NIM (free hosted open models): LLM_MODEL=nvidia_nim/... + NVIDIA_NIM_API_KEY
#   B) Ollama (fully local, free, private):  LLM_MODEL=ollama_chat/qwen2.5:7b + LLM_API_BASE
#   C) any other LiteLLM provider:           LLM_MODEL=... (+ LLM_API_KEY / LLM_API_BASE)
```

For the grounded extraction this does, a fast instruct model fits better than a heavy
reasoning one. For your parents' financial data, a local Ollama model keeps everything on
the machine. Verify the LLM end to end (answers from a source, cites it, abstains otherwise):

```bash
SMOKE_MODEL=ollama_chat/qwen2.5:7b ./.venv/bin/python scripts/live_smoke.py
```

## Setup and run

```bash
cd india-stock-research
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/streamlit run app.py
```

Open the URL it prints (default http://localhost:8501). Tick "Use sample portfolio" to try
it, or upload your own CSV.

## Deploy

This is a standard Streamlit app, so the simplest host is Streamlit Community Cloud (free,
git-based):

1. Push this repo to GitHub.
2. At share.streamlit.io, create an app pointing at `app.py` on your branch; pick Python 3.12.
3. For the research mentor, add your LLM config in the app's **Secrets** UI as TOML:
   ```toml
   LLM_MODEL = "nvidia_nim/deepseek-ai/deepseek-v3.2"
   NVIDIA_NIM_API_KEY = "nvapi-..."
   ```
   Streamlit exposes secrets as environment variables, which is how the app reads them.

Two honest notes for a hosted deploy:
- A **local Ollama** model is not reachable from a hosted server. On Streamlit Cloud use a
  hosted provider (NVIDIA NIM). For a fully local, private setup, run on your own machine.
- A hosted deploy means portfolio uploads and questions travel to that server. For maximum
  privacy of your family's data, run locally with a local Ollama model.

### Docker (build-tested)

```bash
docker build -t india-equity-research .
docker run --rm -p 8501:8501 india-equity-research      # then open http://localhost:8501
```

For the research mentor, pass LLM config as env vars:
`docker run --rm -p 8501:8501 -e LLM_MODEL=nvidia_nim/... -e NVIDIA_NIM_API_KEY=... india-equity-research`.
A local Ollama on the host is reachable from the container at `host.docker.internal`
(`-e LLM_MODEL=ollama_chat/qwen2.5:7b -e LLM_API_BASE=http://host.docker.internal:11434`).
The image runs headless with telemetry off and has a `/_stcore/health` healthcheck.

## Portfolio CSV format

Columns are matched loosely, so Zerodha and Groww holdings exports work as-is. Minimum:

```csv
Symbol,Quantity,Avg Cost,Sector
RELIANCE,15,1180.00,Energy
TCS,10,2350.00,IT
```

`Sector` is optional. Symbols may be bare (`RELIANCE`), suffixed (`RELIANCE.NS`), prefixed
(`NSE:RELIANCE`), or carry the equity series tag (`RELIANCE-EQ`). A purely numeric symbol is
treated as a BSE scrip code.

## Tests

```bash
./verify.sh                          # full gate: compile + tests + headless app smoke
./.venv/bin/python -m pytest -q      # just the unit suite
```

`verify.sh` exits non-zero on the first failure, so it works as a pre-deploy gate. The
suite covers the money math and the safety contract: CSV loading, portfolio analysis
(P&L, weights, concentration, risk), the source registry and credibility tiers, the
claim/citation contract (including downgrade of unsourced facts), document retrieval and
abstention, the AMFI parser, the LLM client wiring, document ingestion, SIP math, and the
glossary.

## Layout

```
app.py                       Streamlit UI: portfolio, MF/SIP, research mentor, glossary
config/sources.example.yaml  source registry template (copy to sources.yaml)
documents/                   your source files (named by source id); else sample_data/
scripts/live_smoke.py        live end-to-end check of the grounded pipeline
src/constants.py             domain constants (one source of truth)
src/glossary.py              plain-English term definitions
src/instruments.py           instrument taxonomy (stock/MF/SIP/IPO/other)
src/sip.py                   SIP projection math (pure, tested)
src/portfolio/               models, CSV loader, analysis math (pure, tested)
src/data/                    MarketDataProvider + yfinance adapter; AMFI MF NAV provider
src/sources/                 credibility-tiered source registry
src/research/                grounding/retrieval, claim+citation contract, grounded analyst,
                             document ingestion, portfolio research notes
src/llm/                     provider-agnostic LLM client (LiteLLM)
tests/                       unit tests for all of the above
sample_data/                 sample portfolio CSV + synthetic sources and documents
```
