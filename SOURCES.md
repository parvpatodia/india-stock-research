# Source map (honest, current as of 2026-06-30)

How each source is tiered, accessed, what it costs, and whether it works today. Access is via
`DocumentSourceAdapter` (src/sources/adapters.py) so anything below plugs into the same
grounding + cross-verification path. Nothing is trusted unless registered and tiered.

## Reality check
There is NO single free official API for "all AGM transcripts and annual reports for all
companies." Official vendor announcement feeds cost ~Rs 3 lakh/year. So we assemble: a generic
document fetcher (works now), owner-provided REST APIs (slot in when keys arrive), and a
browser fallback for sites that block plain HTTP.

## Tier 1 — Primary (citable as fact, after cross-verification)
| Source | Access | Cost | Status |
|---|---|---|---|
| NSE annual reports (`nsearchives.nseindia.com`) | HttpDocumentAdapter (direct PDF) | free | **WORKS, verified**: pulled Infosys FY26 AR, 1.3M chars |
| BSE annual reports / announcements | unofficial API (BseIndiaApi) or browser | free | to wire; may need browser fallback |
| SEBI corporate filings (`sebi.gov.in`) | HttpDocumentAdapter / browser | free | to wire |
| Company IR pages (AR, AGM, investor PPT, transcripts) | HttpDocumentAdapter (per-URL) | free | works for direct PDF/HTML URLs |
| NSE announcements (results, board meetings, dividends, allotments, AGM notices) | `AnnouncementSource` (injectable fetcher; default primes NSE cookies) | free (personal-use) | **wired, offline-tested**: parser mirrors the real `/api/corporate-announcements` JSON; feeds the freshness log |
| BSE announcements (same categories) | `BseAnnouncementSource` (injectable fetcher; default primes BSE cookies) | free (personal-use) | **wired, offline-tested** (H4): `parse_bse_announcements` maps the `AnnGetData` shape (NEWSSUB/HEADLINE, CATEGORYNAME, NEWS_DT, ATTACHMENTNAME) into the same `Announcement`; caller supplies a numeric SCRIP code; live JSON field names unverified offline, parser degrades to [] on drift |

Note: `api.nseindia.com` JSON endpoints block datacenter IPs; the `nsearchives` PDF host does
not. For blocked hosts (NSE api, Screener behind login) use the browser MCP.

## Freshness ingestion (SPEC v4 W1)
Scheduled ingestion keeps the corpus up to date. `scripts/ingest_freshness.py` runs, per symbol,
three feeds into ONE append-only event log (`src/freshness`, JSONL; content-hash dedup + version
supersession + as-of/staleness fall out of the stream):

| Feed | Source module | Log `kind` | as-of date |
|---|---|---|---|
| Recent news | `NewsSource` (Google News RSS + yfinance) | `news` | publisher's published date |
| Corporate announcements | `AnnouncementSource` (NSE/BSE) | `announcement` | disclosure datetime |
| Latest annual report | `NseAnnualReportResolver.latest_report` | `annual_report` | fiscal-year end (31 Mar) |

Run (owner's Mac, residential IP): `./.venv/bin/python scripts/ingest_freshness.py "NSE:RELIANCE=Reliance Industries" INFY`.
Log path: `FRESHNESS_LOG_PATH` or `data/freshness/events.jsonl` (`/data/` is gitignored).
**ToS reality (SPEC v4 §5):** the exchange endpoints are personal-use / non-redistributable and
block datacenter IPs, so this runs on the owner's Mac like the daily job, NOT on Streamlit Cloud;
every fetcher is an injectable seam so a licensed feed (GDFL/TrueData/indianapi) swaps in for a
public product without touching the parser or ingestion path. Scheduling is documented in the
script docstring; **no launchd/cron job is installed by the build** — the owner wires it.

## Tier 2 — Aggregators (cross-check against primary; shown as such)
| Source | Access | Cost | Status |
|---|---|---|---|
| yfinance | wired (`src/data/yfinance_provider.py`) | free | works (prices, fundamentals) |
| AMFI NAVAll | wired (`src/data/amfi_provider.py`) | free | works (MF NAVs) |
| Screener.in | no official API; login + browser, or unofficial scraper | Rs 4,999/yr premium | needs owner login |
| Tickertape Pro | unofficial python client | Rs 2,399/yr | needs owner account |
| Tijori Finance | no official API; extracts AR operational metrics | free tier | evaluate |
| RapidAPI "NSE BSE financial data" / indianapi.in | REST API | paid tiers | needs owner key |
| ICICI Breeze / Zerodha Kite | broker API | free / paid | needs owner account |

## Tier 3 — Context only (attributed, dated, never a fact)
Finfluencers, forums.

## What I need from the owner to widen coverage
1. API keys for any paid REST source you want (RapidAPI / indianapi / broker) — I add an adapter.
2. Logins for Screener/Tickertape if we use them (I drive them via the browser MCP).
3. Either a list of company IR/AR URLs, or a go-ahead to auto-derive annual-report URLs from
   NSE archives per symbol (works now, free).
