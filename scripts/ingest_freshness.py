"""Scheduled freshness ingestion (SPEC v4 W1): keep the research corpus up to date.

For each holding/watchlist symbol, pull recent NEWS + corporate ANNOUNCEMENTS + the latest
ANNUAL REPORT availability and record each into the SAME append-only ingestion log
(src/freshness), where content-hash dedup, version supersession, and as-of/staleness fall out of
the event stream. Re-running is cheap: unchanged items dedup to no-ops.

Run it:
    python scripts/ingest_freshness.py RELIANCE INFY TCS
    python scripts/ingest_freshness.py "NSE:RELIANCE=Reliance Industries" "INFY-EQ=Infosys"

A "SYMBOL=Company Name" pair feeds the news search a verified company name (a bare ticker is an
unsafe free-text query -- common English-word tickers like PAGE/IDEA/SAIL mismatch; see
NewsSource). The log path is FRESHNESS_LOG_PATH or data/freshness/events.jsonl (data/ is
gitignored at the repo root).

Scheduling (personal use, owner's Mac -- DO NOT install from the build): the exchange endpoints
are personal-use / against ToS from a datacenter IP (SPEC v4 §5), so this runs on the owner's
residential-IP Mac like the daily-suggestions job. A launchd StartCalendarInterval plist (or a
cron line) invokes this daily, e.g.:

    0 7 * * *  cd /path/to/india-stock-research && ./.venv/bin/python scripts/ingest_freshness.py RELIANCE INFY

The fetchers are injectable, so a licensed data feed swaps in without touching this entrypoint.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.announcements_source import AnnouncementSource  # noqa: E402
from src.data.news_source import NewsSource  # noqa: E402
from src.data.nse_annual_reports import NseAnnualReportResolver  # noqa: E402
from src.freshness.event_log import IngestionLog  # noqa: E402
from src.freshness.filings_ingest import (  # noqa: E402
    AnnualReportIngestResult,
    FilingsIngestSummary,
    ingest_announcements,
    ingest_annual_report,
)
from src.freshness.news_ingest import IngestSummary, ingest_news  # noqa: E402
from src.portfolio.loader import normalize_symbol  # noqa: E402

_DEFAULT_LOG = _ROOT / "data" / "freshness" / "events.jsonl"


@dataclass
class SymbolFreshness:
    """One symbol's run across the three feeds."""
    symbol: str
    news: IngestSummary
    announcements: FilingsIngestSummary
    annual_report: AnnualReportIngestResult


def parse_symbol_args(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """Turn CLI args into normalized symbols + an optional company-name map. Each arg is a symbol
    or a 'SYMBOL=Company Name' pair; symbols are normalized the SAME way holdings are (strip
    .NS/.BO/NSE:/BSE:/-EQ), blanks dropped, order preserved, dedup kept simple (first wins)."""
    symbols: list[str] = []
    names: dict[str, str] = {}
    for arg in args:
        raw, sep, name = arg.partition("=")
        sym = normalize_symbol(raw)
        if not sym or sym == "NAN":
            continue
        if sym not in symbols:
            symbols.append(sym)
        if sep and name.strip():
            names[sym] = name.strip()
    return symbols, names


def run_ingest(log: IngestionLog, symbols: list[str], *,
               news_source, announce_source, ar_resolver,
               company_names: dict[str, str] | None = None) -> list[SymbolFreshness]:
    """Ingest news + announcements + the latest annual report for each symbol into the log.
    Thin orchestration: each feed already degrades on failure, so one dead feed or symbol never
    aborts the run. Returns a per-symbol summary."""
    company_names = company_names or {}
    results: list[SymbolFreshness] = []
    for symbol in symbols:
        news = ingest_news(log, news_source, symbol, company_names.get(symbol, ""))
        announcements = ingest_announcements(log, announce_source, symbol)
        annual_report = ingest_annual_report(log, ar_resolver, symbol)
        results.append(SymbolFreshness(symbol=symbol, news=news,
                                       announcements=announcements, annual_report=annual_report))
    return results


def format_summary(results: list[SymbolFreshness]) -> str:
    """A per-symbol run summary (one block per symbol) for the console / scheduler log."""
    lines: list[str] = []
    for r in results:
        ar = r.annual_report
        if ar.found:
            ar_line = f"annual report FY{ar.fiscal_year} ({ar.as_of or 'undated'}) [{ar.action or 'unchanged'}]"
        else:
            ar_line = "annual report: not available"
        lines.append(
            f"{r.symbol}: "
            f"news new={r.news.new} superseded={r.news.superseded} skipped={r.news.skipped} "
            f"errors={r.news.errors}; "
            f"announcements new={r.announcements.new} superseded={r.announcements.superseded} "
            f"skipped={r.announcements.skipped} errors={r.announcements.errors}; "
            f"{ar_line}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    symbols, names = parse_symbol_args(argv)
    if not symbols:
        print("usage: ingest_freshness.py SYMBOL[=Company Name] [SYMBOL ...]", file=sys.stderr)
        return 2

    import os
    log_path = Path(os.environ.get("FRESHNESS_LOG_PATH", str(_DEFAULT_LOG)))
    log = IngestionLog(log_path)
    results = run_ingest(
        log, symbols,
        news_source=NewsSource(),
        announce_source=AnnouncementSource(),
        ar_resolver=NseAnnualReportResolver(),
        company_names=names,
    )
    print(format_summary(results))
    print(f"log: {log_path} ({len(log.events())} total events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
