"""Scheduled freshness ingestion (SPEC v4 W1): keep the research corpus up to date.

For each holding/watchlist symbol, pull recent NEWS + corporate ANNOUNCEMENTS + the latest
ANNUAL REPORT availability and record each into the SAME append-only ingestion log
(src/freshness), where content-hash dedup, version supersession, and as-of/staleness fall out of
the event stream. Re-running is cheap: unchanged items dedup to no-ops.

Run it:
    python scripts/ingest_freshness.py RELIANCE INFY TCS
    python scripts/ingest_freshness.py "NSE:RELIANCE=Reliance Industries" "INFY-EQ=Infosys"
    python scripts/ingest_freshness.py --window-days 90 RELIANCE

A "SYMBOL=Company Name" pair feeds the news search a verified company name (a bare ticker is an
unsafe free-text query -- common English-word tickers like PAGE/IDEA/SAIL mismatch; see
NewsSource). The log path is FRESHNESS_LOG_PATH or data/freshness/events.jsonl (data/ is
gitignored at the repo root).

RECENCY WINDOW (H1): a routine run records only filings/news dated within a window of "today", so
the append-only log stays bounded and reads as recent rather than the whole exchange history (one
live RELIANCE pull returned ~2,871 announcements). Default 120 days; tune with `--window-days N`
or FRESHNESS_WINDOW_DAYS (the flag wins). Undated items are still recorded (flagged, not dropped);
the annual-report record is already bounded (one latest per symbol) and is NOT windowed.

Scheduling (personal use, owner's Mac -- DO NOT install from the build): the exchange endpoints
are personal-use / against ToS from a datacenter IP (SPEC v4 §5), so this runs on the owner's
residential-IP Mac like the daily-suggestions job. A launchd StartCalendarInterval plist (or a
cron line) invokes this daily, e.g.:

    0 7 * * *  cd /path/to/india-stock-research && ./.venv/bin/python scripts/ingest_freshness.py RELIANCE INFY

The fetchers are injectable, so a licensed data feed swaps in without touching this entrypoint.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.announcements_source import AnnouncementSource, BseAnnouncementSource  # noqa: E402
from src.data.bse_scrip_codes import BseScripResolver  # noqa: E402
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
from src.freshness.snapshot import (  # noqa: E402
    SymbolSnapshot,
    snapshot_for,
    snapshot_rows,
    FRESHNESS_TAB,
    SNAPSHOT_HEADER,
)
from src.freshness.staleness import DEFAULT_RECENCY_WINDOW_DAYS  # noqa: E402
from src.portfolio.loader import normalize_symbol  # noqa: E402

_DEFAULT_LOG = _ROOT / "data" / "freshness" / "events.jsonl"


@dataclass
class SymbolFreshness:
    """One symbol's run across the feeds (NSE news + NSE/BSE announcements + annual report)."""
    symbol: str
    news: IngestSummary
    announcements: FilingsIngestSummary
    annual_report: AnnualReportIngestResult
    bse_announcements: FilingsIngestSummary = field(default_factory=FilingsIngestSummary)


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
               bse_source=None, bse_resolver=None,
               company_names: dict[str, str] | None = None,
               window_days: int = DEFAULT_RECENCY_WINDOW_DAYS,
               today: date | None = None) -> list[SymbolFreshness]:
    """Ingest news + NSE/BSE announcements + the latest annual report for each symbol into the log.
    Thin orchestration: each feed already degrades on failure, so one dead feed or symbol never
    aborts the run. The recency window (H1) bounds news + announcements to items dated within
    `window_days` of `today` (annual reports are already bounded -- one latest per symbol -- and
    are NOT windowed). With today=None nothing is windowed (existing callers unchanged).

    BSE (optional): when BOTH `bse_source` and `bse_resolver` are given, each symbol is resolved to
    its BSE scrip code and its BSE announcements are ingested into the SAME log. A symbol that can't
    be resolved safely is simply skipped for BSE (NSE announcements still cover it). Omitting either
    argument leaves BSE off (existing callers unchanged). Returns a per-symbol summary."""
    company_names = company_names or {}
    results: list[SymbolFreshness] = []
    for symbol in symbols:
        news = ingest_news(log, news_source, symbol, company_names.get(symbol, ""),
                           window_days=window_days, today=today)
        announcements = ingest_announcements(log, announce_source, symbol,
                                             window_days=window_days, today=today)
        bse_announcements = FilingsIngestSummary()
        if bse_source is not None and bse_resolver is not None:
            scrip = bse_resolver.resolve(symbol)
            if scrip:
                # ingest_announcements takes whatever identifier the source's fetch expects; the BSE
                # source is keyed by scrip code. BSE attachment URLs differ from NSE, so the same
                # filing on both exchanges lands as two distinct events, never a dedup collision.
                bse_announcements = ingest_announcements(log, bse_source, scrip,
                                                         window_days=window_days, today=today)
        annual_report = ingest_annual_report(log, ar_resolver, symbol)
        results.append(SymbolFreshness(symbol=symbol, news=news, announcements=announcements,
                                       annual_report=annual_report,
                                       bse_announcements=bse_announcements))
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
        b = r.bse_announcements
        bse_line = (f"bse new={b.new} superseded={b.superseded} skipped={b.skipped} "
                    f"skipped_old={b.skipped_old} undated={b.undated} errors={b.errors}; "
                    if b.fetched or b.new or b.errors else "")
        lines.append(
            f"{r.symbol}: "
            f"news new={r.news.new} superseded={r.news.superseded} skipped={r.news.skipped} "
            f"skipped_old={r.news.skipped_old} undated={r.news.undated} errors={r.news.errors}; "
            f"announcements new={r.announcements.new} superseded={r.announcements.superseded} "
            f"skipped={r.announcements.skipped} skipped_old={r.announcements.skipped_old} "
            f"undated={r.announcements.undated} errors={r.announcements.errors}; "
            f"{bse_line}{ar_line}"
        )
    return "\n".join(lines)


def build_snapshot(results: list[SymbolFreshness], *, checked_at: str,
                   window_days: int) -> list[SymbolSnapshot]:
    """Project the per-symbol run summaries into publishable freshness snapshots (H7). Pure: the
    symbol context is unambiguous here (we ran each symbol), so this never reverse-engineers a
    symbol from a log key."""
    return [
        snapshot_for(r.symbol, r.news, r.announcements, r.annual_report,
                     checked_at=checked_at, window_days=window_days,
                     bse_announcements=r.bse_announcements)
        for r in results
    ]


def publish_snapshot(gateway, snaps: list[SymbolSnapshot]) -> bool:
    """Write the snapshot to the Sheets backend's Freshness tab so the DEPLOYED app can read it
    (SPEC v4 W1 + §5: Cloud can't run this scheduler and NSE/BSE block its IP). BEST-EFFORT: the
    ingestion itself already succeeded, so any gateway failure is swallowed and returns False rather
    than aborting the run or raising on the scheduler."""
    try:
        gateway.write(FRESHNESS_TAB, SNAPSHOT_HEADER, snapshot_rows(snaps))
        return True
    except Exception:
        return False


def gateway_from_env(env: dict[str, str]):
    """Build the Sheets gateway for the Mac-side scheduled run from env vars (the app reads the same
    backend from st.secrets). Returns the AppsScriptGateway when APPS_SCRIPT_URL + APPS_SCRIPT_TOKEN
    are set, else None (publishing is skipped -- the ingest still ran). Kept to the Apps Script
    bridge, the keyless backend the app already uses."""
    url = (env.get("APPS_SCRIPT_URL") or "").strip()
    token = (env.get("APPS_SCRIPT_TOKEN") or "").strip()
    if url and token:
        from src.data.sheets_backend import AppsScriptGateway
        return AppsScriptGateway(url, token)
    return None


def resolve_publish(argv: list[str]) -> tuple[list[str], bool]:
    """Pull an optional `--publish` flag out of the args, returning the remaining (symbol) args and
    whether to publish the snapshot to the Sheets backend after ingesting."""
    publish = False
    remaining: list[str] = []
    for arg in argv:
        if arg == "--publish":
            publish = True
        else:
            remaining.append(arg)
    return remaining, publish


def resolve_window_days(argv: list[str], env: dict[str, str]) -> tuple[list[str], int]:
    """Pull the recency window out of the args/env, returning the remaining (symbol) args and the
    window. Precedence: an explicit `--window-days N` / `--window-days=N` flag beats the
    FRESHNESS_WINDOW_DAYS env var, which beats the 120-day default. A non-positive or unparseable
    value falls back to the default (the ingest core also rejects it hard, but the entrypoint is
    forgiving so a typo never aborts the scheduled run)."""
    window = DEFAULT_RECENCY_WINDOW_DAYS
    env_val = env.get("FRESHNESS_WINDOW_DAYS", "").strip()
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                window = parsed
        except ValueError:
            pass
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        val = None
        if arg.startswith("--window-days="):
            val = arg.split("=", 1)[1]
        elif arg == "--window-days" and i + 1 < len(argv):
            val = argv[i + 1]
            i += 1
        else:
            remaining.append(arg)
            i += 1
            continue
        try:
            parsed = int(val)
            if parsed > 0:
                window = parsed
        except (ValueError, TypeError):
            pass
        i += 1
    return remaining, window


def main(argv: list[str] | None = None) -> int:
    import os
    # WHY (CLI only): let the scheduled Mac run pick up APPS_SCRIPT_URL/TOKEN + FRESHNESS_* from the
    # gitignored .env, the same way daily_suggestions.py does, so --publish "just works" from launchd.
    # Only ever runs in main() (the entrypoint), never in tests (they call the helpers directly), so
    # it cannot pollute the test environment (LESSONS 2026-07-08).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, publish = resolve_publish(argv)
    argv, window_days = resolve_window_days(argv, dict(os.environ))
    symbols, names = parse_symbol_args(argv)
    if not symbols:
        print("usage: ingest_freshness.py [--publish] [--window-days N] "
              "SYMBOL[=Company Name] [SYMBOL ...]", file=sys.stderr)
        return 2

    log_path = Path(os.environ.get("FRESHNESS_LOG_PATH", str(_DEFAULT_LOG)))
    log = IngestionLog(log_path)
    today = date.today()           # the run's "today"; the recency window is computed against it
    results = run_ingest(
        log, symbols,
        news_source=NewsSource(),
        announce_source=AnnouncementSource(),
        ar_resolver=NseAnnualReportResolver(),
        bse_source=BseAnnouncementSource(),        # BSE announcements, keyed by scrip code
        bse_resolver=BseScripResolver(),           # NSE symbol -> BSE scrip (seed + live fallback)
        company_names=names,
        window_days=window_days,
        today=today,
    )
    print(format_summary(results))
    print(f"log: {log_path} ({len(log.events())} total events, "
          f"recency window {window_days} days)")

    # H7 (SPEC v4 W1 + §5): publish a per-symbol freshness snapshot to the Sheets backend so the
    # DEPLOYED app can show real freshness -- Cloud can't run this scheduler and NSE/BSE block its
    # IP, so the residential-IP Mac is the only place that sees live data. Best-effort: the ingest
    # already succeeded, so a missing config or a Sheet blip only prints a note, never a failure.
    if publish:
        gateway = gateway_from_env(dict(os.environ))
        if gateway is None:
            print("publish: skipped (set APPS_SCRIPT_URL + APPS_SCRIPT_TOKEN to publish)")
        else:
            snaps = build_snapshot(results, checked_at=today.isoformat(), window_days=window_days)
            ok = publish_snapshot(gateway, snaps)
            print(f"publish: {'ok' if ok else 'failed (Sheet unreachable; ingest still recorded)'} "
                  f"-> {FRESHNESS_TAB} tab ({len(snaps)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
