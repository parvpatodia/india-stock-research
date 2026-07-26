"""Wire the PRIMARY exchange filings into the ingestion log: corporate announcements and the
latest available annual report.

Both feed the SAME append-only IngestionLog the news feed uses, so dedup, version supersession,
and as-of/staleness fall out of the event stream for free. The log records provenance (source id,
as-of date, ref/URL); it does NOT promote a filing's text to a verified number -- that still runs
the grounding + cross-verification path.

Degrade, never crash (LESSONS 2026-07-09): the announcement fetcher and the AR resolver already
abstain to []/None on a feed failure; this adds the outer guard around the whole fetch and a
per-item guard so one malformed filing cannot take down the batch (reject-hard at the core,
degrade-per-item at the batch -- LESSONS 2026-06-18).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..data.announcements_source import AnnouncementSource
from ..data.nse_annual_reports import NseAnnualReportResolver
from .event_log import IngestionLog
from .staleness import DEFAULT_RECENCY_WINDOW_DAYS, is_recent

# The AR log record's provenance id matches AnnualReportFigureSource.source_id, so the freshness
# entry and the figure-extraction source name the same underlying primary document.
ANNUAL_REPORT_SOURCE_ID = "annual_report"


@dataclass
class FilingsIngestSummary:
    fetched: int = 0        # filings pulled from the feed
    new: int = 0            # first-seen filings appended
    superseded: int = 0     # filings whose content changed (prior version superseded)
    skipped: int = 0        # dedup no-ops (identical content already recorded)
    errors: int = 0         # filings that could not be recorded (skipped, batch survived)
    skipped_old: int = 0    # filings older than the recency window -> not ingested (H1)
    undated: int = 0        # filings with no parseable date -> KEPT and counted (never dropped)


def ingest_announcements(log: IngestionLog, source: AnnouncementSource,
                         symbol: str, *,
                         window_days: int = DEFAULT_RECENCY_WINDOW_DAYS,
                         today: date | None = None) -> FilingsIngestSummary:
    """Fetch and record a symbol's corporate announcements into the log. Returns a run summary.

    Recency window (H1): a routine run should record only RECENT filings, not the whole exchange
    history (one live RELIANCE pull returned ~2,871 announcements). When `today` is supplied, an
    announcement whose as-of date is strictly older than `window_days` before it is SKIPPED
    (counted `skipped_old`); an undated announcement is KEPT and counted `undated`, never dropped
    by the window. When `today` is None the window cannot be computed, so nothing is filtered --
    existing callers keep their behavior and the routine entrypoint supplies the run's today."""
    summary = FilingsIngestSummary()
    try:
        items = source.fetch(symbol)
    except Exception:
        # AnnouncementSource already degrades internally; this is the outer abstain-on-failure guard.
        return summary
    summary.fetched = len(items)

    for ann in items:
        key = ann.item_key
        if not key:
            summary.errors += 1     # no usable identity (blank ref AND title) -> skip, survive
            continue
        if today is not None:
            recent = is_recent(ann.as_of, today, window_days)
            if recent is None:
                summary.undated += 1        # undated -> keep + count, never a silent drop
            elif not recent:
                summary.skipped_old += 1    # older than the window -> not ingested
                continue
        try:
            result = log.ingest(item_key=key, source_id=ann.source_id, content=ann.as_text,
                                as_of=ann.as_of, kind="announcement", ref=ann.ref, title=ann.title)
        except ValueError:
            # Reject-hard at the core, degrade-per-item at the batch.
            summary.errors += 1
            continue
        if result.action == "new":
            summary.new += 1
        elif result.action == "superseded":
            summary.superseded += 1
        else:
            summary.skipped += 1
    return summary


@dataclass
class AnnualReportIngestResult:
    found: bool = False          # a report URL was resolved from the exchange listing
    recorded: bool = False       # an event was logged (appended new/superseded, or deduped)
    action: str = ""             # "new" | "superseded" | "skipped" | "" (nothing recorded)
    fiscal_year: int = -1        # the report's fiscal year, or -1 if none
    as_of: str = ""              # ISO date the report is effective (FY end)
    item_key: str = ""           # the log key the record lives under


def ar_item_key(symbol: str) -> str:
    """One logical annual-report record per symbol, so next year's report SUPERSEDES this one
    (same key, new content -> a supersession event) and the app can ask the log for 'the latest
    annual report' by a single stable key."""
    return f"annual_report:{symbol.strip().upper()}"


def ingest_annual_report(log: IngestionLog, resolver: NseAnnualReportResolver,
                         symbol: str) -> AnnualReportIngestResult:
    """Record the latest available annual report for a symbol, dated by its fiscal-year end, so
    the app can show 'latest annual report: FY24, dated 2024-03-31 (stale/fresh)' via the log's
    freshness helper. Degrades to found=False if the exchange listing is blocked/empty."""
    try:
        ref = resolver.latest_report(symbol)
    except Exception:
        # AR resolver already abstains to None; outer guard for anything it lets through.
        return AnnualReportIngestResult(found=False)
    if ref is None or not ref.url:
        return AnnualReportIngestResult(found=False)

    key = ar_item_key(symbol)
    content = (f"Latest annual report for {symbol.strip().upper()}: "
               f"FY{ref.fiscal_year} ({ref.as_of or 'undated'}) {ref.url}")
    title = f"Annual report FY{ref.fiscal_year}"
    try:
        result = log.ingest(item_key=key, source_id=ANNUAL_REPORT_SOURCE_ID, content=content,
                            as_of=ref.as_of, kind="annual_report", ref=ref.url, title=title)
    except ValueError:
        # Defensive: a malformed as_of from a swapped-in resolver must not crash the run.
        return AnnualReportIngestResult(found=True, recorded=False, fiscal_year=ref.fiscal_year,
                                        as_of=ref.as_of, item_key=key)
    return AnnualReportIngestResult(found=True, recorded=True, action=result.action,
                                    fiscal_year=ref.fiscal_year, as_of=ref.as_of, item_key=key)
