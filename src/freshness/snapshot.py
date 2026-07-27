"""Per-symbol freshness SNAPSHOT: the bridge that carries the W1 freshness engine's state to the
DEPLOYED app (SPEC v4 W1 + §5).

Streamlit Cloud can't run the scheduler, and NSE/BSE block its datacenter IP, so the freshness
engine only ever sees live data on the owner's residential-IP Mac. This snapshot closes that gap:
a scheduled Mac-side run projects each symbol's ingestion-run summaries into a compact row and
publishes it to the SAME Google Sheets backend the app already uses (a new "Freshness" tab, no
server-side change -- the Apps Script bridge already does generic tab read/write). The Cloud app
reads the tab back and shows the parents a real "data last refreshed ..." banner.

Design choices:
  * Built from the RUN SUMMARIES, not a log projection. The ingestion log's item keys carry symbol
    only for the annual report; news/announcement keys don't. The symbol context is unambiguous at
    ingest time (we ran the symbol), so we project there and never reverse-engineer it.
  * Never fabricates a date. AR fiscal year / as-of come straight from the resolver result (or the
    -1 / "" sentinels when nothing resolved); the reader drops a snapshot with no checked_at.
  * Round-trips through the gateway's plain {header: value} row shape; from_row is tolerant because
    Google Sheets hands every cell back as a string.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .filings_ingest import AnnualReportIngestResult, FilingsIngestSummary
from .news_ingest import IngestSummary

_LEADING_ISO_DATE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})")

# The tab the Mac-side publisher writes and the app reads. Kept here so both sides agree on one name.
FRESHNESS_TAB = "Freshness"

SNAPSHOT_HEADER = [
    "symbol", "checked_at", "window_days", "news_recent",
    "announcements_recent", "annual_report_fy", "annual_report_as_of",
]


def _to_int(value, default: int) -> int:
    """Parse a cell to int, degrading to `default` on blank/garbled input (Sheets returns strings)."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _to_iso_date(value) -> str:
    """Normalize a date cell read back from the Sheet to a bare 'YYYY-MM-DD'.

    We publish plain 'YYYY-MM-DD' strings, but Google Sheets can hand a date-typed cell back as an
    ISO *timestamp* ('2026-03-31T00:00:00.000Z'); keep only the leading date part so freshness() can
    parse it (a plain date passes through untouched). Anything not starting with an ISO date is left
    as-is for the date engine to accept or reject -- it degrades to 'unknown' (banner hidden), never
    a fabricated date. Residual (needs the one live round-trip check): if the Sheet's timezone
    date-shifts a UTC-midnight timestamp, the stripped date can be off by a day; set the Sheet TZ to
    match or keep the column plain-text to avoid coercion entirely (see DEPLOY.md §4d)."""
    s = str(value or "").strip()
    m = _LEADING_ISO_DATE.match(s)
    return m.group(1) if m else s


@dataclass(frozen=True)
class SymbolSnapshot:
    """One symbol's freshness state as published to / read from the Freshness tab."""
    symbol: str
    checked_at: str            # ISO date the run pulled data ("" if unknown -> reader shows nothing)
    window_days: int           # the recency window the counts cover
    news_recent: int           # news items dated within the window, now tracked in the log
    announcements_recent: int  # corporate announcements dated within the window, tracked
    annual_report_fy: int      # latest AR fiscal year, or -1 if none resolved
    annual_report_as_of: str   # AR effective date (FY end), or "" if none

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "checked_at": self.checked_at,
            "window_days": self.window_days,
            "news_recent": self.news_recent,
            "announcements_recent": self.announcements_recent,
            "annual_report_fy": self.annual_report_fy,
            "annual_report_as_of": self.annual_report_as_of,
        }

    @classmethod
    def from_row(cls, row: dict) -> "SymbolSnapshot":
        row = row or {}
        return cls(
            symbol=str(row.get("symbol", "")).strip().upper(),
            checked_at=_to_iso_date(row.get("checked_at")),
            window_days=_to_int(row.get("window_days"), 0),
            news_recent=_to_int(row.get("news_recent"), 0),
            announcements_recent=_to_int(row.get("announcements_recent"), 0),
            annual_report_fy=_to_int(row.get("annual_report_fy"), -1),
            annual_report_as_of=_to_iso_date(row.get("annual_report_as_of")),
        )


def _recent_count(summary: FilingsIngestSummary | IngestSummary | None) -> int:
    """Items KNOWN-DATED within the window and now tracked in the log =
    new + superseded + skipped(dedup) - undated. skipped_old is already outside the window; undated
    is subtracted because it has no date to vouch for freshness (see snapshot_for)."""
    if summary is None:
        return 0
    return max(0, summary.new + summary.superseded + summary.skipped - summary.undated)


def snapshot_for(symbol: str, news: IngestSummary, announcements: FilingsIngestSummary,
                 annual_report: AnnualReportIngestResult, *, checked_at: str,
                 window_days: int,
                 bse_announcements: FilingsIngestSummary | None = None) -> SymbolSnapshot:
    """Project one symbol's run summaries into a snapshot row.

    The "recent" counts are items KNOWN-DATED within the window and now tracked in the log =
    new + superseded + skipped(dedup) - undated. Both skipped_old and undated are excluded: an
    undated item is ingested (kept, never dropped) so it lands in new/superseded/skipped, but it has
    no date to vouch for freshness -- and the banner labels this count "from the last N days", so an
    undated item must NOT inflate it (an item we kept precisely because we don't know its date).
    skipped_old is already outside the window.

    announcements_recent is the MAX of the NSE and BSE counts, not their sum: the same filing is
    disclosed to BOTH exchanges, so summing would double-count it. Max is a safe "at least this many
    distinct recent filings" floor that also covers the case where one exchange is down that day
    (bse_announcements defaults None -> NSE only, existing callers unchanged). The AR fields come
    straight from the resolver result, with -1 / "" when nothing resolved (never a fabricated date).
    """
    return SymbolSnapshot(
        symbol=(symbol or "").strip().upper(),
        checked_at=checked_at,
        window_days=window_days,
        news_recent=_recent_count(news),
        announcements_recent=max(_recent_count(announcements), _recent_count(bse_announcements)),
        annual_report_fy=annual_report.fiscal_year if annual_report.found else -1,
        annual_report_as_of=annual_report.as_of if annual_report.found else "",
    )


def snapshot_rows(snaps: Iterable[SymbolSnapshot]) -> list[dict]:
    """Serialize snapshots to gateway rows (the shape SheetGateway.write expects)."""
    return [s.as_row() for s in snaps]


def parse_snapshot(rows: Iterable[dict] | None) -> dict[str, SymbolSnapshot]:
    """Rows read back from the Freshness tab -> {SYMBOL: snapshot}. Rows with no symbol are dropped;
    a re-publish overwrites the whole tab, so a symbol should appear once (last wins if not)."""
    out: dict[str, SymbolSnapshot] = {}
    for row in rows or []:
        snap = SymbolSnapshot.from_row(row)
        if snap.symbol:
            out[snap.symbol] = snap
    return out
