"""H7: per-symbol freshness SNAPSHOT projection + row round-trip.

The snapshot is the bridge that carries the W1 freshness engine's state to the DEPLOYED app: a
Mac-side scheduled run (residential IP, where NSE/BSE actually answer) projects each symbol's run
summaries into a compact row, publishes it to the Sheets backend, and the Cloud app reads it back
(Cloud can't run the scheduler and the exchange endpoints block its datacenter IP -- SPEC v4 W1/§5).
These tests pin the projection math, the tolerant row parse (Sheets returns strings), and grouping.
"""
from __future__ import annotations

from src.freshness.filings_ingest import AnnualReportIngestResult, FilingsIngestSummary
from src.freshness.news_ingest import IngestSummary
from src.freshness.snapshot import (
    FRESHNESS_TAB,
    SNAPSHOT_HEADER,
    SymbolSnapshot,
    parse_snapshot,
    snapshot_for,
    snapshot_rows,
)


def test_tab_and_header_are_stable():
    # the app reader and the publisher must agree on the tab name + column order
    assert FRESHNESS_TAB == "Freshness"
    assert SNAPSHOT_HEADER[0] == "symbol"
    assert set(SNAPSHOT_HEADER) == {
        "symbol", "checked_at", "window_days", "news_recent",
        "announcements_recent", "annual_report_fy", "annual_report_as_of",
    }


def test_snapshot_for_counts_tracked_items_and_carries_ar():
    news = IngestSummary(fetched=10, new=3, superseded=1, skipped=4, skipped_old=2, undated=1)
    ann = FilingsIngestSummary(fetched=8, new=2, superseded=0, skipped=5, skipped_old=1)
    ar = AnnualReportIngestResult(found=True, recorded=True, action="new",
                                  fiscal_year=2026, as_of="2026-03-31",
                                  item_key="annual_report:RELIANCE")
    snap = snapshot_for("reliance", news, ann, ar, checked_at="2026-07-26", window_days=120)
    assert snap.symbol == "RELIANCE"                 # normalized upper
    assert snap.checked_at == "2026-07-26"
    assert snap.window_days == 120
    # KNOWN-DATED recent = new + superseded + skipped(dedup) - undated; skipped_old already excluded
    assert snap.news_recent == 3 + 1 + 4 - 1
    assert snap.announcements_recent == 2 + 0 + 5 - 0
    assert snap.annual_report_fy == 2026
    assert snap.annual_report_as_of == "2026-03-31"


def test_bse_announcements_take_the_max_not_the_sum():
    # the same filing is disclosed to BOTH exchanges, so announcements_recent is max(NSE, BSE), never
    # the sum (which would double-count). Covers the "one exchange quiet that day" case too.
    news = IngestSummary()
    nse = FilingsIngestSummary(new=10, skipped=4)          # recent = 14
    bse = FilingsIngestSummary(new=12, skipped=3)          # recent = 15
    snap = snapshot_for("RELIANCE", news, nse, AnnualReportIngestResult(found=False),
                        checked_at="2026-07-26", window_days=120, bse_announcements=bse)
    assert snap.announcements_recent == 15                 # max(14, 15), not 29


def test_bse_announcements_default_none_is_nse_only():
    news = IngestSummary()
    nse = FilingsIngestSummary(new=7)
    snap = snapshot_for("RELIANCE", news, nse, AnnualReportIngestResult(found=False),
                        checked_at="2026-07-26", window_days=120)   # no bse arg
    assert snap.announcements_recent == 7


def test_undated_items_do_not_inflate_the_recent_count():
    # WHY (honesty): an undated item is KEPT (ingested -> lands in new/superseded/skipped) but has no
    # date, so it must NOT be counted in a "from the last N days" total. Here every tracked item is
    # undated -> the recent count is 0, never "3 news from the last N days" for undated items.
    news = IngestSummary(new=3, superseded=0, skipped=0, undated=3)
    ann = FilingsIngestSummary(new=1, superseded=1, skipped=0, undated=2)
    snap = snapshot_for("RELIANCE", news, ann, AnnualReportIngestResult(found=False),
                        checked_at="2026-07-26", window_days=120)
    assert snap.news_recent == 0
    assert snap.announcements_recent == 0


def test_from_row_strips_a_sheets_iso_timestamp_to_the_date():
    # WHY (Finding 2): Google Sheets can return a date-typed cell as an ISO timestamp; keep only the
    # date part so freshness() can parse it, instead of the banner silently never appearing.
    row = {"symbol": "RELIANCE", "checked_at": "2026-07-26T00:00:00.000Z", "window_days": "120",
           "news_recent": "8", "announcements_recent": "42", "annual_report_fy": "2026",
           "annual_report_as_of": "2026-03-31T00:00:00.000Z"}
    snap = SymbolSnapshot.from_row(row)
    assert snap.checked_at == "2026-07-26"
    assert snap.annual_report_as_of == "2026-03-31"


def test_from_row_leaves_a_non_iso_date_string_for_the_date_engine_to_reject():
    # a locale reformat we don't confidently parse is left as-is -> freshness() degrades to unknown
    # (banner hidden), never a fabricated/guessed date.
    snap = SymbolSnapshot.from_row({"symbol": "X", "checked_at": "7/26/2026"})
    assert snap.checked_at == "7/26/2026"


def test_snapshot_for_no_annual_report_yields_sentinels():
    snap = snapshot_for("INFY", IngestSummary(), FilingsIngestSummary(),
                        AnnualReportIngestResult(found=False),
                        checked_at="2026-07-26", window_days=90)
    assert snap.annual_report_fy == -1
    assert snap.annual_report_as_of == ""
    assert snap.news_recent == 0
    assert snap.announcements_recent == 0


def test_as_row_from_row_round_trip():
    snap = SymbolSnapshot(symbol="TCS", checked_at="2026-07-26", window_days=120,
                          news_recent=5, announcements_recent=2,
                          annual_report_fy=2026, annual_report_as_of="2026-03-31")
    row = snap.as_row()
    assert set(row) == set(SNAPSHOT_HEADER)
    assert SymbolSnapshot.from_row(row) == snap


def test_from_row_tolerates_string_cells_and_junk():
    # Sheets hands every cell back as a string; a blank/garbled numeric must degrade, not crash
    row = {"symbol": " tcs ", "checked_at": "2026-07-26", "window_days": "120",
           "news_recent": "5", "announcements_recent": "", "annual_report_fy": "oops",
           "annual_report_as_of": "2026-03-31"}
    snap = SymbolSnapshot.from_row(row)
    assert snap.symbol == "TCS"
    assert snap.window_days == 120
    assert snap.news_recent == 5
    assert snap.announcements_recent == 0     # blank -> 0
    assert snap.annual_report_fy == -1        # unparseable -> sentinel, never a fabricated year


def test_parse_snapshot_groups_by_symbol_and_skips_empty():
    rows = [
        SymbolSnapshot("RELIANCE", "2026-07-26", 120, 3, 2, 2026, "2026-03-31").as_row(),
        {"symbol": "", "checked_at": "2026-07-26"},           # no symbol -> dropped
        SymbolSnapshot("INFY", "2026-07-26", 120, 1, 0, 2026, "2026-03-31").as_row(),
    ]
    parsed = parse_snapshot(rows)
    assert set(parsed) == {"RELIANCE", "INFY"}
    assert parsed["RELIANCE"].news_recent == 3


def test_snapshot_rows_serializes_each():
    snaps = [SymbolSnapshot("RELIANCE", "2026-07-26", 120, 3, 2, 2026, "2026-03-31")]
    rows = snapshot_rows(snaps)
    assert rows == [snaps[0].as_row()]


def test_parse_snapshot_none_or_empty_is_empty_dict():
    assert parse_snapshot(None) == {}
    assert parse_snapshot([]) == {}
