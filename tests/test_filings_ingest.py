"""Record corporate announcements and the latest annual-report availability into the SAME
append-only ingestion log the news feed uses. All offline: the announcement fetcher and the NSE
annual-report resolver's fetcher are injected against fixtures; a fixed clock + temp dir."""
import json
from datetime import date, datetime, timezone

from src.data.announcements_source import (
    NSE_ANNOUNCE_SOURCE_ID,
    Announcement,
    AnnouncementSource,
)
from src.data.nse_annual_reports import AnnualReportRef, NseAnnualReportResolver
from src.freshness.event_log import IngestionLog
from src.freshness.filings_ingest import (
    ANNUAL_REPORT_SOURCE_ID,
    ar_item_key,
    ingest_announcements,
    ingest_annual_report,
)

_T0 = datetime(2026, 7, 9, 10, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 7, 9, 11, 0, 0, tzinfo=timezone.utc)


def _clock(*instants):
    seq = list(instants) or [_T0]

    def clock():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return clock


ANN_FIXTURE = json.dumps([
    {"symbol": "RELIANCE", "desc": "Financial Results",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/RELIANCE_Q1.pdf",
     "attchmntText": "Audited financial results for the year ended March 31, 2024",
     "sort_date": "2024-07-19 18:32:00"},
    {"symbol": "RELIANCE", "desc": "Dividend",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/RELIANCE_DIV.pdf",
     "attchmntText": "Recommendation of final dividend of Rs 10 per equity share",
     "sort_date": "2024-07-19 18:40:00"},
])

AR_LISTING_2024 = json.dumps({"data": [
    {"toYr": "2024", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2024.pdf"},
    {"toYr": "2023", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2023.pdf"},
]})
AR_LISTING_2025 = json.dumps({"data": [
    {"toYr": "2025", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2025.pdf"},
    {"toYr": "2024", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2024.pdf"},
]})


# --- corporate announcements ------------------------------------------------------------------

def test_ingest_announcements_records_dated_primary_events(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    src = AnnouncementSource(fetcher=lambda s: ANN_FIXTURE)
    summary = ingest_announcements(log, src, symbol="RELIANCE")
    assert summary.fetched == 2
    assert summary.new == 2
    assert summary.errors == 0
    events = log.current()
    assert len(events) == 2
    for ev in events:
        assert ev.kind == "announcement"
        assert ev.source_id == NSE_ANNOUNCE_SOURCE_ID
        assert ev.as_of == "2024-07-19"
        assert ev.ref.endswith(".pdf")


def test_reingesting_same_announcements_is_all_skipped(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    src = AnnouncementSource(fetcher=lambda s: ANN_FIXTURE)
    ingest_announcements(log, src, symbol="RELIANCE")
    n_before = len(log.events())
    summary = ingest_announcements(log, src, symbol="RELIANCE")
    assert summary.new == 0
    assert summary.skipped == 2                    # identical content -> dedup no-ops
    assert len(log.events()) == n_before           # append-only log did not grow


def test_ingest_announcements_degrades_when_fetch_fails(tmp_path):
    def boom(_):
        raise RuntimeError("network down")
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    summary = ingest_announcements(log, AnnouncementSource(fetcher=boom), symbol="RELIANCE")
    assert summary.fetched == 0
    assert summary.new == 0
    assert len(log.events()) == 0


def test_ingest_announcements_errors_counter_survives_bad_items(tmp_path):
    # (1) an announcement with no usable identity (empty ref AND empty title) and (2) one the
    # core rejects hard (empty source_id) both bump errors; the good one still lands.
    good = Announcement(symbol="X", title="Board meeting outcome dividend declared",
                        as_of="2024-07-19", source_id=NSE_ANNOUNCE_SOURCE_ID,
                        ref="https://x/good.pdf", category="Board Meeting")
    no_key = Announcement(symbol="", title="", as_of="", source_id=NSE_ANNOUNCE_SOURCE_ID)
    bad_source = Announcement(symbol="X", title="Credit rating upgrade by CRISIL",
                              as_of="2024-07-19", source_id="", ref="https://x/rating.pdf")

    class Stub:
        def fetch(self, symbol):
            return [good, no_key, bad_source]

    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    summary = ingest_announcements(log, Stub(), symbol="X")
    assert summary.errors == 2
    assert summary.new == 1
    assert len(log.current()) == 1


# --- annual-report availability ---------------------------------------------------------------

def test_ingest_annual_report_records_latest_with_fy_end_as_of(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    resolver = NseAnnualReportResolver(fetcher=lambda s: AR_LISTING_2024)
    res = ingest_annual_report(log, resolver, symbol="RELIANCE")
    assert res.found is True
    assert res.recorded is True
    assert res.action == "new"
    assert res.fiscal_year == 2024
    assert res.as_of == "2024-03-31"               # Indian FY ends 31 March
    ev = log.latest_for(ar_item_key("RELIANCE"))
    assert ev is not None
    assert ev.kind == "annual_report"
    assert ev.source_id == ANNUAL_REPORT_SOURCE_ID
    assert ev.ref.endswith("AR_2024.pdf")


def test_annual_report_freshness_reads_the_fy_end(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    resolver = NseAnnualReportResolver(fetcher=lambda s: AR_LISTING_2024)
    ingest_annual_report(log, resolver, symbol="RELIANCE")
    f = log.freshness_for(ar_item_key("RELIANCE"), today=date(2026, 7, 9), threshold_days=400)
    assert f is not None
    assert f.stale is True                          # FY24 end is > 400 days before 2026-07-09
    assert f.as_of == "2024-03-31"


def test_new_fiscal_year_supersedes_prior_annual_report(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock(_T0, _T1))
    ingest_annual_report(log, NseAnnualReportResolver(fetcher=lambda s: AR_LISTING_2024),
                         symbol="RELIANCE")
    res = ingest_annual_report(log, NseAnnualReportResolver(fetcher=lambda s: AR_LISTING_2025),
                               symbol="RELIANCE")
    assert res.action == "superseded"
    assert res.fiscal_year == 2025
    current = log.current()
    assert len(current) == 1                        # one logical AR record per symbol
    assert current[0].as_of == "2025-03-31"


def test_ingest_annual_report_absent_when_resolver_returns_none(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    resolver = NseAnnualReportResolver(fetcher=lambda s: None)   # NSE blocked / no report
    res = ingest_annual_report(log, resolver, symbol="RELIANCE")
    assert res.found is False
    assert res.recorded is False
    assert len(log.events()) == 0


def test_ingest_annual_report_degrades_when_resolver_raises(tmp_path):
    class Boom:
        def latest_report(self, symbol):
            raise RuntimeError("resolver blew up")
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    res = ingest_annual_report(log, Boom(), symbol="RELIANCE")
    assert res.found is False
    assert len(log.events()) == 0


def test_ingest_annual_report_rejects_a_bad_as_of_without_crashing(tmp_path):
    # Defensive: a resolver that hands back an unparseable as_of must not crash the run; the core
    # rejects it and the result reports recorded=False rather than raising.
    class BadRef:
        def latest_report(self, symbol):
            return AnnualReportRef(symbol="X", url="https://x/AR.pdf", fiscal_year=2024,
                                   as_of="not-a-date")
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock())
    res = ingest_annual_report(log, BadRef(), symbol="X")
    assert res.found is True
    assert res.recorded is False
    assert len(log.events()) == 0
