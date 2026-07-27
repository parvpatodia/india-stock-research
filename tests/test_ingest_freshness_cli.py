"""The freshness ingestion entrypoint (scripts/ingest_freshness.py) orchestrates news +
announcements + annual-report ingestion per symbol into the shared log. The orchestration is a
thin, testable function; the network is injected so this runs fully offline."""
import json
from datetime import date, datetime, timezone

from src.data.announcements_source import AnnouncementSource, BseAnnouncementSource
from src.data.bse_scrip_codes import BseScripResolver
from src.data.news_source import NewsItem
from src.data.nse_annual_reports import NseAnnualReportResolver
from src.freshness.event_log import IngestionLog

from scripts.ingest_freshness import format_summary, parse_symbol_args, run_ingest

_T0 = datetime(2026, 7, 9, 10, 0, 0, tzinfo=timezone.utc)


def _clock():
    return _T0


ANN_FIXTURE = json.dumps([
    {"symbol": "RELIANCE", "desc": "Financial Results",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/RELIANCE_Q1.pdf",
     "attchmntText": "Audited financial results for the year ended March 31, 2024",
     "sort_date": "2024-07-19 18:32:00"}])

AR_LISTING = json.dumps({"data": [
    {"toYr": "2024", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2024.pdf"}]})


class _StubNews:
    """A NewsSource-shaped stub returning fixed items, offline."""

    def __init__(self, items):
        self._items = items

    def fetch(self, symbol, company_name=""):
        return list(self._items)


def _news(symbol):
    return _StubNews([NewsItem(title=f"{symbol} wins a large new order today",
                               publisher="Mint", url=f"https://x/{symbol}",
                               published="2026-07-08", source_id="news_google")])


def test_run_ingest_records_all_three_sources(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_news("RELIANCE"),
        announce_source=AnnouncementSource(fetcher=lambda s: ANN_FIXTURE),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: AR_LISTING),
    )
    assert len(results) == 1
    r = results[0]
    assert r.symbol == "RELIANCE"
    assert r.news.new == 1
    assert r.announcements.new == 1
    assert r.annual_report.recorded is True
    kinds = {ev.kind for ev in log.current()}
    assert kinds == {"news", "announcement", "annual_report"}


def test_run_ingest_handles_multiple_symbols(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE", "TCS"],
        news_source=_StubNews([]),
        announce_source=AnnouncementSource(fetcher=lambda s: "[]"),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: None),
    )
    assert [r.symbol for r in results] == ["RELIANCE", "TCS"]


def test_run_ingest_degrades_when_every_source_fails(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("all feeds down")

    class DeadNews:
        def fetch(self, symbol, company_name=""):
            raise RuntimeError("news down")

    class DeadResolver:
        def latest_report(self, symbol):
            raise RuntimeError("nse blocked")

    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=DeadNews(),
        announce_source=AnnouncementSource(fetcher=boom),
        ar_resolver=DeadResolver(),
    )
    assert len(results) == 1
    assert results[0].news.new == 0
    assert results[0].announcements.new == 0
    assert results[0].annual_report.found is False
    assert len(log.events()) == 0                  # nothing recorded, no crash


def test_parse_symbol_args_normalizes_and_reads_company_names():
    symbols, names = parse_symbol_args(
        ["NSE:RELIANCE=Reliance Industries", "INFY-EQ", "  ", "tcs.ns"])
    assert symbols == ["RELIANCE", "INFY", "TCS"]
    assert names["RELIANCE"] == "Reliance Industries"
    assert "INFY" not in names                       # no company name supplied


OLD_ANN_FIXTURE = json.dumps([
    {"symbol": "RELIANCE", "desc": "Financial Results",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/OLD.pdf",
     "attchmntText": "Old audited results from years ago",
     "sort_date": "2020-01-15 10:00:00"},
    {"symbol": "RELIANCE", "desc": "Board Meeting",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/NEW.pdf",
     "attchmntText": "Recent board meeting outcome",
     "sort_date": "2026-07-01 10:00:00"}])


def test_run_ingest_applies_recency_window(tmp_path):
    # today + a 120-day window -> only the recent announcement lands; the 2020 one is skipped_old.
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_StubNews([]),
        announce_source=AnnouncementSource(fetcher=lambda s: OLD_ANN_FIXTURE),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: None),
        window_days=120, today=date(2026, 7, 9),
    )
    ann = results[0].announcements
    assert ann.new == 1
    assert ann.skipped_old == 1
    text = format_summary(results)
    assert "skipped_old=1" in text


BSE_ANN_FIXTURE = json.dumps({"Table": [
    {"SCRIP_CD": 500325, "CATEGORYNAME": "Board Meeting",
     "NEWSSUB": "Reliance Industries Ltd - Outcome of Board Meeting",
     "NEWS_DT": "2026-07-01T18:32:00",
     "ATTACHMENTNAME": "bse-reliance-boardmeeting.pdf"}]})


def _never(_symbol):
    raise AssertionError("live scrip fetcher must not be called for a seeded/static symbol")


def test_run_ingest_adds_bse_announcements_when_resolver_maps(tmp_path):
    # BSE wired in: a resolvable symbol pulls BSE announcements into the SAME log, tagged with the
    # bse_announcements source (a different attachment URL than NSE, so they coexist, no dedup clash).
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_StubNews([]),
        announce_source=AnnouncementSource(fetcher=lambda s: "[]"),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: None),
        bse_source=BseAnnouncementSource(fetcher=lambda scrip: BSE_ANN_FIXTURE),
        bse_resolver=BseScripResolver(static={"RELIANCE": "500325"}, fetcher=_never),
        window_days=120, today=date(2026, 7, 9),
    )
    assert results[0].bse_announcements.new == 1
    src_ids = {ev.source_id for ev in log.current()}
    assert "bse_announcements" in src_ids


def test_run_ingest_skips_bse_for_an_unresolved_symbol(tmp_path):
    # an unmapped symbol resolves to None -> BSE is skipped, NSE announcement is unaffected.
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_StubNews([]),
        announce_source=AnnouncementSource(fetcher=lambda s: ANN_FIXTURE),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: None),
        bse_source=BseAnnouncementSource(fetcher=lambda scrip: BSE_ANN_FIXTURE),
        bse_resolver=BseScripResolver(static={}, fetcher=lambda s: None),   # never resolves
    )
    assert results[0].bse_announcements.new == 0
    assert results[0].announcements.new == 1                       # NSE unaffected
    assert "bse_announcements" not in {ev.source_id for ev in log.current()}


def test_run_ingest_without_bse_sources_is_unchanged(tmp_path):
    # backward-compat: omitting bse_source/bse_resolver leaves an empty BSE summary, no BSE work.
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_StubNews([]),
        announce_source=AnnouncementSource(fetcher=lambda s: "[]"),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: None),
    )
    assert results[0].bse_announcements.new == 0
    assert results[0].bse_announcements.fetched == 0


def test_format_summary_contains_per_symbol_counts(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    results = run_ingest(
        log, ["RELIANCE"],
        news_source=_news("RELIANCE"),
        announce_source=AnnouncementSource(fetcher=lambda s: ANN_FIXTURE),
        ar_resolver=NseAnnualReportResolver(fetcher=lambda s: AR_LISTING),
    )
    text = format_summary(results)
    assert "RELIANCE" in text
    assert "news" in text.lower()
    assert "FY2024" in text or "2024" in text        # the AR fiscal year is surfaced
