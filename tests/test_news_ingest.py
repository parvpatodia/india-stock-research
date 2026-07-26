from datetime import date, datetime, timezone

from src.data.news_source import NewsItem, NewsSource
from src.freshness.event_log import IngestionLog
from src.freshness.news_ingest import ingest_news

_T0 = datetime(2026, 7, 9, 10, 0, 0, tzinfo=timezone.utc)


def _clock():
    return _T0


# Google News RSS: three near-duplicate rewrites of ONE story + one unrelated story.
GOOGLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<item>
  <title>Reliance shares slip after SEBI warning - India Infoline</title>
  <link>https://news.google.com/rss/articles/A?oc=5</link>
  <pubDate>Wed, 08 Jul 2026 06:40:49 GMT</pubDate>
  <source url="https://www.indiainfoline.com">India Infoline</source>
</item>
<item>
  <title>SEBI warning sends Reliance shares slipping - Moneycontrol</title>
  <link>https://news.google.com/rss/articles/B?oc=5</link>
  <pubDate>Wed, 08 Jul 2026 07:10:00 GMT</pubDate>
  <source url="https://www.moneycontrol.com">Moneycontrol</source>
</item>
<item>
  <title>Reliance stock falls following SEBI warning today - Mint</title>
  <link>https://news.google.com/rss/articles/C?oc=5</link>
  <pubDate>Wed, 08 Jul 2026 08:00:00 GMT</pubDate>
  <source url="https://www.livemint.com">Mint</source>
</item>
<item>
  <title>Tata Motors quarterly profit jumps twenty percent - ET</title>
  <link>https://news.google.com/rss/articles/D?oc=5</link>
  <pubDate>Tue, 07 Jul 2026 03:00:00 GMT</pubDate>
  <source url="https://economictimes.com">Economic Times</source>
</item>
</channel></rss>"""


def _news_source(rss=GOOGLE_RSS):
    # both fetchers injected -> fully offline; yahoo empty so the RSS path is exercised.
    return NewsSource(rss_fetcher=lambda q: rss, yahoo_fetcher=lambda s: [], max_items=20,
                      today=datetime(2026, 7, 9).date())


def test_ingest_news_collapses_rewrites_and_records_events(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, _news_source(), symbol="RELIANCE",
                          company_name="Reliance Industries", cluster_threshold=0.5)
    # 4 fetched items -> 2 clusters (3 Reliance rewrites collapse, Tata alone) -> 2 events
    assert summary.fetched == 4
    assert summary.clusters == 2
    assert summary.new == 2
    assert len(log.current()) == 2
    for ev in log.current():
        assert ev.source_id == "news_google"
        assert ev.kind == "news"
        assert ev.as_of != ""            # each carries the published date as as-of


def test_reingesting_same_feed_is_all_skipped(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    ingest_news(log, _news_source(), symbol="RELIANCE", company_name="Reliance Industries",
                cluster_threshold=0.5)
    n_before = len(log.events())
    summary = ingest_news(log, _news_source(), symbol="RELIANCE",
                          company_name="Reliance Industries", cluster_threshold=0.5)
    assert summary.skipped == summary.clusters      # nothing changed -> every cluster deduped
    assert summary.new == 0
    assert len(log.events()) == n_before            # append-only log did not grow


def test_ingest_news_degrades_when_all_feeds_fail(tmp_path):
    # WHY (LESSONS 2026-07-09 abstain-on-failure): a fetch that raises must yield an empty run,
    # never crash the caller or leave a half-written log.
    def boom(_):
        raise RuntimeError("network down")
    source = NewsSource(rss_fetcher=boom, yahoo_fetcher=boom,
                        today=datetime(2026, 7, 9).date())
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, source, symbol="RELIANCE", company_name="Reliance Industries")
    assert summary.fetched == 0
    assert summary.new == 0
    assert len(log.events()) == 0


def test_ingest_news_survives_a_single_bad_item(tmp_path):
    # A blank-title item slips through with an empty logical key; it is skipped, the batch survives.
    rss = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title> - India Infoline</title><link>https://x/a</link>
  <pubDate>Wed, 08 Jul 2026 06:40:49 GMT</pubDate>
  <source url="https://www.indiainfoline.com">India Infoline</source></item>
<item><title>Reliance profit rises sharply this quarter - Mint</title><link>https://x/b</link>
  <pubDate>Wed, 08 Jul 2026 07:10:00 GMT</pubDate>
  <source url="https://www.livemint.com">Mint</source></item>
</channel></rss>"""
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, _news_source(rss), symbol="RELIANCE",
                          company_name="Reliance Industries")
    # the good item still lands; the batch did not crash on the empty-title one
    assert summary.new >= 1
    assert any("Reliance profit rises" in ev.title for ev in log.current())


# --- recency window (H1: bound the ingestion log to recent news) ------------------------------

class _StubNews:
    """A NewsSource-shaped stub returning fixed items, offline (source-level recency bypassed so
    the INGEST-level window is what's under test)."""

    def __init__(self, items):
        self._items = items

    def fetch(self, symbol, company_name=""):
        return list(self._items)


def _n(title, published, url):
    return NewsItem(title=title, publisher="Mint", url=url, published=published,
                    source_id="news_google")


def test_news_recency_window_skips_old_keeps_recent(tmp_path):
    today = date(2026, 7, 9)                     # window 120 -> cutoff 2026-03-11
    src = _StubNews([_n("Reliance wins a large new order today", "2026-07-01", "https://x/a"),
                     _n("Reliance signed a deal two years ago now", "2024-07-01", "https://x/b")])
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, src, symbol="RELIANCE", cluster_threshold=0.5,
                          window_days=120, today=today)
    assert summary.new == 1
    assert summary.skipped_old == 1
    assert len(log.current()) == 1
    assert "wins a large new order" in log.current()[0].title


def test_news_recency_window_boundary_is_inclusive(tmp_path):
    # window 60 -> 2026-05-10 is exactly 60 days before today (in), 2026-05-09 is 61 (out). The two
    # headlines share no tokens so they stay separate clusters (isolating the window, not clustering).
    today = date(2026, 7, 9)
    src = _StubNews([_n("Reliance quarterly profit climbs on strong refining margins", "2026-05-10",
                        "https://x/e"),
                     _n("Global shipping freight rates tumble amid weak demand", "2026-05-09",
                        "https://x/o")])
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, src, symbol="RELIANCE", cluster_threshold=0.5,
                          window_days=60, today=today)
    assert summary.clusters == 2                  # the two headlines did not collapse
    assert summary.new == 1                       # 60-day-old item included
    assert summary.skipped_old == 1
    assert "quarterly profit climbs" in log.current()[0].title


def test_news_recency_window_keeps_and_counts_undated(tmp_path):
    today = date(2026, 7, 9)
    src = _StubNews([_n("Reliance undated headline no date given", "", "https://x/u"),
                     _n("Reliance old story from years ago here", "2023-01-01", "https://x/old")])
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, src, symbol="RELIANCE", cluster_threshold=0.5,
                          window_days=120, today=today)
    assert summary.undated == 1
    assert summary.new == 1                       # undated item recorded, not dropped
    assert summary.skipped_old == 1
    assert any("undated headline" in ev.title for ev in log.current())


def test_news_recency_window_off_by_default_backward_compatible(tmp_path):
    src = _StubNews([_n("Reliance ancient news from long ago", "2020-01-01", "https://x/a"),
                     _n("Reliance recent news from this week", "2026-07-01", "https://x/r")])
    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, src, symbol="RELIANCE", cluster_threshold=0.5)  # no today
    assert summary.new == 2
    assert summary.skipped_old == 0


def test_ingest_news_errors_counter_covers_both_bad_input_classes(tmp_path):
    # WHY (coverage, real money): the errors counter must catch BOTH ways a record fails and the
    # batch must survive either -- (1) a cluster whose representative has no usable item_key (blank
    # title AND blank url), and (2) a record the core rejects hard at the boundary (empty source_id
    # -> ValueError). Neither aborts the run; the one good item still lands. Fully offline via a
    # stub source returning crafted NewsItems (the parse/cluster layers are covered elsewhere).
    from src.data.news_source import NewsItem

    good = NewsItem(title="Infosys wins a large cloud deal in Europe today", publisher="Mint",
                    url="https://x/good", published="2026-07-08", source_id="news_google")
    no_key = NewsItem(title="", publisher="", url="", published="2026-07-08",
                      source_id="news_google")            # blank title AND url -> empty item_key
    bad_source = NewsItem(title="Adani ports posts record cargo volume this quarter",
                          publisher="ET", url="https://x/adani", published="2026-07-08",
                          source_id="")                    # empty source_id -> core rejects hard

    class StubSource:
        def fetch(self, symbol, company_name=""):
            return [good, no_key, bad_source]

    log = IngestionLog(tmp_path / "events.jsonl", clock=_clock)
    summary = ingest_news(log, StubSource(), symbol="X", cluster_threshold=0.5)
    assert summary.errors == 2          # one no-key + one hard-rejected
    assert summary.new == 1             # the good item survived the bad neighbours
    assert len(log.current()) == 1
    assert log.current()[0].source_id == "news_google"
