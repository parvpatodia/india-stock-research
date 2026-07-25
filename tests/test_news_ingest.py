from datetime import datetime, timezone

from src.data.news_source import NewsSource
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
