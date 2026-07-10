from datetime import date

from src.data.news_source import (
    NEWS_SOURCES,
    NewsSource,
    parse_google_news_rss,
    parse_yahoo_news,
    registry_with_news,
)

_TODAY = date(2026, 7, 9)   # fixed reference so the recency filter is deterministic in tests
from src.research.claims import FACT, UNVERIFIED, Citation, Claim, ResearchResult, enforce_citations
from src.research.grounding import DocumentStore
from src.sources.adapters import ingest_documents
from src.sources.registry import CredibilityTier, Source, SourceRegistry

# Shapes below mirror the real feeds (Google News RSS 2.0; yfinance Ticker.news).

GOOGLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>
<item>
  <title>Reliance shares slip after SEBI warning - India Infoline</title>
  <link>https://news.google.com/rss/articles/ABC?oc=5</link>
  <pubDate>Wed, 08 Jul 2026 06:40:49 GMT</pubDate>
  <description>&lt;a href="x"&gt;Reliance shares slip&lt;/a&gt; on the news.</description>
  <source url="https://www.indiainfoline.com">India Infoline</source>
</item>
<item>
  <title>Reliance Q1 earnings preview - Moneycontrol</title>
  <link>https://news.google.com/rss/articles/DEF?oc=5</link>
  <pubDate>Tue, 07 Jul 2026 03:00:00 GMT</pubDate>
  <description>Preview ahead of results.</description>
  <source url="https://www.moneycontrol.com">Moneycontrol</source>
</item>
</channel></rss>"""

YAHOO_NEWS = [
    {"id": "1", "content": {
        "title": "Meta builds India data center with Reliance",
        "summary": "Meta is partnering with Reliance Industries on a data center.",
        "description": "",
        "pubDate": "2026-06-22T05:09:16Z",
        "provider": {"displayName": "Simply Wall St."},
        "canonicalUrl": {"url": "https://finance.yahoo.com/x.html"},
    }},
    {"id": "2", "content": {"title": "", "provider": {}}},   # empty title -> dropped
    {"garbage": True},                                        # no content -> dropped
]


def test_parse_google_rss_strips_publisher_suffix_and_dates():
    items = parse_google_news_rss(GOOGLE_RSS)
    assert len(items) == 2
    first = items[0]
    assert first.title == "Reliance shares slip after SEBI warning"   # " - Publisher" stripped
    assert first.publisher == "India Infoline"                        # from <source>
    assert first.published == "2026-07-08"                            # RFC822 -> ISO
    assert first.url.startswith("https://news.google.com/rss/articles/ABC")
    assert first.summary == "Reliance shares slip on the news."       # HTML stripped/unescaped


def test_parse_yahoo_news_extracts_and_drops_empty():
    items = parse_yahoo_news(YAHOO_NEWS)
    assert len(items) == 1                                            # empty title + garbage dropped
    it = items[0]
    assert it.title == "Meta builds India data center with Reliance"
    assert it.publisher == "Simply Wall St."
    assert it.url == "https://finance.yahoo.com/x.html"
    assert it.published == "2026-06-22"
    assert "partnering with Reliance" in it.summary


def test_parse_garbage_returns_empty():
    assert parse_google_news_rss(b"not xml at all") == []
    assert parse_yahoo_news([]) == []
    assert parse_yahoo_news(None) == []


def test_newssource_aggregates_dedups_and_sorts_newest_first():
    # yahoo repeats the top google headline (must dedup) and adds an older dated item.
    yahoo = [{"id": "d", "content": {
        "title": "Reliance shares slip after SEBI warning",   # duplicate of google item 1
        "pubDate": "2026-07-01T00:00:00Z",
        "provider": {"displayName": "Yahoo"},
        "canonicalUrl": {"url": "https://y/dup.html"}}}] + YAHOO_NEWS
    ns = NewsSource(rss_fetcher=lambda q: GOOGLE_RSS, yahoo_fetcher=lambda s: yahoo, max_items=8,
                    today=_TODAY)
    items = ns.fetch("RELIANCE", "Reliance Industries")
    assert [i.title for i in items] == [
        "Reliance shares slip after SEBI warning",   # 2026-07-08 (google kept, yahoo dup dropped)
        "Reliance Q1 earnings preview",              # 2026-07-07
        "Meta builds India data center with Reliance",  # 2026-06-22
    ]
    assert items[0].published == "2026-07-08"


def test_newssource_skips_google_search_without_a_verified_company_name():
    # WHY (real money, honesty): live-verified that several real NSE tickers are common English
    # words (PAGE = Page Industries; also IDEA, SAIL, RAIN) -- yfinance's own name resolution can
    # fail (confirmed live for PAGE: a 404 on fundamentals -> name=None), and searching Google News
    # for the BARE TICKER as free text then pulls in wildly unrelated results (confirmed live:
    # "PAGE stock" returned a Rocket Lab article matched on the unrelated idiom "takes a page out
    # of...", mixed indistinguishably with genuine Page Industries news). Only search Google News
    # with a genuinely resolved company name; an unresolved name must skip the search entirely,
    # not silently substitute the ticker as if it were safe free text.
    rss_calls = []

    def rss_fetcher(query):
        rss_calls.append(query)
        return GOOGLE_RSS

    ns = NewsSource(rss_fetcher=rss_fetcher, yahoo_fetcher=lambda s: [], today=_TODAY)
    ns.fetch("PAGE", "")                          # no resolved company name
    assert rss_calls == []                        # Google search skipped entirely

    ns.fetch("RELIANCE", "Reliance Industries")    # a genuinely resolved name
    assert len(rss_calls) == 1                     # Google search proceeds normally


def test_newssource_caps_and_survives_one_feed_failing():
    def boom(_):
        raise RuntimeError("feed down")
    ns = NewsSource(rss_fetcher=lambda q: GOOGLE_RSS, yahoo_fetcher=boom, max_items=1, today=_TODAY)
    items = ns.fetch("RELIANCE", "Reliance Industries")   # a resolved name -> google search runs
    assert len(items) == 1                       # yahoo failed, google still returned; capped to 1
    assert items[0].published == "2026-07-08"


def test_newssource_drops_stale_items_beyond_max_age():
    # WHY (honesty): news shows under "Recent news". A dated item older than the max-age window is
    # not recent and must not appear as current context for a real-money decision.
    stale = [{"id": "old", "content": {
        "title": "Reliance ancient headline from years ago",
        "pubDate": "2023-01-01T00:00:00Z",                 # ~3.5 years before _TODAY
        "provider": {"displayName": "OldWire"},
        "canonicalUrl": {"url": "https://y/old.html"}}}]
    ns = NewsSource(rss_fetcher=lambda q: GOOGLE_RSS, yahoo_fetcher=lambda s: stale,
                    max_items=8, max_age_days=365, today=_TODAY)
    titles = [i.title for i in ns.fetch("RELIANCE", "Reliance Industries")]
    assert "Reliance ancient headline from years ago" not in titles       # stale dropped
    assert "Reliance shares slip after SEBI warning" in titles            # recent kept


def test_newssource_keeps_undated_items():
    # An undated item can't be shown to be stale; keep it (it's labeled 'undated', not 'recent').
    undated = [{"id": "u", "content": {
        "title": "Reliance undated note",
        "provider": {"displayName": "X"},
        "canonicalUrl": {"url": "https://y/u.html"}}}]                     # no pubDate -> undated
    ns = NewsSource(rss_fetcher=lambda q: b"<rss></rss>", yahoo_fetcher=lambda s: undated,
                    max_items=8, max_age_days=365, today=_TODAY)
    titles = [i.title for i in ns.fetch("X", "X")]
    assert "Reliance undated note" in titles


def test_as_documents_carry_attribution_and_source_id():
    items = parse_google_news_rss(GOOGLE_RSS)
    docs = NewsSource.as_documents(items)
    assert len(docs) == 2
    assert docs[0].source_id == "news_google"
    assert docs[0].text.startswith("[India Infoline, 2026-07-08]")
    assert docs[0].url.startswith("https://news.google.com")


def test_news_sources_are_analyst_tier_never_fact():
    for s in NEWS_SOURCES:
        assert s.tier == CredibilityTier.ANALYST
        assert s.citable_as_fact is False


def test_registry_with_news_merges_and_accepts_ingestion():
    base = SourceRegistry([Source("acme_ar", "Acme AR", CredibilityTier.PRIMARY)])
    reg = registry_with_news(base)
    assert reg.get("acme_ar").tier == CredibilityTier.PRIMARY      # base preserved
    assert reg.get("news_google").tier == CredibilityTier.ANALYST  # news merged in
    store = DocumentStore(registry=reg)
    docs = NewsSource.as_documents(parse_google_news_rss(GOOGLE_RSS))
    assert ingest_documents(store, docs) == 2                      # news ingests fine
    assert "news_google" in store.source_ids()


def test_news_citation_cannot_become_a_verified_fact():
    # WHY: even if the model labels a news-derived claim "fact", enforce_citations downgrades it.
    news_cite = Citation("news_google", CredibilityTier.ANALYST, "India Infoline, 2026-07-08")
    claim = Claim(text="Reliance profit was 100cr", citations=(news_cite,), kind=FACT)
    assert claim.is_verified_fact is False
    fixed = enforce_citations(ResearchResult("q", (claim,)))
    assert fixed.claims[0].kind == UNVERIFIED
