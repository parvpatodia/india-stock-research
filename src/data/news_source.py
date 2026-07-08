"""Recent-news context sources.

News is CONTEXT ONLY. It is fetched live, attributed to the underlying publisher, and dated.
It enters the system at the ANALYST tier (reputable aggregated press), so the registry +
citation contract guarantee it can never back a verified fact or move a number: only a PRIMARY
source is `citable_as_fact`, and `enforce_citations` downgrades any "fact" that isn't backed by
one. News answers "what is being reported right now", never "what is true".

Feeds: Google News RSS (India edition) and yfinance's news list. Fetchers are injected so the
parsing is tested fully offline; the live HTTP/yfinance calls are the default fallbacks.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Callable

from ..sources.adapters import FetchedDocument
from ..sources.registry import CredibilityTier, Source, SourceRegistry

# Registered feed ids. ANALYST tier: attributed reporting, never a bare fact or a number.
GOOGLE_NEWS_SOURCE_ID = "news_google"
YAHOO_NEWS_SOURCE_ID = "news_yahoo"

NEWS_SOURCES: tuple[Source, ...] = (
    Source(id=GOOGLE_NEWS_SOURCE_ID, name="Google News (aggregated Indian press)",
           tier=CredibilityTier.ANALYST, url="https://news.google.com",
           notes="Live headline aggregator. Context only; attributed to the underlying "
                 "publisher and dated. Never the basis for a fact or a number."),
    Source(id=YAHOO_NEWS_SOURCE_ID, name="Yahoo Finance news",
           tier=CredibilityTier.ANALYST, url="https://finance.yahoo.com",
           notes="Live finance news feed. Context only; attributed and dated."),
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_KEY = re.compile(r"[^a-z0-9]+")


def _clean(text: str | None) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _to_iso_date(value: str | None, rfc822: bool = False) -> str:
    """Return YYYY-MM-DD, or "" if the date can't be parsed. rfc822 for RSS pubDate."""
    if not value:
        return ""
    try:
        if rfc822:
            dt = parsedate_to_datetime(value)
        else:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except (TypeError, ValueError):
        return ""


@dataclass(frozen=True)
class NewsItem:
    title: str
    publisher: str
    url: str
    published: str          # ISO date (YYYY-MM-DD) or "" if unknown
    summary: str = ""
    source_id: str = GOOGLE_NEWS_SOURCE_ID

    @property
    def as_text(self) -> str:
        """The document text ingested for grounding: attribution + date up front so any answer
        the model draws from it carries the publisher and 'as of' date."""
        pub = self.publisher or "unknown source"
        date = self.published or "undated"
        body = self.title.rstrip(".") + "."
        if self.summary:
            body += f" {self.summary}"
        return f"[{pub}, {date}] {body}"


def parse_google_news_rss(xml_bytes: bytes,
                          source_id: str = GOOGLE_NEWS_SOURCE_ID) -> list[NewsItem]:
    """Parse a Google News RSS feed. Google appends ' - Publisher' to each title and carries
    the publisher in a <source> element; we strip the suffix and use <source> for attribution."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items: list[NewsItem] = []
    for it in root.findall(".//item"):
        title = _clean(it.findtext("title"))
        if not title:
            continue
        source_el = it.find("source")
        publisher = _clean(source_el.text) if source_el is not None else ""
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")].strip()
        link = (it.findtext("link") or "").strip()
        published = _to_iso_date(it.findtext("pubDate"), rfc822=True)
        summary = _clean(it.findtext("description"))
        items.append(NewsItem(title, publisher, link, published, summary, source_id))
    return items


def parse_yahoo_news(raw: list[dict],
                     source_id: str = YAHOO_NEWS_SOURCE_ID) -> list[NewsItem]:
    """Parse yfinance's Ticker.news list. Each entry is {id, content{...}}."""
    items: list[NewsItem] = []
    for entry in raw or []:
        content = entry.get("content") if isinstance(entry, dict) else None
        if not isinstance(content, dict):
            continue
        title = _clean(content.get("title"))
        if not title:
            continue
        provider = content.get("provider")
        publisher = _clean(provider.get("displayName")) if isinstance(provider, dict) else ""
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            candidate = content.get(key)
            if isinstance(candidate, dict) and candidate.get("url"):
                url = candidate["url"]
                break
        published = _to_iso_date(content.get("pubDate") or content.get("displayTime"))
        summary = _clean(content.get("summary") or content.get("description"))
        items.append(NewsItem(title, publisher, url, published, summary, source_id))
    return items


def _dedup_sort_cap(items: list[NewsItem], cap: int) -> list[NewsItem]:
    """Drop duplicate headlines, sort newest first (undated last), cap the count."""
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in items:
        key = _KEY.sub("", it.title.lower())[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    unique.sort(key=lambda i: i.published or "", reverse=True)
    return unique[:cap]


def _google_query(company_name: str, symbol: str) -> str:
    return f"{(company_name or symbol).strip()} stock"


class NewsSource:
    """Aggregate recent news from Google News RSS + yfinance as dated, attributed context.

    Both fetchers are injected so tests never touch the network; one feed failing never takes
    down the other or the page (mirrors the provider degrade-to-missing contract).
    """

    def __init__(self,
                 rss_fetcher: Callable[[str], bytes] | None = None,
                 yahoo_fetcher: Callable[[str], list[dict]] | None = None,
                 max_items: int = 8):
        self._rss_fetcher = rss_fetcher or self._http_rss
        self._yahoo_fetcher = yahoo_fetcher or self._yahoo_api
        self.max_items = max_items

    @staticmethod
    def _http_rss(query: str) -> bytes:
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()

    @staticmethod
    def _yahoo_api(symbol: str) -> list[dict]:
        import yfinance as yf

        from .yfinance_provider import to_yahoo_symbol
        return yf.Ticker(to_yahoo_symbol(symbol)).news or []

    def fetch(self, symbol: str, company_name: str = "") -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            items += parse_google_news_rss(self._rss_fetcher(_google_query(company_name, symbol)))
        except Exception:
            pass
        try:
            items += parse_yahoo_news(self._yahoo_fetcher(symbol))
        except Exception:
            pass
        return _dedup_sort_cap(items, self.max_items)

    @staticmethod
    def as_documents(items: list[NewsItem]) -> list[FetchedDocument]:
        docs: list[FetchedDocument] = []
        for it in items:
            locator = f"{it.publisher or 'unknown'}, {it.published or 'undated'}"
            docs.append(FetchedDocument(it.source_id, it.as_text, url=it.url, locator=locator))
        return docs


def registry_with_news(base: SourceRegistry | None = None) -> SourceRegistry:
    """A registry that always includes the live news feeds, merged over the owner's config
    sources (if any). WHY: news is a system feed, not a curated document; it must be tiered so
    its text can be shown as attributed context but never promoted to a verified fact."""
    merged = SourceRegistry()
    if base is not None:
        for source in base.all_sources():
            merged.add(source)
    for source in NEWS_SOURCES:
        if merged.get(source.id) is None:
            merged.add(source)
    return merged
