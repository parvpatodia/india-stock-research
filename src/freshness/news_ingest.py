"""Wire the live news RSS feed into the ingestion log (W1 increment 1's one real source).

Flow: NewsSource (Google News RSS + yfinance, both fetchers injectable/offline in tests) ->
cluster near-duplicate rewrites -> record ONE event per cluster in the append-only log with its
content hash, publisher/date attribution, and as-of date. Dedup + supersession fall out of the
log. News stays ANALYST-tier context: the log records provenance, it does not promote anything to
a fact.

Degrade, never crash (LESSONS 2026-07-09): NewsSource.fetch already abstains to [] on a feed
failure; this adds a second guard around the whole fetch and a per-item guard so one malformed
headline cannot take down the batch (LESSONS 2026-06-18, per-item ingestion must degrade).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.news_source import NewsItem, NewsSource
from .clustering import cluster_items, normalize_title
from .event_log import IngestionLog


@dataclass
class IngestSummary:
    fetched: int = 0        # news items pulled from the feeds
    clusters: int = 0       # near-duplicate clusters after collapsing rewrites
    new: int = 0            # first-seen items appended
    superseded: int = 0     # items whose content changed (prior version superseded)
    skipped: int = 0        # dedup no-ops (identical content already recorded)
    errors: int = 0         # items that could not be recorded (skipped, batch survived)


def _representative(cluster: list[NewsItem]) -> NewsItem:
    """The freshest item in a near-duplicate cluster: newest published date (undated sorts last),
    ties broken by feed order. One representative event stands in for the whole cluster."""
    return max(cluster, key=lambda it: it.published or "")


def _item_key(item: NewsItem) -> str:
    """A stable logical identity for a story: its normalized title (so a later reworded UPDATE of
    the same headline supersedes), falling back to the URL when the title is content-free."""
    key = normalize_title(item.title)[:80]
    return key or (item.url or "").strip()


def ingest_news(log: IngestionLog, source: NewsSource, symbol: str, company_name: str = "",
                cluster_threshold: float = 0.5) -> IngestSummary:
    """Fetch, cluster, and record recent news for a symbol into the log. Returns a run summary."""
    summary = IngestSummary()
    try:
        items = source.fetch(symbol, company_name)
    except Exception:
        # NewsSource already degrades internally; this is the outer abstain-on-failure guard.
        return summary
    summary.fetched = len(items)

    clusters = cluster_items(items, key=lambda it: it.title, threshold=cluster_threshold)
    summary.clusters = len(clusters)

    for cluster in clusters:
        rep = _representative(cluster)
        key = _item_key(rep)
        if not key:
            summary.errors += 1     # no usable identity (blank title AND url) -> skip, survive
            continue
        try:
            result = log.ingest(item_key=key, source_id=rep.source_id, content=rep.as_text,
                                as_of=rep.published, kind="news", ref=rep.url, title=rep.title,
                                cluster_id=key)
        except ValueError:
            # Reject-hard at the core, degrade-per-item at the batch: a bad record is skipped, the
            # rest of the run continues.
            summary.errors += 1
            continue
        if result.action == "new":
            summary.new += 1
        elif result.action == "superseded":
            summary.superseded += 1
        else:
            summary.skipped += 1
    return summary
