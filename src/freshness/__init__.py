"""Freshness engine (SPEC v4 W1): ingestion event log, content-hash dedup, near-duplicate news
clustering, and as-of/staleness dating. Cross-source freshness infrastructure that sits above the
per-source fetchers in src/data/; each source feeds the same append-only log."""
from .clustering import cluster_items, normalize_title, title_tokens, token_similarity
from .event_log import IngestionEvent, IngestionLog, IngestResult, content_hash
from .filings_ingest import (
    ANNUAL_REPORT_SOURCE_ID,
    AnnualReportIngestResult,
    FilingsIngestSummary,
    ar_item_key,
    ingest_announcements,
    ingest_annual_report,
)
from .news_ingest import IngestSummary, ingest_news
from .staleness import Freshness, describe_freshness, freshness, parse_iso_date

__all__ = [
    "IngestionLog", "IngestionEvent", "IngestResult", "content_hash",
    "cluster_items", "normalize_title", "title_tokens", "token_similarity",
    "freshness", "describe_freshness", "parse_iso_date", "Freshness",
    "ingest_news", "IngestSummary",
    "ingest_announcements", "ingest_annual_report", "FilingsIngestSummary",
    "AnnualReportIngestResult", "ar_item_key", "ANNUAL_REPORT_SOURCE_ID",
]
