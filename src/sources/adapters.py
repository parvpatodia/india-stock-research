"""Source adapters: the pluggable seam for getting documents in.

There is no single clean API for all Indian AGM transcripts and annual reports, so this is an
interface, not a hard-coded integration. A concrete adapter turns a reference (a URL now, an
owner-provided API endpoint later) into text tagged with its source id, which then flows
through the SAME grounding + verification path as everything else. Fetched text is only ever
ingested under a source that is registered and tiered; nothing bypasses the trust boundary.
"""
from __future__ import annotations

import html
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..research.grounding import DocumentStore

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class FetchedDocument:
    source_id: str
    text: str
    url: str
    locator: str = ""


class DocumentSourceAdapter(ABC):
    @abstractmethod
    def fetch(self, ref: str) -> list[FetchedDocument]:
        """Turn a reference (URL / id) into fetched documents. Empty list if nothing usable."""


class HttpDocumentAdapter(DocumentSourceAdapter):
    """Fetch a document from a URL and extract text (PDF via pypdf, HTML tag-stripped, else
    decoded). The network call is injectable so the parsing is testable offline."""

    def __init__(self, source_id: str,
                 fetcher: Callable[[str], tuple[bytes, str]] | None = None):
        self.source_id = source_id
        self._fetcher = fetcher or self._http_fetch

    @staticmethod
    def _http_fetch(url: str) -> tuple[bytes, str]:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read(), (resp.headers.get_content_type() or "")

    def fetch(self, ref: str) -> list[FetchedDocument]:
        raw, content_type = self._fetcher(ref)
        text = self._to_text(raw, content_type, ref)
        if not text.strip():
            return []
        return [FetchedDocument(self.source_id, text, url=ref, locator=ref)]

    def _to_text(self, raw: bytes, content_type: str, url: str) -> str:
        ct = content_type.lower()
        if "pdf" in ct or url.lower().endswith(".pdf"):
            # WHY (H5): pdfplumber keeps a page's TABLES as real grids so their scaled cells stay
            # typed numeric records; pypdf's flat extract collapses columns and loses the table.
            # DEGRADE-SAFE (real money): the table extractor abstains (returns None) if pdfplumber
            # is missing or the PDF won't parse, so we always have the existing pypdf path to fall
            # back to and never crash on a Research fetch.
            from .pdf_tables import extract_pdf_tables_text
            tables_text = extract_pdf_tables_text(raw)
            if tables_text and tables_text.strip():
                return tables_text
            return self._pypdf_text(raw)
        decoded = raw.decode("utf-8", errors="replace")
        if "html" in ct or url.lower().endswith((".html", ".htm")):
            return _WS.sub(" ", html.unescape(_TAG.sub(" ", decoded))).strip()
        return decoded

    @staticmethod
    def _pypdf_text(raw: bytes) -> str:
        """The original flat PDF text extraction, kept as the degrade-safe fallback for the
        pdfplumber table path (missing dependency / unparseable PDF)."""
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def ingest_documents(store: DocumentStore, docs: list[FetchedDocument]) -> int:
    """Ingest fetched docs into the store. The store is registry-bound, so an unregistered
    source_id raises there; fetched content cannot skip the trust boundary."""
    n = 0
    for doc in docs:
        if doc.text.strip():
            store.add_document(doc.source_id, doc.text, locator_prefix=doc.locator)
            n += 1
    return n
