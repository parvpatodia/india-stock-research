from src.research.grounding import DocumentStore
from src.sources.adapters import HttpDocumentAdapter, ingest_documents
from src.sources.registry import CredibilityTier, Source, SourceRegistry


def test_text_document():
    a = HttpDocumentAdapter("acme_ar", fetcher=lambda url: (b"Revenue was 100 cr", "text/plain"))
    docs = a.fetch("http://x/ar.txt")
    assert len(docs) == 1
    assert docs[0].text == "Revenue was 100 cr"
    assert docs[0].source_id == "acme_ar"


def test_html_is_tag_stripped_and_unescaped():
    raw = b"<html><body>Revenue <b>100</b> cr &amp; rising</body></html>"
    a = HttpDocumentAdapter("acme_ar", fetcher=lambda url: (raw, "text/html"))
    text = a.fetch("http://x")[0].text
    assert "Revenue 100 cr & rising" in " ".join(text.split())


def test_empty_content_returns_no_docs():
    a = HttpDocumentAdapter("s", fetcher=lambda url: (b"   ", "text/plain"))
    assert a.fetch("http://x") == []


def test_ingest_only_into_registered_source():
    reg = SourceRegistry([Source("acme_ar", "Acme AR", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    a = HttpDocumentAdapter("acme_ar", fetcher=lambda url: (b"Revenue was 100 cr in FY24", ""))
    assert ingest_documents(store, a.fetch("http://x/ar.txt")) == 1
    assert "acme_ar" in store.source_ids()


def test_ingest_unregistered_source_is_blocked():
    import pytest
    reg = SourceRegistry([])  # nothing registered
    store = DocumentStore(registry=reg)
    a = HttpDocumentAdapter("rogue", fetcher=lambda url: (b"untiered text here", ""))
    with pytest.raises(ValueError):
        ingest_documents(store, a.fetch("http://x/doc.txt"))
