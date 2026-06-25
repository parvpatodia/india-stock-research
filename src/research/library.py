"""Ingest the owner's documents into a grounded DocumentStore.

Files are matched to sources by filename stem: documents/acme_ar_fy24.pdf is ingested under
the source id 'acme_ar_fy24', which MUST already exist in the registry (config/sources.yaml).
A file whose stem is not a registered source is skipped and reported, never ingested under
an untiered id, because the tier is what decides whether its text can back a fact.
"""
from __future__ import annotations

from pathlib import Path

from ..sources.registry import SourceRegistry
from .grounding import DocumentStore

_SUPPORTED = {".txt", ".md", ".pdf"}


def load_document_text(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"unsupported document type: {suffix}")


def build_library(registry: SourceRegistry, documents_dir: str | Path,
                  store: DocumentStore | None = None) -> tuple[DocumentStore, list[str]]:
    """Build a DocumentStore from every supported file in documents_dir.

    Returns (store, skipped) where skipped lists filenames whose stem is not a registered
    source. The store is registry-bound, so ingestion of an unregistered id cannot happen.
    """
    store = store or DocumentStore(registry=registry)
    skipped: list[str] = []
    directory = Path(documents_dir)
    if not directory.is_dir():
        return store, skipped
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
            continue
        source_id = path.stem
        if registry.get(source_id) is None:
            skipped.append(path.name)
            continue
        text = load_document_text(path)
        if text.strip():
            store.add_document(source_id, text, locator_prefix=path.name)
    return store, skipped
