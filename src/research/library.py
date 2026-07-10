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

_FALSY_SECRET_STRINGS = {"", "0", "false", "no", "off"}


def parse_demo_enabled_secret(value) -> bool:
    """Coerce a Streamlit secret value into a strict boolean for the demo_sample_library gate.

    WHY (real money, adversarial-review finding): a bare `bool(value)` would read a mistakenly-
    quoted TOML string `demo_sample_library = "false"` as True -- Python's bool() treats ANY
    non-empty string as truthy, so writing exactly what looks like "turn this off" would
    silently RE-ENABLE the sample/demo document fallback this secret exists to keep off by
    default (see resolve_curated_library_paths / sample_data/sources.yaml's citability fix).
    A real TOML/Python boolean still passes straight through via bool(value).
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY_SECRET_STRINGS
    return bool(value)


def resolve_curated_library_paths(config_yaml: Path, docs_dir: Path, sample_yaml: Path,
                                  sample_docs_dir: Path,
                                  demo_enabled: bool) -> tuple[Path, Path]:
    """Decide which sources.yaml/documents dir the curated library should load.

    Prefers the owner's real config if it exists. Otherwise falls back to the bundled
    sample/demo library ONLY when explicitly opted in (demo_enabled).

    WHY (real money, HIGH severity, live-verified): config/sources.yaml is gitignored, so it can
    NEVER exist in this app's git-based Streamlit Cloud deployment. Without this gate, EVERY
    deployed session would silently load synthetic sample data ("Acme Industries", "XYZ Fund")
    into the Ask tab's curated library as if it were real -- live-verified that this fake data
    scores well above the retrieval floor against ordinary questions about a real stock,
    surfacing an unrelated, non-existent company's data in an answer about a real one. When
    neither the real config exists nor demo mode is enabled, this returns the (missing)
    config_yaml/docs_dir unchanged, so the caller ends up with "no curated library" -- an honest
    empty state (the Ask tab already works fine on news + verified figures without one) -- never
    a silent substitution.
    """
    if config_yaml.exists():
        return config_yaml, docs_dir
    if demo_enabled:
        return sample_yaml, sample_docs_dir
    return config_yaml, docs_dir


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
                  store: DocumentStore | None = None
                  ) -> tuple[DocumentStore, list[str], list[str]]:
    """Build a DocumentStore from every supported file in documents_dir.

    Returns (store, skipped, failed):
      skipped = filenames whose stem is not a registered source (untiered, not loaded).
      failed  = filenames that could not be read (corrupt/encrypted).
    The store is registry-bound, so ingestion of an unregistered id cannot happen, and one
    unreadable file is reported rather than crashing ingestion of the rest.
    """
    store = store or DocumentStore(registry=registry)
    skipped: list[str] = []
    failed: list[str] = []
    directory = Path(documents_dir)
    if not directory.is_dir():
        return store, skipped, failed
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
            continue
        source_id = path.stem
        if registry.get(source_id) is None:
            skipped.append(path.name)
            continue
        try:
            text = load_document_text(path)
        except Exception:
            # WHY: a single corrupt/encrypted file must not take down the whole library.
            failed.append(path.name)
            continue
        if text.strip():
            store.add_document(source_id, text, locator_prefix=path.name)
    return store, skipped, failed
