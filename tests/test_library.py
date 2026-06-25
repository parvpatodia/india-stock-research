import pytest

from src.research.library import build_library, load_document_text
from src.sources.registry import CredibilityTier, Source, SourceRegistry


def test_load_txt(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world")
    assert load_document_text(p) == "hello world"


def test_unsupported_type_raises(tmp_path):
    p = tmp_path / "x.docx"
    p.write_text("z")
    with pytest.raises(ValueError):
        load_document_text(p)


def test_build_library_ingests_registered_and_skips_unregistered(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "acme_ar_fy24.txt").write_text("Revenue was Rs 974000 crore in FY2024.")
    (docs / "rogue_source.txt").write_text("text from a source nobody tiered")
    (docs / "image.png").write_bytes(b"\x89PNG")  # unsupported type, ignored
    reg = SourceRegistry([Source("acme_ar_fy24", "Acme AR", CredibilityTier.PRIMARY)])

    store, skipped, failed = build_library(reg, docs)

    assert "acme_ar_fy24" in store.source_ids()
    assert "rogue_source.txt" in skipped
    assert "rogue_source" not in store.source_ids()  # untiered id never ingested
    assert failed == []
    assert len(store) >= 1


def test_build_library_missing_dir_is_empty(tmp_path):
    store, skipped, failed = build_library(SourceRegistry([]), tmp_path / "nope")
    assert len(store) == 0 and skipped == [] and failed == []


def test_build_library_unreadable_file_degrades_not_crashes(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "acme_ar_fy24.txt").write_text("Revenue was Rs 974000 crore in FY2024.")
    (docs / "acme_ar_fy24_v2.pdf").write_bytes(b"")  # empty/corrupt PDF -> pypdf raises
    reg = SourceRegistry([
        Source("acme_ar_fy24", "Acme AR", CredibilityTier.PRIMARY),
        Source("acme_ar_fy24_v2", "Acme AR v2", CredibilityTier.PRIMARY),
    ])

    store, skipped, failed = build_library(reg, docs)

    assert "acme_ar_fy24" in store.source_ids()       # good file still ingested
    assert "acme_ar_fy24_v2.pdf" in failed            # bad file degraded, did not crash
    assert "acme_ar_fy24_v2" not in store.source_ids()
