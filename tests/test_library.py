import pytest

from src.research.library import (
    build_library,
    load_document_text,
    parse_demo_enabled_secret,
    resolve_curated_library_paths,
)
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


def test_resolve_curated_library_paths_prefers_real_config(tmp_path):
    real_yaml = tmp_path / "config" / "sources.yaml"
    real_yaml.parent.mkdir()
    real_yaml.write_text("primary: []")
    real_docs = tmp_path / "documents"
    sample_yaml = tmp_path / "sample_data" / "sources.yaml"
    sample_docs = tmp_path / "sample_data" / "documents"

    yaml_path, docs_path = resolve_curated_library_paths(
        real_yaml, real_docs, sample_yaml, sample_docs, demo_enabled=False)
    assert yaml_path == real_yaml and docs_path == real_docs


def test_resolve_curated_library_paths_no_real_config_and_demo_disabled_is_honest_empty(tmp_path):
    # WHY (real money, HIGH severity, live-verified): config/sources.yaml is gitignored, so it
    # can never exist in this app's git-based Streamlit Cloud deployment -- without this gate,
    # EVERY deployed session would silently load synthetic sample data (Acme Industries, XYZ
    # Fund) into the Ask tab's curated library, indistinguishable from a real one. Live-verified
    # that this fake data scores well above the retrieval floor against ordinary questions about
    # a REAL stock, surfacing an unrelated fake company's data in a real answer. When there's no
    # real config AND demo mode isn't explicitly enabled, the caller must get back the (missing)
    # real-config path, so downstream code sees "no curated library" -- an honest empty state --
    # never a silent substitution.
    real_yaml = tmp_path / "config" / "sources.yaml"           # does not exist
    real_docs = tmp_path / "documents"
    sample_yaml = tmp_path / "sample_data" / "sources.yaml"
    sample_docs = tmp_path / "sample_data" / "documents"

    yaml_path, docs_path = resolve_curated_library_paths(
        real_yaml, real_docs, sample_yaml, sample_docs, demo_enabled=False)
    assert yaml_path == real_yaml and docs_path == real_docs
    assert not yaml_path.exists()


def test_resolve_curated_library_paths_falls_back_to_sample_only_when_demo_enabled(tmp_path):
    real_yaml = tmp_path / "config" / "sources.yaml"           # does not exist
    real_docs = tmp_path / "documents"
    sample_yaml = tmp_path / "sample_data" / "sources.yaml"
    sample_docs = tmp_path / "sample_data" / "documents"

    yaml_path, docs_path = resolve_curated_library_paths(
        real_yaml, real_docs, sample_yaml, sample_docs, demo_enabled=True)
    assert yaml_path == sample_yaml and docs_path == sample_docs


def test_parse_demo_enabled_secret_handles_real_booleans():
    assert parse_demo_enabled_secret(True) is True
    assert parse_demo_enabled_secret(False) is False
    assert parse_demo_enabled_secret(None) is False


def test_parse_demo_enabled_secret_rejects_a_mistakenly_quoted_false_string():
    # WHY (real money, adversarial-review finding): an adversarial review flagged that
    # bool(_secret("demo_sample_library", False)) would read a mistakenly-quoted TOML string
    # `demo_sample_library = "false"` as True -- Python's bare bool() treats ANY non-empty string
    # as truthy, so writing exactly what looks like "turn this off" would silently RE-ENABLE the
    # sample/demo document fallback this secret exists to gate off by default (see
    # resolve_curated_library_paths / sample_data/sources.yaml's citability fix). Confirmed live:
    # bool("false") is True in plain Python.
    assert parse_demo_enabled_secret("false") is False
    assert parse_demo_enabled_secret("False") is False
    assert parse_demo_enabled_secret("FALSE") is False
    assert parse_demo_enabled_secret("no") is False
    assert parse_demo_enabled_secret("0") is False
    assert parse_demo_enabled_secret("") is False
    assert parse_demo_enabled_secret("off") is False


def test_parse_demo_enabled_secret_accepts_a_quoted_true_string():
    assert parse_demo_enabled_secret("true") is True
    assert parse_demo_enabled_secret("True") is True
    assert parse_demo_enabled_secret("1") is True
    assert parse_demo_enabled_secret("yes") is True
