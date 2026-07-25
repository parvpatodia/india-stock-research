"""Structured ingestion + retrieval wiring: the DocumentStore can ingest a document element-
aware (tables kept intact, rich metadata) and every retrieved chunk carries its typed numeric
records so a grounded answer can cite a number to its exact record/locator. The existing blind
retrieve+abstain contract is unchanged."""
from src.research.grounding import (
    ChunkMetadata,
    DocumentStore,
    find_record,
    numeric_records,
)
from src.research.numeric_records import NumericRecord
from src.sources.registry import CredibilityTier, Source, SourceRegistry

AR = """Management Discussion and Analysis
Revenue grew across segments with a healthy order book this year.

Statement of Profit and Loss
(Rs in crore)
Particulars                 FY2024     FY2023
Revenue from operations    9,00,000   7,92,000
Net profit for the year      73,670     66,700
"""


def _registry():
    return SourceRegistry([Source("annual_report", "Annual report", CredibilityTier.PRIMARY)])


def test_structured_ingestion_preserves_table_and_metadata():
    store = DocumentStore(registry=_registry())
    n = store.add_document("annual_report", AR, locator_prefix="AR", structured=True,
                           company="RELIANCE", symbol="RELIANCE", doc_type="annual_report")
    assert n >= 3
    hits = store.retrieve("net profit for the year", k=5)
    table_hits = [h for h in hits if h.chunk.metadata.element_kind == "table"]
    assert table_hits, "the P&L table should be retrievable as an intact chunk"
    meta = table_hits[0].chunk.metadata
    assert meta.unit_scale == "crore"
    assert meta.currency == "INR"
    assert meta.fiscal_period == "FY2024"
    assert meta.company == "RELIANCE"
    assert meta.doc_type == "annual_report"


def test_retrieval_carries_typed_records_citable_to_locator():
    store = DocumentStore(registry=_registry())
    store.add_document("annual_report", AR, structured=True, company="RELIANCE")
    hits = store.retrieve("net profit", k=5)
    recs = numeric_records(hits)
    assert recs, "structured retrieval must surface typed numeric records"
    npr = find_record("73,670", recs)
    assert npr is not None
    assert npr.scale == "crore"
    assert npr.absolute_value == 73670.0 * 1e7      # normalized deterministically
    assert npr.source_doc == "annual_report"
    assert npr.locator and npr.locator.startswith("annual_report#")
    # no record -> no numeric claim: a figure absent from the filing resolves to nothing
    assert find_record("123456", recs) is None


def test_blind_mode_unchanged_and_new_fields_have_safe_defaults():
    store = DocumentStore(registry=_registry())
    store.add_document("annual_report", "Revenue was strong this year. " * 30)
    hits = store.retrieve("revenue")
    assert hits
    chunk = hits[0].chunk
    assert isinstance(chunk.metadata, ChunkMetadata)
    assert chunk.records == ()                       # no numeric figures in this prose -> none


def test_blind_mode_still_extracts_inline_unit_bearing_records():
    store = DocumentStore(registry=_registry())
    store.add_document("annual_report", "Net profit was Rs 73,670 crore and ROE was 22.5%.")
    recs = numeric_records(store.retrieve("net profit ROE"))
    assert any(r.scale == "crore" and r.absolute_value == 73670.0 * 1e7 for r in recs)
    assert any(r.unit == "percent" and r.value == 22.5 for r in recs)
    assert all(isinstance(r, NumericRecord) for r in recs)


def test_structured_ingestion_respects_registry_gate():
    # an untiered source is still refused at ingestion, structured or not
    store = DocumentStore(registry=_registry())
    try:
        store.add_document("unknown_src", AR, structured=True)
        assert False, "expected a registry rejection"
    except ValueError:
        pass
