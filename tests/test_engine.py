import pytest

from src.instruments import InstrumentType
from src.research.claims import (
    FACT,
    OPINION,
    UNVERIFIED,
    Citation,
    Claim,
    ResearchResult,
    enforce_citations,
)
from src.research.grounded_analyst import _assemble_result, numbers_grounded
from src.research.grounding import Chunk, DocumentStore, RetrievedChunk
from src.sources.registry import CredibilityTier, Source, SourceRegistry


# --- G6 instruments ---

def test_instrument_taxonomy():
    assert {t.value for t in InstrumentType} == {"stock", "mutual_fund", "sip", "ipo", "other"}
    assert InstrumentType.MUTUAL_FUND.label == "Mutual fund"


# --- G1 sources ---

def test_citable_as_fact_only_primary():
    assert Source("a", "A", CredibilityTier.PRIMARY).citable_as_fact
    assert not Source("b", "B", CredibilityTier.ANALYST).citable_as_fact
    assert not Source("c", "C", CredibilityTier.CREATOR).citable_as_fact


def test_registry_duplicate_raises():
    with pytest.raises(ValueError):
        SourceRegistry([Source("a", "A", CredibilityTier.PRIMARY),
                        Source("a", "B", CredibilityTier.ANALYST)])


def test_registry_from_config(tmp_path):
    p = tmp_path / "sources.yaml"
    p.write_text(
        "primary:\n"
        "  - id: nse\n"
        "    name: NSE\n"
        "    url: https://nseindia.com\n"
        "creator:\n"
        "  - id: yt\n"
        "    name: Creator\n"
    )
    reg = SourceRegistry.from_config(p)
    assert len(reg) == 2
    assert reg.get("nse").citable_as_fact is True
    assert reg.get("yt").citable_as_fact is False
    assert reg.by_tier(CredibilityTier.PRIMARY)[0].id == "nse"


def test_registry_unknown_tier_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bogus:\n  - id: a\n    name: A\n")
    with pytest.raises(ValueError):
        SourceRegistry.from_config(p)


# --- G2 grounding ---

def test_retrieve_returns_relevant_and_abstains_on_no_match():
    store = DocumentStore(words_per_chunk=25, overlap=5)
    store.add_document(
        "amfi",
        "A mutual fund SIP invests a fixed amount every month into a scheme. "
        "NAV is the net asset value per unit of the fund.",
    )
    assert len(store) >= 1
    hits = store.retrieve("what is a SIP mutual fund", k=3)
    assert len(hits) >= 1
    assert hits[0].chunk.source_id == "amfi"
    assert store.retrieve("quantum chromodynamics gluon lattice", k=3) == []


def test_empty_store_retrieves_nothing():
    assert DocumentStore().retrieve("anything") == []


# --- G3 claims contract ---

def test_enforce_downgrades_unsourced_fact():
    claim = Claim("x", (Citation("yt", CredibilityTier.CREATOR, "v1"),), FACT)
    fixed = enforce_citations(ResearchResult("q", (claim,)))
    assert fixed.claims[0].kind == UNVERIFIED


# --- G4 assembly (the model's output is never trusted as-is) ---

def _setup_assembly():
    reg = SourceRegistry([
        Source("ar", "Annual Report", CredibilityTier.PRIMARY),
        Source("yt", "Creator", CredibilityTier.CREATOR),
    ])
    ar = Chunk("ar#0", "ar", "Revenue was 100 cr in FY24.", "p1")
    yt = Chunk("yt#0", "yt", "I think you should buy.", "vid1")
    retrieved = [RetrievedChunk(ar, 0.5), RetrievedChunk(yt, 0.4)]
    return reg, retrieved


def test_assemble_enforces_tiers_drops_hallucinated_and_mixed():
    reg, retrieved = _setup_assembly()
    payload = {"abstain": False, "claims": [
        {"text": "Revenue was 100 cr", "chunk_ids": ["ar#0"], "kind": "fact"},
        {"text": "Creator suggests buying", "chunk_ids": ["yt#0"], "kind": "fact"},
        {"text": "Ghost claim", "chunk_ids": ["ghost#9"], "kind": "fact"},
        {"text": "Mixed cite", "chunk_ids": ["ar#0", "yt#0"], "kind": "fact"},
        {"text": "Creator is bullish", "chunk_ids": ["yt#0"], "kind": "opinion"},
    ]}
    res = _assemble_result("q", payload, retrieved, reg, as_of="2026-06-18")
    assert not res.abstained
    by = {c.text: c for c in res.claims}
    assert "Ghost claim" not in by                               # H2: uncited claim dropped
    rev = by["Revenue was 100 cr"]
    assert rev.kind == FACT and rev.is_verified_fact             # primary-only fact stays
    assert rev.citations[0].as_of == "2026-06-18"
    assert by["Creator suggests buying"].kind == UNVERIFIED      # fact on creator -> downgraded
    assert by["Mixed cite"].kind == UNVERIFIED                   # H1: primary + creator is NOT a fact
    assert by["Creator is bullish"].kind == OPINION
    assert by["Creator is bullish"].citations[0].tier == CredibilityTier.CREATOR


def test_numbers_grounded_helper():
    src = ["Revenue was 1,234 cr and profit 957 cr in FY24."]
    assert numbers_grounded("Revenue was 1234 cr", src)             # digit-normalized match
    assert numbers_grounded("Profit was 957 cr", src)
    assert numbers_grounded("It grew about 5% last year", src)      # <3 digits ignored
    assert numbers_grounded("The outlook is positive", src)         # no number -> grounded
    assert not numbers_grounded("Profit was 9575 cr", src)          # 9575 not in source (no substring)
    assert not numbers_grounded("Revenue was 4000 cr", src)         # fabricated figure


def test_assemble_downgrades_fact_with_ungrounded_number():
    # WHY (real money): the model cites the right primary chunk but misquotes the figure. Tier is
    # fine, so tier-only checks pass it as a verified fact. Numeric grounding must catch it.
    reg = SourceRegistry([Source("ar", "Annual Report", CredibilityTier.PRIMARY)])
    ar = Chunk("ar#0", "ar", "Revenue was 100 cr in FY24.", "p1")
    retrieved = [RetrievedChunk(ar, 0.9)]
    payload = {"claims": [
        {"text": "Revenue was 100 cr", "chunk_ids": ["ar#0"], "kind": "fact"},   # correct
        {"text": "Revenue was 250 cr", "chunk_ids": ["ar#0"], "kind": "fact"},   # misquoted
    ]}
    by = {c.text: c for c in _assemble_result("q", payload, retrieved, reg, None).claims}
    assert by["Revenue was 100 cr"].is_verified_fact                # grounded number stays a fact
    assert by["Revenue was 250 cr"].kind == UNVERIFIED              # misquote never shows as ✓
    assert not by["Revenue was 250 cr"].is_verified_fact


def test_assemble_abstain_payload():
    reg, retrieved = _setup_assembly()
    res = _assemble_result("q", {"abstain": True, "reason": "nope"}, retrieved, reg, None)
    assert res.abstained and res.abstain_reason == "nope"


def test_assemble_all_uncited_abstains():
    reg, retrieved = _setup_assembly()
    payload = {"claims": [{"text": "ghost only", "chunk_ids": ["nope#1"], "kind": "fact"}]}
    res = _assemble_result("q", payload, retrieved, reg, None)
    assert res.abstained


def test_assemble_bad_claims_type_abstains():
    reg, retrieved = _setup_assembly()
    assert _assemble_result("q", {"claims": "not a list"}, retrieved, reg, None).abstained
    assert _assemble_result("q", {"claims": ["junk", 5]}, retrieved, reg, None).abstained


def test_documentstore_rejects_unregistered_source():
    reg = SourceRegistry([Source("ar", "AR", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    store.add_document("ar", "some text about revenue and earnings")
    with pytest.raises(ValueError):
        store.add_document("unknown_src", "text from a source nobody tiered")
