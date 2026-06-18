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
from src.research.grounded_analyst import _assemble_result
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


def test_assemble_enforces_tiers_and_drops_hallucinated_chunks():
    reg, retrieved = _setup_assembly()
    payload = {"abstain": False, "claims": [
        {"text": "Revenue was 100 cr", "chunk_ids": ["ar#0"], "kind": "fact"},
        {"text": "Creator suggests buying", "chunk_ids": ["yt#0"], "kind": "fact"},
        {"text": "Ghost claim", "chunk_ids": ["ghost#9"], "kind": "fact"},
        {"text": "Creator is bullish", "chunk_ids": ["yt#0"], "kind": "opinion"},
    ]}
    res = _assemble_result("q", payload, retrieved, reg, as_of="2026-06-18")
    assert not res.abstained
    c = list(res.claims)
    assert c[0].kind == FACT and c[0].is_verified_fact          # primary-backed fact stays
    assert c[0].citations[0].as_of == "2026-06-18"
    assert c[1].kind == UNVERIFIED                              # "fact" on a creator -> downgraded
    assert c[2].kind == UNVERIFIED and len(c[2].citations) == 0  # hallucinated chunk dropped
    assert c[3].kind == OPINION
    assert c[3].citations[0].tier == CredibilityTier.CREATOR


def test_assemble_abstain_payload():
    reg, retrieved = _setup_assembly()
    res = _assemble_result("q", {"abstain": True, "reason": "nope"}, retrieved, reg, None)
    assert res.abstained and res.abstain_reason == "nope"
