"""W6 span-level citations (SPEC v4 §4 W6, §3 "cited but not verified").

Every Citation must carry the EXACT supporting text span inside its cited chunk -- the fragment
that actually contains the claim's material number (or, absent a number, its best key-phrase
match) -- plus a resolvable locator, so a UI can click through to "source · locator: '<quote>'".
That same span drives a citation-QUALITY check: a FACT whose cited chunks contain no supporting
span for its material number is downgraded (never a green fact with a citation that doesn't back
its figure). Offline and deterministic: _assemble_result is exercised with a hand-built payload,
no LLM.
"""
from src.research.claims import (
    FACT,
    UNVERIFIED,
    Citation,
    Claim,
)
from src.research.grounded_analyst import (
    _assemble_result,
    citation_supports_numbers,
    supporting_span,
)
from src.research.grounding import DocumentStore
from src.sources.registry import CredibilityTier, Source, SourceRegistry


def _primary(source_id: str = "ar") -> SourceRegistry:
    return SourceRegistry([Source(source_id, source_id, CredibilityTier.PRIMARY)])


def _payload(text: str, chunk_id: str, kind: str = "fact") -> dict:
    return {"abstain": False,
            "claims": [{"text": text, "chunk_ids": [chunk_id], "kind": kind}]}


# --- Citation schema: optional span field + backward compatibility ----------------------------

def test_citation_quote_defaults_empty_and_positional_construction_still_valid():
    # a 3-positional construction (as used across the existing tests) stays valid; quote defaults ""
    c = Citation("ar", CredibilityTier.PRIMARY, "p.42")
    assert c.quote == ""
    assert c.as_of is None


def test_citation_as_span_returns_source_locator_quote():
    c = Citation("ar", CredibilityTier.PRIMARY, "p.42", quote="Net profit Rs 500 crore.")
    assert c.as_span() == {"source_id": "ar", "locator": "p.42",
                           "quote": "Net profit Rs 500 crore."}


def test_claim_spans_exposes_every_citation():
    c1 = Citation("ar", CredibilityTier.PRIMARY, "p.1", quote="Net profit Rs 500 crore.")
    c2 = Citation("news", CredibilityTier.ANALYST, "Reuters, 2026-05-15", quote="")
    claim = Claim("x", (c1, c2), FACT)
    assert claim.spans() == (
        {"source_id": "ar", "locator": "p.1", "quote": "Net profit Rs 500 crore."},
        {"source_id": "news", "locator": "Reuters, 2026-05-15", "quote": ""},
    )


# --- supporting_span: the exact quote for a citation ------------------------------------------

def test_supporting_span_returns_the_fragment_containing_the_material_number():
    chunk = ("The company did well overall. Net profit for the year was Rs 500 crore. "
             "Costs rose sharply.")
    assert supporting_span("Net profit was Rs 500 crore.", chunk) == \
        "Net profit for the year was Rs 500 crore."


def test_supporting_span_falls_back_to_key_phrase_when_no_number():
    chunk = "Risks include competition. Management flagged a strong order book this year."
    span = supporting_span("Management cites a strong order book.", chunk)
    assert "order book" in span


def test_supporting_span_never_fabricates_the_claimed_number():
    # the claim states 999 crore, absent from the chunk; the fallback fragment must NOT contain 999
    chunk = "Net profit for the year was Rs 500 crore."
    span = supporting_span("Net profit was Rs 999 crore.", chunk)
    assert "999" not in span


# --- citation_supports_numbers: the "cited but not verified" predicate -------------------------

def test_supports_numbers_true_when_number_is_locatable_in_a_cited_chunk():
    assert citation_supports_numbers(
        "Net profit was Rs 500 crore.",
        ["The year was solid. Net profit for the year was Rs 500 crore."]) is True


def test_supports_numbers_false_when_number_absent_from_every_cited_chunk():
    assert citation_supports_numbers(
        "Net profit was Rs 999 crore.",
        ["Net profit for the year was Rs 500 crore."]) is False


def test_supports_numbers_requires_all_material_numbers_to_have_a_span():
    # two material numbers, only one present in the cited chunk -> not fully supported
    assert citation_supports_numbers(
        "Revenue was Rs 800 crore and profit Rs 120 crore.",
        ["Revenue was Rs 800 crore."]) is False


def test_supports_numbers_true_when_claim_has_no_material_number():
    assert citation_supports_numbers(
        "Management is optimistic about the order book.", ["unrelated text"]) is True


# --- _assemble_result integration: span attached; downgrade on an unsupported figure ----------

def test_fact_with_supported_number_keeps_verified_status_and_span():
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    store.add_document(
        "ar",
        "The company did well overall. Net profit for the year was Rs 500 crore. Costs rose.",
        company="ACME", doc_type="annual_report")
    retrieved = store.retrieve("net profit", k=5, min_score=0.0)
    cid = retrieved[0].chunk.chunk_id
    result = _assemble_result("q", _payload("Net profit for the year was Rs 500 crore.", cid),
                              retrieved, reg, None)
    assert not result.abstained
    claim = result.claims[0]
    assert claim.is_verified_fact
    span = claim.spans()[0]
    assert span["source_id"] == "ar"
    assert span["locator"]                 # a resolvable, non-empty locator for click-through
    assert "500 crore" in span["quote"]    # the exact supporting fragment


def test_fact_with_unsupported_number_is_downgraded_to_unverified():
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    store.add_document("ar", "Net profit for the year was Rs 500 crore.",
                       company="ACME", doc_type="annual_report")
    retrieved = store.retrieve("net profit", k=5, min_score=0.0)
    cid = retrieved[0].chunk.chunk_id
    # the model cites the real chunk but states a figure (999 crore) it does not contain
    result = _assemble_result("q", _payload("Net profit for the year was Rs 999 crore.", cid),
                              retrieved, reg, None)
    claim = result.claims[0]
    assert claim.kind == UNVERIFIED
    assert not claim.is_verified_fact
    # and the citation never quotes back the fabricated figure
    assert "999" not in claim.citations[0].quote


def test_non_numeric_fact_keeps_verification_and_gets_a_key_phrase_span():
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    store.add_document(
        "ar",
        "Risks include competition and input costs. Management flagged a strong order book.",
        company="ACME", doc_type="annual_report")
    retrieved = store.retrieve("order book", k=5, min_score=0.0)
    cid = retrieved[0].chunk.chunk_id
    result = _assemble_result("q", _payload("Management flagged a strong order book.", cid),
                              retrieved, reg, None)
    claim = result.claims[0]
    assert claim.is_verified_fact               # no number -> not downgraded by the span check
    assert "order book" in claim.spans()[0]["quote"]
