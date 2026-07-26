"""H3: in-house Okapi BM25 lexical scoring + a hybrid (TF-IDF cosine + BM25) ranking.

Design under test (see DocumentStore.retrieve / _bm25_scores):
- The abstention GATE stays on the TF-IDF cosine floor. Over-abstaining is the safe failure for a
  real-money tool, and in a tiny corpus BM25's IDF barely penalizes common words, so a blended-score
  gate would surface chunks that overlap the query only on a stopword. BM25 therefore only RE-RANKS
  the chunks that already clear the cosine floor; it never lowers the evidence bar.
- BM25 is implemented in-house, stdlib math only (no rank_bm25/torch/faiss -- deploy weight on the
  free Streamlit tier is load-bearing), over the same _tokenize the cosine path uses.

These tests pin: the BM25 math (hand-computed), that the hybrid genuinely re-ranks, and that the
abstention / no-match contract is preserved (a below-floor chunk with a positive BM25 stays out).
"""
import math

from src.research.grounding import DocumentStore


def test_bm25_matches_hand_computed_fixture():
    # Corpus: d0="profit profit growth" (len 3), d1="revenue growth" (len 2). N=2, avgdl=2.5.
    # Query "profit" matches only d0. df(profit)=1, k1=1.5, b=0.75.
    #   idf   = ln(1 + (N - n + 0.5)/(n + 0.5)) = ln(1 + 1.5/1.5) = ln(2) = 0.6931471805599453
    #   denom = f + k1*(1 - b + b*dl/avgdl) = 2 + 1.5*(0.25 + 0.75*3/2.5) = 3.725
    #   bm25  = idf * f*(k1+1)/denom = ln(2) * (2*2.5)/3.725 = 0.9303989000804636
    store = DocumentStore()
    store.add_document("d0", "profit profit growth")
    store.add_document("d1", "revenue growth")
    scores = store._bm25_scores("profit")
    assert math.isclose(scores[0], 0.9303989000804636, rel_tol=1e-12)
    assert scores[1] == 0.0                      # query term absent -> no lexical signal


def test_bm25_is_zero_when_no_query_term_appears():
    store = DocumentStore()
    store.add_document("d0", "revenue and margin discussion")
    assert store._bm25_scores("quantum chromodynamics gluon") == [0.0]


def test_hybrid_promotes_chunk_covering_all_query_terms_over_keyword_stuffed():
    # Pure TF-IDF cosine ranks the keyword-stuffed chunk first (it repeats one term many times);
    # the BM25 term correctly lifts the chunk that actually covers BOTH query words. Both chunks
    # clear the cosine floor, so this is a re-ranking property, not a recall/abstention change.
    store = DocumentStore()
    store.add_document(
        "covers_both",
        "The company's cash flow improved this year, though the note also touches on logistics, "
        "warehousing, procurement, staffing and various routine administrative matters.")
    store.add_document(
        "stuffed_one",
        "Cash reserves, cash balances, cash position, cash buffers, cash holdings and overall "
        "cash strength all stayed cash rich.")
    hits = store.retrieve("cash flow", k=5)
    assert {h.chunk.source_id for h in hits} == {"covers_both", "stuffed_one"}   # both cleared floor
    assert hits[0].chunk.source_id == "covers_both"                             # BM25 re-ranked it up


def test_hybrid_preserves_abstention_on_no_match():
    store = DocumentStore(words_per_chunk=25, overlap=5)
    store.add_document("amfi", "A mutual fund SIP invests a fixed amount every month into a scheme.")
    assert store.retrieve("quantum chromodynamics gluon lattice", k=3) == []


def test_hybrid_empty_store_abstains():
    assert DocumentStore().retrieve("anything") == []


def _news_plus_below_floor_store():
    """8 short news chunks (each repeats the company name) + one small authoritative chunk whose
    only surface overlap with the earnings-quality question is a common word. Reproduces the real
    Ask-tab shape where the authoritative chunk scores below the cosine floor."""
    store = DocumentStore()
    news = [
        "Reliance shares slip after a regulatory warning affecting the stock price today.",
        "Reliance earnings preview: analysts expect strong retail and telecom growth this quarter.",
        "Reliance stock hits a 52-week high on strong subscriber additions this month.",
        "Reliance retail expands into new cities and the stock reacts positively to the news.",
        "Reliance announces a new green energy investment plan for the coming decade ahead.",
        "Reliance telecom price hike is expected to boost average revenue and profit margins.",
        "Reliance is in talks for a new petrochemical joint venture deal with partners.",
        "Reliance stock outlook: brokerages raise the target price after a strong quarter.",
    ]
    for text in news:
        store.add_document("news_google", text)
    store.add_document(
        "other_income_share",
        "Other income share of profit for RELIANCE: 27% of profit before tax came from "
        "non-operating \"other income\" (investment gains, interest income, or one-off items) "
        "rather than the core business -- worth checking how repeatable that income is (not "
        "cross-verified, Screener only).")
    return store


def test_positive_bm25_does_not_override_the_cosine_abstention_gate():
    # The core safety property of the design: the below-floor chunk has a POSITIVE BM25 (it shares
    # surface tokens with the query), yet it stays out of the results because the gate is the
    # cosine floor, not the blended score. Withholding thin evidence is the intended behavior;
    # recall for such chunks is handled deliberately by pin_source_ids, not by lowering the bar.
    store = _news_plus_below_floor_store()
    query = "Is earnings quality good?"
    idx = next(i for i, c in enumerate(store._chunks) if c.source_id == "other_income_share")
    assert store._bm25_scores(query)[idx] > 0.0                 # BM25 "wants" the chunk
    hits = store.retrieve(query, k=5)
    assert not any(h.chunk.source_id == "other_income_share" for h in hits)   # cosine gate withholds


def test_hybrid_retrieve_is_deterministic():
    store = DocumentStore()
    store.add_document("a", "revenue growth and margin expansion this year")
    store.add_document("b", "debt reduction and cash flow discipline this year")
    query = "revenue margin"
    assert store.retrieve(query, k=5) == store.retrieve(query, k=5)
