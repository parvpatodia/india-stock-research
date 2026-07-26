"""Document grounding: the LLM only ever sees real text retrieved from the owner's sources.

A DocumentStore holds chunks of ingested documents, each tagged with its source id. Retrieval is
dependency-light and stdlib-only (no dense-embedding dependency -- deploy weight is load-bearing
on the free tier): a chunk is admitted only if its TF-IDF cosine clears the abstention floor, and
the admitted chunks are then RE-RANKED by a hybrid score that adds an in-house Okapi BM25 term
(_bm25_scores) so keyword/term-heavy queries surface the exact-match chunk. If nothing clears the
cosine floor, retrieve returns nothing and the caller abstains. No chunk, no claim.

Ingestion has two modes. The default (blind) mode chunks by overlapping word windows, unchanged.
The opt-in `structured=True` mode chunks element-aware (src/research/structure.py): section
headings, paragraphs, and TABLES kept intact with their caption/units/period line, each chunk
tagged with rich metadata (company/doc_type/fiscal_period/section/currency/unit_scale). In BOTH
modes every chunk carries the typed numeric records extracted from its text
(src/research/numeric_records.py), so a retrieved chunk can cite a stated number back to its exact
record and its normalized absolute magnitude -- the seam a compute-don't-generate layer uses.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .numeric_records import NumericRecord, extract_records, find_record, number_key
from .structure import structure_chunks, window_split

# Kept as a module-private alias so the blind ingestion path reads unchanged; the single
# canonical implementation now lives in structure.window_split (reused by both paths).
_chunk_text = window_split

# Re-exported for callers that resolve a stated number to its record (no record -> no claim).
__all__ = [
    "Chunk", "ChunkMetadata", "RetrievedChunk", "DocumentStore",
    "NumericRecord", "find_record", "number_key", "numeric_records",
]

_TOKEN = re.compile(r"[a-z0-9]+")

# Okapi BM25 hyperparameters (standard defaults) + the hybrid re-rank weight. In-house, stdlib
# math only -- no rank_bm25/torch/faiss/embedding dependency, because the app deploys on the free
# Streamlit Cloud tier where a heavy ML dependency would break the parents' live build.
_BM25_K1 = 1.5      # term-frequency saturation
_BM25_B = 0.75      # document-length normalization
_W_BM25 = 0.5       # weight of the (per-query normalized) BM25 term added on top of cosine


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class ChunkMetadata:
    """Provenance/context tags carried by a chunk. All optional so blind ingestion (which passes
    none) is unaffected and existing chunk construction stays valid."""
    company: str | None = None
    symbol: str | None = None
    doc_type: str | None = None        # "annual_report" | "news" | ...
    fiscal_period: str | None = None   # e.g. "FY2024"
    as_of: str | None = None
    section: str | None = None
    currency: str | None = None        # "INR"
    unit_scale: str | None = None      # "crore" | "lakh" | "million" | "absolute"
    element_kind: str | None = None    # "heading" | "paragraph" | "table"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    locator: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    records: tuple[NumericRecord, ...] = ()


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def numeric_records(retrieved: list[RetrievedChunk]) -> list[NumericRecord]:
    """Flatten the typed numeric records carried by a list of retrieved chunks. This is the
    retrieval-side wiring: a grounded answer takes its citable numbers from here (each record
    knows its source_doc + locator), and a number with no matching record cannot be claimed."""
    out: list[NumericRecord] = []
    for rc in retrieved:
        out.extend(rc.chunk.records)
    return out


# Table scale -> the extractor's default_scale (a plain-rupee "absolute" table mines cells as
# scale "none"; a scaled table mines cells with its multiplier; None disables cell mining).
_EXTRACT_SCALE = {"crore": "crore", "lakh": "lakh", "million": "million", "absolute": "none"}


class DocumentStore:
    def __init__(self, words_per_chunk: int = 120, overlap: int = 20, registry=None):
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []   # parallel to _chunks
        self.words_per_chunk = words_per_chunk
        self.overlap = overlap
        # WHY: if a registry is given, every ingested chunk's source must be tiered in it.
        # This closes the trust boundary at ingestion so no chunk with an unknown (untiered)
        # source can later ride into a fact via a co-cited primary chunk.
        self._registry = registry

    def add_document(self, source_id: str, text: str, locator_prefix: str = "", *,
                     structured: bool = False, doc_type: str | None = None,
                     company: str | None = None, symbol: str | None = None,
                     fiscal_period: str | None = None, as_of: str | None = None,
                     currency: str | None = None) -> int:
        """Ingest a document as chunks. Default: blind overlapping word windows (unchanged).
        structured=True: element-aware chunking (tables kept intact) with per-piece metadata.
        Either way, each chunk carries the typed numeric records extracted from its text."""
        if self._registry is not None and self._registry.get(source_id) is None:
            raise ValueError(
                f"source '{source_id}' is not in the registry; add it to config/sources.yaml "
                "before ingesting its documents")
        base = ChunkMetadata(company=company, symbol=symbol, doc_type=doc_type,
                             fiscal_period=fiscal_period, as_of=as_of, currency=currency)
        if structured:
            return self._add_structured(source_id, text, locator_prefix, base)
        return self._add_blind(source_id, text, locator_prefix, base)

    def _add_blind(self, source_id: str, text: str, locator_prefix: str,
                   base: ChunkMetadata) -> int:
        pieces = _chunk_text(text, self.words_per_chunk, self.overlap)
        for i, piece in enumerate(pieces):
            locator = f"{locator_prefix} chunk {i}".strip()
            self._append(source_id, piece, locator, base,
                         default_scale=None, currency=base.currency)
        return len(pieces)

    def _add_structured(self, source_id: str, text: str, locator_prefix: str,
                        base: ChunkMetadata) -> int:
        pieces = structure_chunks(text, self.words_per_chunk, self.overlap)
        for i, sp in enumerate(pieces):
            loc_bits = [b for b in (locator_prefix, sp.section, f"chunk {i}") if b]
            locator = " · ".join(loc_bits).strip()
            meta = ChunkMetadata(
                company=base.company, symbol=base.symbol, doc_type=base.doc_type,
                fiscal_period=sp.fiscal_period or base.fiscal_period, as_of=base.as_of,
                section=sp.section, currency=sp.currency or base.currency,
                unit_scale=sp.unit_scale, element_kind=sp.element_kind)
            default_scale = _EXTRACT_SCALE.get(sp.unit_scale) if sp.element_kind == "table" else None
            self._append(source_id, sp.text, locator, meta,
                         default_scale=default_scale, currency=meta.currency)
        return len(pieces)

    def _append(self, source_id: str, text: str, locator: str, meta: ChunkMetadata, *,
                default_scale: str | None, currency: str | None) -> None:
        chunk_id = f"{source_id}#{len(self._chunks)}"
        records = tuple(extract_records(
            text, default_scale=default_scale, currency=currency,
            period=meta.fiscal_period, company=meta.company,
            source_doc=source_id, locator=chunk_id))
        self._chunks.append(Chunk(chunk_id, source_id, text, locator, meta, records))
        self._tokens.append(_tokenize(text))

    def __len__(self) -> int:
        return len(self._chunks)

    def source_ids(self) -> set[str]:
        """Distinct source ids that have at least one chunk in the store."""
        return {c.source_id for c in self._chunks}

    def _idf(self) -> dict[str, float]:
        n = len(self._chunks)
        df: Counter = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                df[term] += 1
        # smoothed idf, always positive
        return {term: math.log((1 + n) / (1 + d)) + 1.0 for term, d in df.items()}

    @staticmethod
    def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        return {t: (c / total) * idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _bm25_scores(self, query: str) -> list[float]:
        """Okapi BM25 lexical relevance of every chunk to the query, parallel to self._chunks.

        In-house, stdlib math only (no rank_bm25/torch/faiss -- deploy weight is load-bearing on
        the free tier). Corpus stats (per-doc length, average length, document frequency) come
        from the same _tokenize output the TF-IDF path uses, so both lexical signals see identical
        tokens. IDF is the non-negative Robertson/Sparck-Jones variant log(1 + (N - n + 0.5)/(n +
        0.5)), so no term ever contributes a negative score. Query terms are de-duplicated (each
        distinct query term is scored once), the standard bag-of-words BM25 formulation.
        Recomputed per call (mirrors the existing _idf pattern); the corpus is small and this keeps
        the stats consistent with the current chunk set with no cache to invalidate on ingestion.
        """
        n = len(self._tokens)
        if n == 0:
            return []
        lengths = [len(t) for t in self._tokens]
        avgdl = sum(lengths) / n if lengths else 0.0
        df: Counter = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                df[term] += 1
        q_terms = set(_tokenize(query))
        scores: list[float] = []
        for tokens, dl in zip(self._tokens, lengths):
            s = 0.0
            if q_terms:
                tf = Counter(tokens)
                for term in q_terms:
                    f = tf.get(term, 0)
                    if not f:
                        continue
                    nt = df[term]
                    idf = math.log(1.0 + (n - nt + 0.5) / (nt + 0.5))
                    denom = f + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (dl / avgdl if avgdl else 0.0))
                    s += idf * (f * (_BM25_K1 + 1.0)) / denom
            scores.append(s)
        return scores

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.10,
                pin_source_ids: frozenset[str] = frozenset()) -> list[RetrievedChunk]:
        # WHY: a higher cosine floor than a token-overlap minimum reduces the chance a barely
        # related chunk (one shared common word) gets retrieved and then cited as fact.
        # Over-abstaining is the safe failure here; tune up if it abstains too often.
        #
        # Hybrid ranking (H3): the returned chunks are ordered by a combined score
        # = cosine + _W_BM25 * bm25_norm, where bm25_norm is the chunk's in-house Okapi BM25
        # (_bm25_scores) normalized by the max BM25 over the candidate set. This lets term/keyword-
        # heavy financial queries rank the exact-match chunk above one that merely repeats a term.
        #
        # WHY the ABSTENTION GATE stays on the raw TF-IDF cosine (not the combined score): in a
        # small corpus BM25's IDF barely penalizes common words, so a combined-score floor would
        # admit chunks that overlap the query only on a stopword ("the"/"is") -- exactly the thin-
        # evidence retrieval this app must abstain on. Cosine's full-length normalization is robust
        # to that, so it remains the gate; BM25 only RE-RANKS what already cleared the floor and
        # never lowers the evidence bar. (Measured: a combined gate surfaced two below-floor chunks
        # on stopword-only matches.) Recall for authoritative-but-buried chunks is handled by
        # pin_source_ids below, deliberately, rather than by weakening the floor.
        #
        # pin_source_ids: chunks from a pinned source are ALWAYS included, bypassing min_score and
        # the k cutoff. Demonstrated bug this closes: a handful of authoritative chunks (e.g. this
        # app's own cross-verified figures for the asked stock) can be crowded out of a mixed-
        # source context by a larger volume of lower-value chunks (news items) that happen to
        # share more surface keywords with the query, even for a question the authoritative chunk
        # directly answers. Pinning guarantees it reaches the model; it still has to be cited to
        # matter, and the numeric-grounding + citation-tier checks still apply.
        if not self._chunks:
            return []
        idf = self._idf()
        q_vec = self._tfidf_vec(_tokenize(query), idf)
        cos = [self._cosine(q_vec, self._tfidf_vec(tokens, idf)) for tokens in self._tokens]
        bm = self._bm25_scores(query)
        # Candidate = pinned (always) OR clears the cosine floor. bm25 is normalized per query over
        # the candidate set so the re-rank term is bounded [0, _W_BM25] and never gates admission.
        cand = [i for i, chunk in enumerate(self._chunks)
                if chunk.source_id in pin_source_ids or cos[i] >= min_score]
        bmax = max((bm[i] for i in cand), default=0.0)
        scored: list[RetrievedChunk] = []
        pinned: list[RetrievedChunk] = []
        for i in cand:
            bm_norm = (bm[i] / bmax) if bmax > 0 else 0.0
            rc = RetrievedChunk(chunk=self._chunks[i], score=cos[i] + _W_BM25 * bm_norm)
            if self._chunks[i].source_id in pin_source_ids:
                pinned.append(rc)
            else:
                scored.append(rc)
        scored.sort(key=lambda rc: rc.score, reverse=True)
        pinned.sort(key=lambda rc: rc.score, reverse=True)
        pinned_ids = {rc.chunk.chunk_id for rc in pinned}
        rest = [rc for rc in scored if rc.chunk.chunk_id not in pinned_ids][:k]
        return pinned + rest
