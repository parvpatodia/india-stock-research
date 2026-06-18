"""Document grounding: the LLM only ever sees real text retrieved from the owner's sources.

A DocumentStore holds chunks of ingested documents, each tagged with its source id. Retrieval
uses a dependency-light TF-IDF cosine score (swappable for embeddings later). If nothing
scores above the floor, retrieve returns nothing and the caller abstains. No chunk, no claim.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    locator: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _chunk_text(text: str, words_per_chunk: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= words_per_chunk:
        return [" ".join(words)]
    step = max(1, words_per_chunk - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + words_per_chunk]
        if piece:
            chunks.append(" ".join(piece))
        if start + words_per_chunk >= len(words):
            break
    return chunks


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

    def add_document(self, source_id: str, text: str, locator_prefix: str = "") -> int:
        if self._registry is not None and self._registry.get(source_id) is None:
            raise ValueError(
                f"source '{source_id}' is not in the registry; add it to config/sources.yaml "
                "before ingesting its documents")
        pieces = _chunk_text(text, self.words_per_chunk, self.overlap)
        for i, piece in enumerate(pieces):
            chunk_id = f"{source_id}#{len(self._chunks)}"
            locator = f"{locator_prefix} chunk {i}".strip()
            self._chunks.append(Chunk(chunk_id, source_id, piece, locator))
            self._tokens.append(_tokenize(piece))
        return len(pieces)

    def __len__(self) -> int:
        return len(self._chunks)

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

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.10) -> list[RetrievedChunk]:
        # WHY: a higher cosine floor than a token-overlap minimum reduces the chance a barely
        # related chunk (one shared common word) gets retrieved and then cited as fact.
        # Over-abstaining is the safe failure here; tune up if it abstains too often.
        if not self._chunks:
            return []
        idf = self._idf()
        q_vec = self._tfidf_vec(_tokenize(query), idf)
        scored: list[RetrievedChunk] = []
        for chunk, tokens in zip(self._chunks, self._tokens):
            score = self._cosine(q_vec, self._tfidf_vec(tokens, idf))
            if score >= min_score:
                scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]
