"""Near-duplicate news clustering.

One market event is rewritten by a dozen outlets ("Reliance shares slip after SEBI warning" /
"SEBI warning sends Reliance shares slipping" / ...). Left alone, N rewrites flood the feed as N
"stories". This collapses them to one cluster using a dependency-light token-set similarity on
normalized titles -- no embeddings, no heavy deps.

Similarity is the OVERLAP COEFFICIENT (|A cap B| / min(|A|,|B|)), not plain Jaccard: a longer
reworded headline that fully contains a shorter one's core should still count as the same story,
and Jaccard penalizes that length asymmetry. A minimum shared-token count guards against merging
two items that only share a single common word.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

_APOSTROPHE = re.compile(r"['’]")          # possessive/contraction: reliance's -> reliances
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)      # any other punctuation -> a word boundary (space)
_WS = re.compile(r"\s+")

# Common headline glue words. Removed before comparison so two rewrites are judged on their
# content words, not shared scaffolding. A title made ONLY of these keeps them (never empty).
_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "as", "at", "by", "is", "are",
    "after", "over", "amid", "its", "with", "from", "up", "down", "new",
})

# Two items must share at least this many content tokens before their overlap score is even
# considered. WHY: the overlap coefficient scores a single shared common word as 1.0 when one
# side is a single token; requiring >= 2 shared tokens stops that spurious merge.
_MIN_SHARED_TOKENS = 2


def normalize_title(title: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Deterministic and script-agnostic
    (Unicode \\w keeps Devanagari and other non-Latin word characters)."""
    if not title:
        return ""
    lowered = _APOSTROPHE.sub("", str(title).lower())   # drop apostrophes without splitting the word
    return _WS.sub(" ", _PUNCT.sub(" ", lowered)).strip()


def title_tokens(title: str | None) -> frozenset[str]:
    """Content tokens of a title: normalized words minus stopwords. If a title is made entirely
    of stopwords, its stopwords are kept rather than returning an empty set (so it can still match
    an identical copy of itself), matching the news dedup rule 'can't prove distinct -> keep'."""
    words = normalize_title(title).split()
    content = [w for w in words if w not in _STOPWORDS]
    return frozenset(content or words)


def token_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient of two token sets in [0, 1]; 0.0 if either side is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cluster_items(items: Sequence[T] | Iterable[T],
                  key: Callable[[T], str] = lambda x: x,  # type: ignore[return-value]
                  threshold: float = 0.5) -> list[list[T]]:
    """Group near-duplicate items by title similarity. Greedy single-best-linkage: each item joins
    the existing cluster whose most-similar member clears both the shared-token floor and the
    similarity threshold; otherwise it starts a new cluster. Input order is preserved. Pure."""
    clusters: list[dict] = []   # each: {"members": [...], "token_sets": [frozenset, ...]}
    for item in items:
        tokens = title_tokens(key(item))
        best_idx, best_sim = -1, 0.0
        for idx, cluster in enumerate(clusters):
            for existing in cluster["token_sets"]:
                if len(tokens & existing) < _MIN_SHARED_TOKENS:
                    continue
                sim = token_similarity(tokens, existing)
                if sim >= threshold and sim > best_sim:
                    best_sim, best_idx = sim, idx
        if best_idx >= 0:
            clusters[best_idx]["members"].append(item)
            clusters[best_idx]["token_sets"].append(tokens)
        else:
            clusters.append({"members": [item], "token_sets": [tokens]})
    return [cluster["members"] for cluster in clusters]
