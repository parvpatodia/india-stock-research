"""The claim/citation contract.

Every research output is a list of Claims. A Claim is a fact only if it cites a Tier-1
primary source. enforce_citations downgrades any "fact" that lacks one to "unverified"
BEFORE it can be shown, so the display layer never has to trust the model's self-labeling.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..sources.registry import CredibilityTier, SourceRegistry

# Claim kinds
FACT = "fact"            # backed by a Tier-1 primary source
OPINION = "opinion"      # attributed view (Tier-2/3), not a verified fact
ESTIMATE = "estimate"    # a derived/approximate figure, not a primary statement
UNVERIFIED = "unverified"  # claimed as fact but not backed by a primary source -> flagged


@dataclass(frozen=True)
class Citation:
    source_id: str
    tier: CredibilityTier
    locator: str            # where in the source, e.g. "FY24 annual report, p.42" or chunk id
    as_of: str | None = None
    # W6 (SPEC v4 §4 W6 / §3, span-level click-through): the EXACT supporting text span inside the
    # cited chunk -- the sentence/fragment that actually contains the claim's material number (or,
    # absent a number, its best key-phrase match). OPTIONAL (defaults empty) so every existing
    # positional/keyword construction stays valid; populated by _assemble_result. Together with
    # source_id + locator this is the click-through payload a UI renders as
    # "this figure came from [source · locator]: '<exact quote>'".
    quote: str = ""

    def as_span(self) -> dict[str, str]:
        """The click-through provenance a UI (W7) renders: which source, where in it, and the exact
        supporting quote. Plain strings so the presentation layer needs no import from here."""
        return {"source_id": self.source_id, "locator": self.locator, "quote": self.quote}


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[Citation, ...]
    kind: str = OPINION

    @property
    def is_verified_fact(self) -> bool:
        # WHY: a fact must be backed by primary sources ONLY. `any` would let a claim ride
        # on one annual-report chunk while co-citing a finfluencer chunk; `all` (with a
        # non-empty guard) means a single non-primary citation breaks fact status.
        return (
            self.kind == FACT
            and len(self.citations) > 0
            and all(c.tier == CredibilityTier.PRIMARY for c in self.citations)
        )

    @property
    def has_any_citation(self) -> bool:
        return len(self.citations) > 0

    def spans(self) -> tuple[dict[str, str], ...]:
        """Every citation's span payload ({source_id, locator, quote}), for W7 click-through
        rendering. A citation with no located quote still appears (quote='') so its source+locator
        are always available to the UI."""
        return tuple(c.as_span() for c in self.citations)


@dataclass(frozen=True)
class ResearchResult:
    question: str
    claims: tuple[Claim, ...]
    abstained: bool = False
    abstain_reason: str | None = None

    @classmethod
    def abstain(cls, question: str, reason: str) -> "ResearchResult":
        return cls(question=question, claims=(), abstained=True, abstain_reason=reason)


def enforce_citations(result: ResearchResult) -> ResearchResult:
    """Downgrade any claim labeled FACT that lacks a Tier-1 citation to UNVERIFIED.

    This is the load-bearing check: the model can label a claim however it likes, but it
    cannot get FACT status without a primary citation. Nothing reaches the user as fact
    unless a primary source backs it.
    """
    fixed: list[Claim] = []
    for claim in result.claims:
        if claim.kind == FACT and not claim.is_verified_fact:
            fixed.append(replace(claim, kind=UNVERIFIED))
        else:
            fixed.append(claim)
    return replace(result, claims=tuple(fixed))


def build_citation(source_id: str, locator: str, registry: SourceRegistry,
                   as_of: str | None = None) -> Citation | None:
    """Resolve a source id against the registry. Returns None if the source is unknown,
    so a hallucinated source id cannot mint a citation."""
    source = registry.get(source_id)
    if source is None:
        return None
    return Citation(source_id=source_id, tier=source.tier, locator=locator, as_of=as_of)
