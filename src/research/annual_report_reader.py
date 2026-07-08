"""Grounded reading of an annual report.

A chartered accountant reads the filing, not just the numbers: management commentary, disclosed
risks, segment/business trends, and governance/accounting flags. This reuses the grounded analyst
(retrieve real chunks -> the model answers ONLY from them -> citations are enforced), so every
point is tied to the filing text or dropped; nothing is invented. The annual report is a PRIMARY
source, so a well-grounded point renders as a verified fact. Returns [] (abstain) when the report
text isn't available or no LLM is configured, rather than filling the gap with a made-up narrative.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm.client import LLMClient, LiteLLMClient
from ..sources.registry import CredibilityTier, Source, SourceRegistry
from .claims import ResearchResult
from .grounded_analyst import GroundedAnalyst
from .grounding import DocumentStore

AR_SOURCE_ID = "annual_report"

# The CA lens over a filing: (topic label shown to the reader, question put to the grounded model).
AR_TOPICS: tuple[tuple[str, str], ...] = (
    ("Management commentary",
     "What does management say about this year's performance and the outlook for the business?"),
    ("Key risks",
     "What key risks, challenges, or headwinds does the company disclose?"),
    ("Business & segment trends",
     "What does the report say about segment performance, capacity, order book, or business trends?"),
    ("Governance & accounting flags",
     "Does the report mention auditor qualifications, related-party transactions, contingent "
     "liabilities, or pledged promoter shares?"),
)


@dataclass(frozen=True)
class FilingReading:
    topic: str
    result: ResearchResult


def read_filing(ar_text: str | None, client: LLMClient | None = None,
                as_of: str | None = None) -> list[FilingReading]:
    """Read a filing's text across the CA topics, each answer grounded in the text and
    citation-enforced. Empty text or no LLM -> [] (the caller shows an honest 'couldn't read it')."""
    if not ar_text or not ar_text.strip():
        return []
    analyst = GroundedAnalyst(client=client or LiteLLMClient())
    if not analyst.available:
        return []
    registry = SourceRegistry([
        Source(AR_SOURCE_ID, "Annual report (company filing)", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=registry)
    store.add_document(AR_SOURCE_ID, ar_text, locator_prefix="annual report")
    return [FilingReading(topic, analyst.answer(question, store, registry, as_of=as_of))
            for topic, question in AR_TOPICS]
