"""Grounded question answering: retrieve real chunks -> ask the model to answer ONLY from
them and return structured claims -> resolve citations against the registry -> enforce the
citation contract. Abstains when retrieval is empty or no key is set.

The model's output is never trusted as-is. _assemble_result resolves every cited chunk id
against what was actually retrieved (hallucinated ids are dropped) and enforce_citations
downgrades any "fact" that lacks a primary source. _assemble_result is pure and tested.
"""
from __future__ import annotations

import json

from ..llm.client import LLMClient, LiteLLMClient
from ..sources.registry import SourceRegistry
from .claims import (
    ESTIMATE,
    FACT,
    OPINION,
    Claim,
    ResearchResult,
    build_citation,
    enforce_citations,
)
from .grounding import DocumentStore, RetrievedChunk

_ALLOWED_KINDS = {FACT, OPINION, ESTIMATE}

_SYSTEM = """You answer questions about Indian investments for a non-expert reader, using \
ONLY the SOURCES provided. The reader uses your answer with real money, so accuracy and \
honesty about uncertainty matter more than completeness.

HARD RULES:
- Use ONLY text from the SOURCES below. Never add a number, name, date, or fact that is not \
in the provided chunks. If the sources do not answer the question, abstain.
- Every claim must cite the chunk id(s) it came from.
- Label each claim's kind: "fact" only for something stated directly in a chunk; "opinion" \
for an attributed view; "estimate" for something you derived or approximated.
- Give NO buy/sell/hold advice, NO price target, NO prediction, NO promise of returns.
- Plain English. Short sentences. No jargon without a one-line explanation.

Return ONLY JSON, no prose, in this exact shape:
{"abstain": false, "claims": [{"text": "...", "chunk_ids": ["id1"], "kind": "fact"}]}
If the sources cannot answer, return {"abstain": true, "reason": "..."}.
"""


class GroundedAnalyst:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LiteLLMClient()

    @property
    def available(self) -> bool:
        return self.client.available

    def answer(self, question: str, store: DocumentStore, registry: SourceRegistry,
               k: int = 5, as_of: str | None = None) -> ResearchResult:
        retrieved = store.retrieve(question, k=k)
        if not retrieved:
            return ResearchResult.abstain(
                question,
                "No source in the library matched this question. Add a relevant primary "
                "source (annual report, filing, exchange/AMFI data) and ask again.",
            )
        if not self.available:
            return ResearchResult.abstain(
                question,
                "Sources matched, but no LLM is configured. Set LLM_MODEL (e.g. an NVIDIA "
                "NIM open model) to generate a grounded answer.",
            )
        payload = self._ask_model(question, retrieved)
        return _assemble_result(question, payload, retrieved, registry, as_of)

    def _ask_model(self, question: str, retrieved: list[RetrievedChunk]) -> dict:
        sources_block = "\n\n".join(
            f"[{rc.chunk.chunk_id}] (source: {rc.chunk.source_id})\n{rc.chunk.text}"
            for rc in retrieved
        )
        user = f"QUESTION:\n{question}\n\nSOURCES:\n{sources_block}"
        try:
            raw = self.client.complete(_SYSTEM, user, max_tokens=1200, json_mode=True)
            return _parse_json(raw)
        except Exception as exc:
            return {"abstain": True, "reason": f"answer generation failed: {exc}"}


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        if text.lstrip().lower().startswith("json"):  # case-insensitive language tag
            text = text.lstrip()[4:]
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {"abstain": True, "reason": "model did not return valid JSON"}


def _assemble_result(question: str, payload: dict, retrieved: list[RetrievedChunk],
                     registry: SourceRegistry, as_of: str | None) -> ResearchResult:
    """Pure: turn the model payload into a validated ResearchResult.

    Resolves cited chunk ids only against what was actually retrieved (drops hallucinated
    ids), resolves each chunk's source against the registry (drops unknown sources), then
    enforces the citation contract so an unsourced 'fact' can never render as fact.
    """
    if not isinstance(payload, dict) or payload.get("abstain"):
        reason = (payload.get("reason") if isinstance(payload, dict) else None) \
            or "no verified answer"
        return ResearchResult.abstain(question, reason)

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return ResearchResult.abstain(question, "model returned no usable claims")

    chunk_by_id = {rc.chunk.chunk_id: rc.chunk for rc in retrieved}
    claims: list[Claim] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        text = str(raw_claim.get("text", "")).strip()
        if not text:
            continue
        cids = raw_claim.get("chunk_ids")
        if not isinstance(cids, list):
            cids = []
        citations = []
        for cid in cids:
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue  # model cited a chunk it was not given -> drop
            citation = build_citation(chunk.source_id, chunk.chunk_id, registry, as_of)
            if citation is not None:
                citations.append(citation)
        # WHY: no chunk, no claim. A claim with no resolved citation is unsourced and must
        # never display (not even as opinion), so it is dropped entirely.
        if not citations:
            continue
        kind = str(raw_claim.get("kind", OPINION)).lower()
        if kind not in _ALLOWED_KINDS:
            kind = OPINION
        claims.append(Claim(text=text, citations=tuple(citations), kind=kind))

    if not claims:
        return ResearchResult.abstain(
            question, "Sources matched, but no claim could be tied to them. No verified answer.")

    result = ResearchResult(question=question, claims=tuple(claims))
    return enforce_citations(result)
