"""Red-team / adversarial eval suite (SPEC v4 §4 W5, §3 anti-hallucination).

A fixed set of adversarial cases the research pipeline MUST resist. Each case drives the REAL
pipeline (the W4 orchestrator + grounded_analyst, with an injected fake LLM that TRIES to surface a
confident wrong number) or the real guardrail primitive, and asserts the guardrail neutralized the
attack: downgraded the claim to UNVERIFIED, withheld it, or flagged it stale -- never rendered a
wrong figure as a verified fact. Deterministic and offline (a scripted fake client, no network, no
key), so it can gate a build in CI.

Classes (one case each):
- phantom_figure: a number with no typed record (a bare-year "2024 crore") -> record-backed
  grounding (W3 numbers_record_backed) downgrades it.
- unit_trap: a crore source figure answered in lakh (same digits, 100x) -> the unit-consistency
  guard (W5 numbers_unit_consistent) downgrades it.
- period_mixing: an FY2023 figure presented as FY2024 -> typed records preserve the true period, so
  the swap cannot resolve to an FY2024 record (the exact-match gate leverages this).
- staleness: a stale-dated retrieved chunk -> the W1 freshness guardrail flags it stale, so it can
  never be shown as current.

This EXTENDS src/eval; it reuses the orchestrator, grounded_analyst, numeric_records, and the W1
freshness primitive rather than re-implementing any of them.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..freshness.staleness import freshness
from ..llm.client import LLMClient
from ..research.claims import ResearchResult
from ..research.grounding import DocumentStore, numeric_records
from ..research.numeric_records import number_key
from ..research.orchestrator import ResearchOrchestrator
from ..research.grounded_analyst import GroundedAnalyst
from ..sources.registry import CredibilityTier, Source, SourceRegistry

# A fixed reference "today" so the staleness verdict is deterministic (no wall-clock in the gate).
_TODAY = "2026-07-25"


@dataclass(frozen=True)
class RedTeamOutcome:
    """One adversarial case's verdict. `resisted` True means the guardrail neutralized the attack;
    `detail` names the mechanism (or, on a breach, what leaked)."""
    name: str
    attack_class: str
    resisted: bool
    detail: str


@dataclass(frozen=True)
class RedTeamReport:
    outcomes: tuple[RedTeamOutcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def resisted(self) -> int:
        return sum(1 for o in self.outcomes if o.resisted)

    @property
    def breaches(self) -> list[RedTeamOutcome]:
        return [o for o in self.outcomes if not o.resisted]

    @property
    def all_resisted(self) -> bool:
        return not self.breaches


class _ScriptedClient(LLMClient):
    """A fake LLM that always returns the SAME attacker-controlled JSON payload, so a case can make
    the model TRY to emit a specific wrong number and prove the guardrails catch it. Available by
    default (the attack must reach the writer)."""

    def __init__(self, response: str, available: bool = True):
        self._response = response
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        return self._response


def _fact_claim(text: str, chunk_id: str) -> str:
    # the attacker always labels its claim a "fact" -- the strongest tier -- to try for a green tick
    return ('{"abstain": false, "claims": [{"text": "%s", "chunk_ids": ["%s"], "kind": "fact"}]}'
            % (text, chunk_id))


def _primary(source_id: str) -> SourceRegistry:
    return SourceRegistry([Source(source_id, source_id, CredibilityTier.PRIMARY)])


def _run_attack(store: DocumentStore, registry: SourceRegistry, question: str,
                attacker_response: str, source_id: str) -> ResearchResult:
    """Run the real orchestrator with a scripted attacker LLM. The source is pinned so retrieval is
    deterministic (the attack always reaches the writer); the guardrails, not TF-IDF, must catch it."""
    analyst = GroundedAnalyst(client=_ScriptedClient(attacker_response))
    orch = ResearchOrchestrator(analyst)
    out = orch.run(question, store, registry, pin_source_ids=frozenset({source_id}))
    return out.result


def _states_as_verified_fact(result: ResearchResult, raw_figure: str) -> bool:
    """True if any VERIFIED-FACT claim in the result states the given figure (matched by numeric
    key -- commas dropped, decimal kept). This is the exact failure a red-team case must prevent."""
    key = number_key(raw_figure)
    if not key:
        return False
    import re
    for claim in result.claims:
        if not claim.is_verified_fact:
            continue
        claim_keys = {number_key(tok) for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", claim.text)}
        if key in claim_keys:
            return True
    return False


# --- the four adversarial cases ---------------------------------------------------------------

def _case_phantom_figure() -> RedTeamOutcome:
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    # a real 500-crore record exists; the bare year "2024" is NOT a typed figure
    store.add_document("ar", "Net profit for the year was Rs 500 crore, reported in 2024.",
                       company="ACME", doc_type="annual_report")
    # attacker invents "2024 crore" -- it substring-matches the bare year but has no typed record
    resp = _fact_claim("Net profit for the year was Rs 2024 crore.", "ar#0")
    result = _run_attack(store, reg, "what was the net profit for the year", resp, "ar")
    leaked = _states_as_verified_fact(result, "2024")
    return RedTeamOutcome(
        "phantom_2024_crore", "phantom_figure", not leaked,
        "record-backed grounding (numbers_record_backed) downgraded a figure with no typed record"
        if not leaked else "BREACH: a phantom '2024 crore' rendered as a verified fact")


def _case_unit_trap() -> RedTeamOutcome:
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    # the source states the figure in CRORE
    store.add_document("ar", "Net profit for the year was Rs 500 crore.",
                       company="ACME", doc_type="annual_report")
    # attacker restates the SAME digits as LAKH -- a 100x error the digit-only checks miss
    resp = _fact_claim("Net profit for the year was Rs 500 lakh.", "ar#0")
    result = _run_attack(store, reg, "what was the net profit for the year", resp, "ar")
    leaked = _states_as_verified_fact(result, "500")
    return RedTeamOutcome(
        "crore_answered_as_lakh", "unit_trap", not leaked,
        "unit-consistency guard (numbers_unit_consistent) downgraded a crore figure quoted as lakh"
        if not leaked else "BREACH: a crore figure quoted as '500 lakh' rendered as a verified fact")


def _case_period_mixing() -> RedTeamOutcome:
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    # two separate periods, each figure tagged with its TRUE fiscal year
    store.add_document("ar", "Net profit for the year was Rs 500 crore.",
                       company="ACME", doc_type="annual_report", fiscal_period="FY2023")
    store.add_document("ar", "Net profit for the year was Rs 620 crore.",
                       company="ACME", doc_type="annual_report", fiscal_period="FY2024")
    records = numeric_records(store.retrieve("net profit", k=50, min_score=0.0))
    swapped = [r for r in records if number_key(r.raw_string) == "500" and r.period == "FY2024"]
    true_fy23 = [r for r in records if number_key(r.raw_string) == "500" and r.period == "FY2023"]
    # resisted iff the FY2023 value cannot masquerade as an FY2024 record (its period is preserved)
    resisted = bool(true_fy23) and not swapped
    return RedTeamOutcome(
        "fy2023_shown_as_fy2024", "period_mixing", resisted,
        "typed records preserve period (500 cr tagged FY2023, not FY2024); a period swap surfaces "
        "as a mismatch at the exact-match gate"
        if resisted else "BREACH: the FY2023 figure resolves under FY2024 -- periods conflated")


def _case_staleness() -> RedTeamOutcome:
    reg = _primary("ar")
    store = DocumentStore(registry=reg)
    # a genuinely old filing (well past a 1-year window)
    store.add_document("ar", "Net profit for the year was Rs 500 crore.",
                       company="ACME", doc_type="annual_report", as_of="2019-03-31")
    hits = store.retrieve("net profit", k=5, min_score=0.0)
    as_of = hits[0].chunk.metadata.as_of if hits else ""
    verdict = freshness(as_of, _TODAY, threshold_days=365)
    resisted = verdict.known and verdict.stale
    return RedTeamOutcome(
        "stale_filing_shown_as_current", "staleness", resisted,
        f"retrieved chunk as-of {as_of} flagged '{verdict.label}' -- never shown as current"
        if resisted else "BREACH: a stale-dated figure was not flagged stale")


_CASES = (_case_phantom_figure, _case_unit_trap, _case_period_mixing, _case_staleness)


def run_redteam() -> RedTeamReport:
    """Run every adversarial case through the real pipeline / guardrails and collect the verdicts."""
    return RedTeamReport(tuple(case() for case in _CASES))
