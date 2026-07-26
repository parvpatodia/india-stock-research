"""Explicit plain-Python research orchestrator (SPEC v4 §4 W4).

DECISION (locked): a PLAIN-PYTHON orchestrator, NOT LangGraph/CrewAI/AutoGen. Maximum
transparency and zero framework risk for a free-deploy, real-money solo tool: an explicit STATE
object flows through discrete, deterministic stages, each a small pure-ish function that reads
the state and returns a new one, appending a trace entry for debuggability.

    PLAN     -> what does the question need? (growth / margin / CAGR intent)
    RETRIEVE -> grounding over the store (the same query expansion GroundedAnalyst uses)
    COMPUTE  -> pre-derive any needed growth / CAGR from the retrieved TYPED records, in Python
                (src/research/computed_figures.py) -- the model never does the arithmetic
    VERIFY   -> the hard abstain GATE: require grounded, record-aware evidence; on empty
                retrieval, no LLM, or a derivation with no typed record to compute from, ABSTAIN
    WRITE    -> hand the pre-computed ComputedFigures to GroundedAnalyst.write_answer so the model
                PHRASES them and cites the underlying sources -- never a fabricated number

The orchestrator COMPOSES existing code: it reuses DocumentStore retrieval, the numeric_records
seam, the computed_figures compute-don't-generate primitives, and GroundedAnalyst for the cited
answer. It rewrites none of them. The result is a structured OrchestrationResult carrying the
answer-or-abstention (a ResearchResult), the plan, the ComputedFigures used, and a per-stage
trace.

MIXED-FIGURE GUARD (real money): typed NumericRecords carry provenance (company / unit / period /
source_doc) but NOT a semantic figure label, so COMPUTE groups a series by (company, source_doc,
unit). Two same-unit figures (revenue and net profit, both rupees) could otherwise land in one
group and be composed into a meaningless CAGR/growth. The guard: a fiscal year that contributes
MORE THAN ONE typed record to a group is AMBIGUOUS (which figure?) and is dropped from the series
(_single_record_years) -- so a multi-figure chunk (the common case: "Revenue ... Net profit ..."
in the same period) can never seed a mixed derivation; the seam withholds rather than risk a wrong
number. RESIDUAL (irreducible without a figure label, disclosed honestly): if EACH year's chunk
mentions exactly ONE, but a DIFFERENT, rupee figure (FY21 revenue-only, FY23 profit-only), the
guard cannot tell them apart. That configuration is unlikely to be retrieved together (a
profit-only chunk shares little vocabulary with a revenue query), and even then a mixed result is
not a literal source substring so numbers_grounded/numbers_record_backed downgrade it to a
cautioned UNVERIFIED (never a green fact) and the human-in-the-loop review gate still stands. A
figure-labelled COMPUTE is future work. COMPUTE stays conservative: it derives only a CAGR (a
self-guarding multi-year series) and a latest-two-year YoY growth, and never a margin (part/whole
ordering is genuinely ambiguous without a label, and a reversed margin is a wrong real-money
number).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from .claims import ResearchResult
from .computed_figures import (
    ComputedFigure,
    _period_year,
    cagr_from_records,
    growth_between,
)
from .grounded_analyst import GroundedAnalyst
from .grounding import DocumentStore, RetrievedChunk, numeric_records
from .numeric_records import NumericRecord
from .structure import detect_period

# WHY module-level keyword sets (real money, transparency): the plan is a deterministic, auditable
# intent read, not an LLM call. A reviewer can see exactly which words trigger a derived-figure
# need. Substring match on the lowercased question keeps it dependency-light and debuggable.
_CAGR_WORDS = ("cagr", "compound annual", "compounded annual", "compound growth")
_GROWTH_WORDS = ("growth", "grow", "grew", "grown", "yoy", "year-over-year",
                 "year on year", "increase", "increased", "rise", "rose", "risen")
_MARGIN_WORDS = ("margin",)


@dataclass(frozen=True)
class Plan:
    """What the question needs, as booleans a reviewer can audit. `derivation_requested` is the
    gate/compute trigger: the question asks for a figure that must be COMPUTED, not just quoted.
    `target_period` (H2, SPEC v4 §2.2) is the fiscal year the question targets ('FY2024') or None --
    a deterministic, auditable read used to keep a figure from ANOTHER year off this year's answer."""
    wants_growth: bool
    wants_cagr: bool
    wants_margin: bool
    target_period: str | None = None

    @property
    def derivation_requested(self) -> bool:
        return self.wants_growth or self.wants_cagr or self.wants_margin


@dataclass(frozen=True)
class StageTrace:
    """One stage's outcome, for debuggability. `ok` is False when a stage failed/gated."""
    stage: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class OrchestrationResult:
    """The orchestrator's structured output: the answer-or-abstention plus everything needed to
    audit HOW it got there (the plan, the ComputedFigures used, a per-stage trace)."""
    question: str
    result: ResearchResult
    plan: Plan
    computed: tuple[ComputedFigure, ...]
    trace: tuple[StageTrace, ...]

    @property
    def abstained(self) -> bool:
        return self.result.abstained


@dataclass(frozen=True)
class OrchestratorState:
    """The explicit state threaded through the stages. Frozen: each stage returns a NEW state via
    `replace`, so the flow is a pure sequence of transformations that is trivial to inspect."""
    question: str
    as_of: str | None = None
    k: int = 5
    pin_source_ids: frozenset[str] = frozenset()
    retrieval_hint: str = ""
    plan: Plan | None = None
    retrieved: tuple[RetrievedChunk, ...] = ()
    records: tuple[NumericRecord, ...] = ()
    computed: tuple[ComputedFigure, ...] = ()
    gate_reason: str | None = None
    result: ResearchResult | None = None
    trace: tuple[StageTrace, ...] = ()

    def log(self, stage: str, ok: bool, detail: str) -> "OrchestratorState":
        return replace(self, trace=self.trace + (StageTrace(stage, ok, detail),))

    def gated(self, reason: str, detail: str) -> "OrchestratorState":
        """Trip the abstain gate: record the reason and mark VERIFY as not-ok."""
        return replace(self, gate_reason=reason).log("VERIFY", False, detail)

    def finished(self, result: ResearchResult, detail: str) -> "OrchestratorState":
        return replace(self, result=result).log("WRITE", not result.abstained, detail)


def plan_question(question: str) -> Plan:
    """Deterministically read the question's intent for a DERIVED figure AND its target fiscal
    period. Substring keyword match on the lowercased text + the shared structure.detect_period FY
    parser; no LLM, no network, fully auditable."""
    q = (question or "").lower()
    return Plan(
        wants_growth=any(w in q for w in _GROWTH_WORDS),
        wants_cagr=any(w in q for w in _CAGR_WORDS),
        wants_margin=any(w in q for w in _MARGIN_WORDS),
        target_period=detect_period(question or ""),
    )


def _plan(state: OrchestratorState) -> OrchestratorState:
    plan = plan_question(state.question)
    wants = [name for name, on in (("growth", plan.wants_growth), ("cagr", plan.wants_cagr),
                                   ("margin", plan.wants_margin)) if on]
    period = f"; target period {plan.target_period}" if plan.target_period else ""
    detail = f"derived-figure need: {', '.join(wants) or 'none'}{period}"
    return replace(state, plan=plan).log("PLAN", True, detail)


def _retrieve(state: OrchestratorState, store: DocumentStore) -> OrchestratorState:
    # WHY the same query expansion as GroundedAnalyst.answer (composition, not divergence): a
    # natural question ("what is the recent news") shares no words with a specific headline, so the
    # caller's company hint is appended to the RETRIEVAL query only. Kept identical so the
    # orchestrated path retrieves exactly what the direct answer path would.
    hint = state.retrieval_hint.strip()
    query = f"{state.question} {state.retrieval_hint}".strip() if hint else state.question
    hits = tuple(store.retrieve(query, k=state.k, pin_source_ids=state.pin_source_ids))
    records = tuple(numeric_records(list(hits)))
    return replace(state, retrieved=hits, records=records).log(
        "RETRIEVE", bool(hits), f"{len(hits)} chunk(s), {len(records)} typed record(s)")


def _single_record_years(records: list[NumericRecord]) -> dict[int, NumericRecord]:
    """Map fiscal year -> its ONE typed record, reusing computed_figures._period_year so this seam
    and the CAGR series parse the period identically (one FY parser, no 4th copy).

    WHY drop a year with >1 record (real money, no mixed-figure math): records carry no semantic
    figure label, so two rupee numbers for the SAME year in a (company, source_doc, unit) group
    could be revenue AND net profit. Composing a CAGR/growth across a year whose figure is
    ambiguous risks a meaningless number handed to the model as 'already computed'. Withhold that
    year instead -- a multi-figure chunk ('Revenue ... Net profit ...' in one period) then can
    never seed a mixed derivation."""
    by_year: dict[int, list[NumericRecord]] = defaultdict(list)
    for r in records:
        year = _period_year(r.period)
        if year is not None:
            by_year[year].append(r)
    return {year: recs[0] for year, recs in by_year.items() if len(recs) == 1}


def _derive_figures(records: tuple[NumericRecord, ...], plan: Plan) -> list[ComputedFigure]:
    """Pre-derive the needed growth / CAGR from typed records, in Python. Grouped by
    (company, source_doc, unit); a year with more than one record in a group is dropped as
    ambiguous (see _single_record_years and the module docstring). Only what the plan asked for is
    computed; a margin is never auto-derived (ambiguous part/whole)."""
    figures: list[ComputedFigure] = []
    groups: dict[tuple, list[NumericRecord]] = defaultdict(list)
    for r in records:
        groups[(r.company, r.source_doc, r.unit)].append(r)
    for recs in groups.values():
        by_year = _single_record_years(recs)
        # CAGR: a genuinely multi-year (>=3 unambiguous FY) single-figure series. Passing only the
        # one-record-per-year records means series_from_records sees a clean series; cagr_from_records
        # delegates to analysis.trends.cagr and withholds unless endpoints are positive.
        if plan.wants_cagr and len(by_year) >= 3:
            fig = cagr_from_records([by_year[y] for y in sorted(by_year)])
            if fig is not None:
                figures.append(fig)
        # YoY growth: only between the latest two ADJACENT unambiguous years, so the "year-over-year"
        # label is honest (a non-adjacent delta would be mislabelled). Skipped when the prior year is
        # absent or ambiguous.
        if plan.wants_growth and by_year:
            latest = max(by_year)
            if (latest - 1) in by_year:
                fig = growth_between(by_year[latest - 1], by_year[latest])
                if fig is not None:
                    figures.append(fig)
    return figures


def _compute(state: OrchestratorState) -> OrchestratorState:
    plan = state.plan
    figures: list[ComputedFigure] = []
    if plan is not None and (plan.wants_growth or plan.wants_cagr):
        figures = _derive_figures(state.records, plan)
    labels = ", ".join(f.label for f in figures) or "none"
    return replace(state, computed=tuple(figures)).log(
        "COMPUTE", True, f"derived {len(figures)} figure(s): {labels}")


def _verify(state: OrchestratorState, *, available: bool) -> OrchestratorState:
    """The hard abstain GATE. Every arm returns an honest 'insufficient verified evidence'
    abstention naming what is missing -- never a fabricated answer (SPEC v4 §2 decision #3)."""
    if not state.retrieved:
        return state.gated(
            "Insufficient verified evidence: no source in the library matched this question. "
            "Add a relevant primary source (annual report, filing, exchange/AMFI data) and ask "
            "again.",
            "empty retrieval")
    if not available:
        return state.gated(
            "Insufficient verified evidence: sources matched, but no LLM is configured to phrase a "
            "grounded answer. Set LLM_MODEL and ask again.",
            "no LLM available")
    if state.plan is not None and state.plan.derivation_requested and not state.records:
        # WHY (real money, compute-don't-generate): a growth/margin/CAGR answer must be COMPUTED in
        # Python from typed records. With none retrieved, the only way to produce a number would be
        # to let the model do the arithmetic -- exactly the forbidden path. Abstain honestly.
        return state.gated(
            "Insufficient verified evidence: this question needs a computed figure "
            "(growth/margin/CAGR) but no typed numeric record was retrieved to compute it from. "
            "The system will not let the model do the arithmetic itself.",
            "derivation requested but no typed records to compute from")
    return state.log("VERIFY", True, "gate passed: grounded, record-aware evidence present")


def _write(state: OrchestratorState, analyst: GroundedAnalyst,
           registry) -> OrchestratorState:
    if state.gate_reason is not None:
        return state.finished(
            ResearchResult.abstain(state.question, state.gate_reason), "skipped: gate abstained")
    result = analyst.write_answer(
        state.question, list(state.retrieved), registry, as_of=state.as_of,
        computed_figures=state.computed,
        target_period=state.plan.target_period if state.plan else None)
    if result.abstained:
        detail = f"writer abstained: {result.abstain_reason}"
    else:
        detail = (f"answered with {len(result.claims)} claim(s); "
                  f"{len(state.computed)} pre-computed figure(s) supplied")
    return state.finished(result, detail)


class ResearchOrchestrator:
    """Runs the explicit PLAN -> RETRIEVE -> COMPUTE -> VERIFY -> WRITE pipeline. Inject a
    GroundedAnalyst (with a fake client) for offline, deterministic testing."""

    def __init__(self, analyst: GroundedAnalyst | None = None):
        self.analyst = analyst or GroundedAnalyst()

    def run(self, question: str, store: DocumentStore, registry, *, k: int = 5,
            as_of: str | None = None, pin_source_ids: frozenset[str] = frozenset(),
            retrieval_hint: str = "") -> OrchestrationResult:
        state = OrchestratorState(question=question, as_of=as_of, k=k,
                                  pin_source_ids=pin_source_ids, retrieval_hint=retrieval_hint)
        state = _plan(state)
        state = _retrieve(state, store)
        state = _compute(state)
        state = _verify(state, available=self.analyst.available)
        state = _write(state, self.analyst, registry)
        assert state.result is not None  # _write always sets a result (answer or abstention)
        return OrchestrationResult(question=question, result=state.result, plan=state.plan,
                                   computed=state.computed, trace=state.trace)
