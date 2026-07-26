"""W4 orchestration (SPEC v4 §4): the plain-Python explicit orchestrator.

No agent framework -- an explicit STATE flows through discrete, deterministic stages
PLAN -> RETRIEVE -> COMPUTE -> VERIFY(gate) -> WRITE. These tests pin: the plan intent
detection, that each stage records a trace, that COMPUTE pre-derives growth/CAGR from typed
records and hands the RESULT to the model (compute-don't-generate, not raw operands), and that
the hard abstain GATE withholds an answer on empty retrieval / no LLM / a derivation with no
typed record to compute from -- never a fabricated answer.
"""
from src.llm.client import LLMClient
from src.research.claims import ResearchResult
from src.research.computed_figures import ComputedFigure
from src.research.grounded_analyst import GroundedAnalyst
from src.research.grounding import DocumentStore
from src.research.orchestrator import (
    OrchestrationResult,
    Plan,
    ResearchOrchestrator,
    plan_question,
)
from src.sources.registry import CredibilityTier, Source, SourceRegistry


class CapturingClient(LLMClient):
    """A fake client that records the exact user prompt it was handed, so a test can prove the
    model received a PRE-COMPUTED value (compute-don't-generate) rather than raw operands."""

    def __init__(self, response: str, available: bool = True):
        self._response = response
        self._available = available
        self.last_user: str | None = None
        self.last_system: str | None = None

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        self.last_system = system
        self.last_user = user
        return self._response


def _claim_payload(text: str, cid: str, kind: str = "fact") -> str:
    return ('{"abstain": false, "claims": [{"text": "%s", "chunk_ids": ["%s"], "kind": "%s"}]}'
            % (text, cid, kind))


def _prose_store():
    reg = SourceRegistry([Source("amfi", "AMFI", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    store.add_document("amfi", "A SIP invests a fixed amount every month into a mutual fund scheme.")
    return store, reg


def _series_store():
    """A store with a real multi-year rupee series for ONE figure of ONE company, so the COMPUTE
    stage can derive a CAGR / YoY growth from typed records. 100 -> 121 over two years = 10%/yr."""
    reg = SourceRegistry([Source("ar", "Annual Report", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    for period, value in (("FY2021", "100"), ("FY2022", "110"), ("FY2023", "121")):
        store.add_document("ar", f"Net profit was Rs {value} crore.",
                           company="RELIANCE", fiscal_period=period)
    return store, reg


# --- PLAN -------------------------------------------------------------------------------------

def test_plan_detects_cagr_growth_and_margin_intents():
    assert plan_question("What is the revenue CAGR over five years?").wants_cagr
    assert plan_question("How fast has net profit grown?").wants_growth
    assert plan_question("What was the operating margin?").wants_margin
    plain = plan_question("What does this company do?")
    assert not plain.wants_cagr and not plain.wants_growth and not plain.wants_margin
    assert not plain.derivation_requested


def test_plan_derivation_requested_is_any_derived_figure():
    assert plan_question("net profit growth").derivation_requested
    assert plan_question("compound annual growth rate").derivation_requested
    assert not plan_question("who is the CEO").derivation_requested


# --- happy path: grounded answer + full trace -------------------------------------------------

def test_orchestrator_end_to_end_grounded_answer_records_every_stage():
    store, reg = _prose_store()
    client = CapturingClient(_claim_payload(
        "A SIP invests a fixed amount monthly.", "amfi#0"))
    orch = ResearchOrchestrator(GroundedAnalyst(client=client))
    out = orch.run("what is a SIP mutual fund", store, reg)
    assert isinstance(out, OrchestrationResult)
    assert isinstance(out.result, ResearchResult)
    assert not out.abstained
    assert out.result.claims[0].citations[0].source_id == "amfi"
    # every stage is traced, in order
    assert [t.stage for t in out.trace] == ["PLAN", "RETRIEVE", "COMPUTE", "VERIFY", "WRITE"]
    assert out.trace[-1].ok  # WRITE succeeded


# --- the hard abstain GATE (never a fabricated answer) ----------------------------------------

def test_gate_abstains_on_empty_retrieval():
    store, reg = _prose_store()
    client = CapturingClient(_claim_payload("should not be used", "amfi#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "quantum chromodynamics gluon lattice", store, reg)
    assert out.abstained
    assert "insufficient verified evidence" in (out.result.abstain_reason or "").lower()
    # the writer must never have been called on an empty context
    assert client.last_user is None
    verify_trace = [t for t in out.trace if t.stage == "VERIFY"][0]
    assert not verify_trace.ok


def test_gate_abstains_without_an_llm():
    store, reg = _prose_store()
    client = CapturingClient("", available=False)
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what is a SIP mutual fund", store, reg)
    assert out.abstained
    assert "llm" in (out.result.abstain_reason or "").lower()
    assert client.last_user is None  # writer not called


def test_gate_abstains_on_a_derivation_question_with_no_typed_records():
    # WHY (real money, compute-don't-generate): a growth/CAGR question needs typed numeric records
    # to compute from IN PYTHON. If retrieval yields only prose with no extractable figure, the
    # ONLY way to answer numerically would be to let the model do the arithmetic -- forbidden. The
    # gate abstains honestly instead of risking a fabricated computed number. The prose here MATCHES
    # the question (so retrieval succeeds) but carries no numbers, so this is the derivation gate,
    # not the empty-retrieval gate.
    reg = SourceRegistry([Source("amfi", "AMFI", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    store.add_document(
        "amfi", "The company revenue has grown across its business segments over the years.")
    client = CapturingClient(_claim_payload("nope", "amfi#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what is the revenue growth over the years", store, reg)
    assert out.abstained
    assert not out.result.claims                                # nothing fabricated
    assert "compute" in (out.result.abstain_reason or "").lower()
    assert client.last_user is None


def test_non_derivation_question_on_the_same_prose_store_is_answered_not_gated():
    # the gate above must be SPECIFIC to derivation questions -- an ordinary question on the same
    # record-less prose store still answers normally (no over-abstaining).
    store, reg = _prose_store()
    client = CapturingClient(_claim_payload("A SIP invests monthly.", "amfi#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what is a SIP mutual fund", store, reg)
    assert not out.abstained


# --- COMPUTE: pre-derive the figure and hand the RESULT to the model --------------------------

def test_orchestrator_computes_cagr_and_hands_the_result_to_the_model():
    store, reg = _series_store()
    client = CapturingClient(_claim_payload(
        "Net profit compounded about 10% a year.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "What is the net profit CAGR over these years?", store, reg)
    # a CAGR figure was pre-computed deterministically
    assert any(isinstance(f, ComputedFigure) and "compound annual growth rate" in f.label
               for f in out.computed)
    cagr_fig = next(f for f in out.computed if "compound annual growth rate" in f.label)
    assert round(cagr_fig.value, 2) == 10.0
    # PROOF the model was handed the finished VALUE (10.00%), framed as already-computed, not two
    # raw operands to divide itself
    assert client.last_user is not None
    assert "10.00%" in client.last_user
    assert "already computed" in client.last_user.lower()


def test_orchestrator_computes_yoy_growth_between_the_latest_two_years():
    store, reg = _series_store()
    client = CapturingClient(_claim_payload("Net profit rose about 10%.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "How much did net profit grow last year?", store, reg)
    assert any("year-over-year growth" in f.label for f in out.computed)
    growth = next(f for f in out.computed if "year-over-year growth" in f.label)
    # latest two years: 110 -> 121 = 10%
    assert round(growth.value, 2) == 10.0
    assert "10.00%" in (client.last_user or "")


def test_compute_never_mixes_two_figures_from_the_same_period_into_one_series():
    # WHY (real money, adversarial-review regression): typed records carry no figure label, so a
    # (company, source_doc, unit) group could hold BOTH revenue and net profit (both rupees). If a
    # year contributes >1 record it is ambiguous, and composing a CAGR/growth across it would emit a
    # MEANINGLESS number as an 'already computed' figure -- the exact confident-wrong-number failure
    # this app exists to prevent. Each period below mentions two rupee figures, so every year is
    # ambiguous and the guard withholds ALL derivation rather than fabricate a mixed series.
    reg = SourceRegistry([Source("ar", "Annual Report", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    for period, rev, profit in (("FY2021", "500", "100"), ("FY2022", "550", "110"),
                                ("FY2023", "600", "121")):
        store.add_document("ar", f"Revenue was Rs {rev} crore. Net profit was Rs {profit} crore.",
                           company="RELIANCE", fiscal_period=period)
    client = CapturingClient(_claim_payload("Revenue and profit both rose.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "What is the revenue CAGR and growth over these years?", store, reg)
    # no mixed CAGR/growth figure was fabricated
    assert out.computed == ()
    assert "already computed" not in (client.last_user or "").lower()
    # records DO exist, so the gate does not trip; the writer answers from grounded prose instead
    assert not out.abstained


def test_non_derivation_question_supplies_no_computed_figures():
    # a plain question over the same series store computes nothing (no derivation requested), so
    # the prompt carries no pre-computed block -- backward-compatible with the plain answer path.
    store, reg = _series_store()
    client = CapturingClient(_claim_payload("Net profit was 121 crore.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "What was the latest net profit?", store, reg)
    assert out.computed == ()
    assert "already computed" not in (client.last_user or "").lower()


def test_writer_abstention_propagates_as_an_honest_abstention():
    # if the writer itself abstains (model returns abstain), the orchestrator surfaces that, not a
    # fabricated answer.
    store, reg = _prose_store()
    client = CapturingClient('{"abstain": true, "reason": "cannot answer from sources"}')
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what is a SIP mutual fund", store, reg)
    assert out.abstained
    assert out.result.abstain_reason == "cannot answer from sources"


def test_plan_type_is_carried_on_the_result():
    store, reg = _series_store()
    client = CapturingClient(_claim_payload("x", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "net profit CAGR", store, reg)
    assert isinstance(out.plan, Plan)
    assert out.plan.wants_cagr


# --- H2 period awareness (SPEC v4 §2.2): a year-targeted question must not surface another year ---

def _period_series_store():
    """A store with net-profit figures for TWO distinct fiscal years, each chunk tagged with its
    TRUE period: FY2023 = 500 crore, FY2024 = 620 crore. This is the FY23/FY24 mixing setup."""
    reg = SourceRegistry([Source("ar", "Annual Report", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    store.add_document("ar", "Net profit for the year was Rs 500 crore.",
                       company="ACME", fiscal_period="FY2023")
    store.add_document("ar", "Net profit for the year was Rs 620 crore.",
                       company="ACME", fiscal_period="FY2024")
    return store, reg


def test_plan_detects_the_target_period_from_the_question():
    assert plan_question("what was the net profit in FY2024").target_period == "FY2024"
    assert plan_question("what was the net profit").target_period is None


def test_period_targeted_question_downgrades_a_different_years_figure():
    # the QUESTION targets FY2024; the model (adversarially) answers with FY2023's 500-crore figure.
    # It resolves to a real FY2023 record and passes the digit-only grounding/record/unit checks, so
    # the period guard is the ONLY thing that can stop it being shown as the verified FY2024 answer.
    store, reg = _period_series_store()
    client = CapturingClient(_claim_payload("Net profit was Rs 500 crore.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what was the net profit in FY2024", store, reg, pin_source_ids=frozenset({"ar"}))
    assert out.plan.target_period == "FY2024"
    assert not out.abstained                       # the figure is shown, just not as a verified fact
    assert not out.result.claims[0].is_verified_fact


def test_period_targeted_question_keeps_the_correct_years_figure_verified():
    # the control: the SAME question, answered with FY2024's OWN 620-crore figure, stays a clean
    # verified fact -- the guard must not over-downgrade the right-year answer (no coverage loss).
    store, reg = _period_series_store()
    client = CapturingClient(_claim_payload("Net profit was Rs 620 crore.", "ar#1"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what was the net profit in FY2024", store, reg, pin_source_ids=frozenset({"ar"}))
    assert out.result.claims[0].is_verified_fact


def test_no_period_in_question_leaves_the_same_figure_verified_unchanged():
    # backward-compat proof: the SAME store + SAME 500-crore claim, but the question names NO year,
    # so the period guard is inert and 500 verifies exactly as it did before H2.
    store, reg = _period_series_store()
    client = CapturingClient(_claim_payload("Net profit was Rs 500 crore.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what was the net profit", store, reg, pin_source_ids=frozenset({"ar"}))
    assert out.plan.target_period is None
    assert out.result.claims[0].is_verified_fact


def test_untagged_record_is_not_hidden_by_a_period_targeted_question():
    # conservatism (real money): a figure whose record carries NO period tag must NOT be withheld
    # from a year-targeted question -- unknown period is not a wrong period.
    reg = SourceRegistry([Source("ar", "Annual Report", CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=reg)
    store.add_document("ar", "Net profit for the year was Rs 500 crore.", company="ACME")
    client = CapturingClient(_claim_payload("Net profit was Rs 500 crore.", "ar#0"))
    out = ResearchOrchestrator(GroundedAnalyst(client=client)).run(
        "what was the net profit in FY2024", store, reg, pin_source_ids=frozenset({"ar"}))
    assert out.result.claims[0].is_verified_fact
