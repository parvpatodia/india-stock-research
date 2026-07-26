"""W5 red-team / adversarial suite (SPEC v4 §4 W5, §3 anti-hallucination).

Each adversarial class is driven through the REAL pipeline (orchestrator / grounded_analyst with
an injected fake LLM that TRIES to emit the wrong number) or the real guardrail primitive, and the
system MUST resist: downgrade to UNVERIFIED / withhold / flag stale, never a confident wrong
number rendered as a verified fact. Deterministic, offline (fake clients), so it can gate a build.

Classes covered (one case each): unit trap (crore/lakh), period mixing (FY23/FY24), phantom
figure (a number with no typed record), and an as-of/staleness violation.
"""
from src.eval.redteam import (
    RedTeamOutcome,
    RedTeamReport,
    run_redteam,
)


def _by_class(report: RedTeamReport, attack_class: str) -> RedTeamOutcome:
    matches = [o for o in report.outcomes if o.attack_class == attack_class]
    assert matches, f"no red-team case for class {attack_class!r}"
    return matches[0]


def test_run_redteam_covers_all_four_adversarial_classes():
    report = run_redteam()
    classes = {o.attack_class for o in report.outcomes}
    assert {"unit_trap", "period_mixing", "phantom_figure", "staleness"} <= classes


def test_phantom_figure_is_downgraded_not_shown_as_fact():
    # A bare-year phantom "2024 crore" (no typed record) must not ride into a verified fact.
    out = _by_class(run_redteam(), "phantom_figure")
    assert out.resisted, out.detail


def test_unit_trap_crore_answered_as_lakh_is_caught():
    # Source states "500 crore"; the attacker answers "500 lakh" -- same digits, 100x error. The
    # digit-only record-backed check passes it, so the unit-consistency guard must catch the scale
    # conflict and downgrade. This case is RED until numbers_unit_consistent exists and is wired.
    out = _by_class(run_redteam(), "unit_trap")
    assert out.resisted, out.detail


def test_period_mixing_is_caught_by_record_provenance():
    # An FY2023 figure must never be presentable as the FY2024 figure: the typed record preserves
    # the true period, so a swap surfaces as a mismatch (the exact-match gate leverages this).
    out = _by_class(run_redteam(), "period_mixing")
    assert out.resisted, out.detail


def test_staleness_violation_is_flagged_stale_not_current():
    # A stale-dated retrieved chunk must be flagged stale by the W1 freshness guardrail, never
    # silently shown as current.
    out = _by_class(run_redteam(), "staleness")
    assert out.resisted, out.detail


def test_period_mixing_at_claim_time_is_downgraded_not_shown_as_fact():
    # H2 (SPEC v4 §2.2): the record layer preserves the true period, but a model can still ANSWER an
    # FY2024 question with FY2023's figure -- it resolves to a real FY2023 record and passes the
    # digit-only checks. Period-aware verification must downgrade it at CLAIM time. Driven through the
    # real orchestrator with a scripted attacker.
    by_name = {o.name: o for o in run_redteam().outcomes}
    out = by_name["fy2023_figure_as_fy2024_answer"]
    assert out.resisted, out.detail


def test_report_all_resisted_is_true_when_every_case_resists():
    report = run_redteam()
    assert report.all_resisted, [o.detail for o in report.breaches]
    assert report.resisted == report.total
    assert report.total >= 4


def test_report_breaches_lists_only_unresisted_cases():
    # A synthetic report with one breach: breaches lists exactly it, all_resisted is False.
    breached = RedTeamReport((
        RedTeamOutcome("a", "phantom_figure", True, "ok"),
        RedTeamOutcome("b", "unit_trap", False, "leaked a wrong number"),
    ))
    assert not breached.all_resisted
    assert [o.name for o in breached.breaches] == ["b"]
    assert breached.resisted == 1 and breached.total == 2
