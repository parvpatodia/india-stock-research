import pytest

from src.research.report import (
    VERDICT_CAVEAT,
    Confidence,
    Leaning,
    QualityTier,
    Report,
    ReviewStatus,
    ValuationTier,
    Verdict,
)
from src.research.verification import SourcedValue, verify_figure


def _verdict():
    return Verdict(ValuationTier.FAIR, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                   Confidence.MEDIUM, reasons=("ROCE rising per FY24 AR p.12",))


def _verified_fig():
    return verify_figure("net profit", [SourcedValue(79000, "ar"), SourcedValue(79010, "screener")])


def _conflict_fig():
    return verify_figure("revenue", [SourcedValue(100, "ar"), SourcedValue(140, "moneycontrol")])


def test_new_report_is_draft_and_not_trusted():
    r = Report(company="Acme", verdict=_verdict())
    assert r.status == ReviewStatus.DRAFT
    assert r.is_trusted is False
    assert r.created_at  # stamped


def test_verdict_always_carries_caveat():
    assert _verdict().caveat == VERDICT_CAVEAT


def test_approve_makes_it_trusted_with_audit():
    r = Report(company="Acme", figures=(_verified_fig(),), verdict=_verdict())
    approved = r.approve(reviewer="expert@dad", note="checked, agree")
    assert approved.is_trusted
    assert approved.status == ReviewStatus.APPROVED
    assert approved.audit[-1].reviewer == "expert@dad"
    assert approved.audit[-1].timestamp
    # original is unchanged (immutable)
    assert r.status == ReviewStatus.DRAFT


def test_reject_captures_corrections_and_is_not_trusted():
    r = Report(company="Acme", verdict=_verdict())
    rejected = r.reject(reviewer="expert", note="wrong debt figure",
                        corrections=("debt is 90 cr not 110 cr per AR p.also",))
    assert rejected.status == ReviewStatus.REJECTED
    assert rejected.is_trusted is False
    assert "debt is 90 cr" in rejected.audit[-1].corrections[0]


def test_review_requires_named_reviewer():
    r = Report(company="Acme")
    with pytest.raises(ValueError):
        r.approve(reviewer="")


def test_approve_blocked_while_conflicts_unless_acknowledged():
    r = Report(company="Acme", figures=(_verified_fig(), _conflict_fig()))
    assert len(r.conflicts) == 1
    assert len(r.uncrossverified) == 1
    with pytest.raises(ValueError):
        r.approve(reviewer="expert")  # must not silently approve over a conflict
    ok = r.approve(reviewer="expert", note="revenue conflict is a units issue, verified by hand",
                   acknowledge_conflicts=True)
    assert ok.is_trusted


def test_audit_trail_preserves_order():
    r = Report(company="Acme", verdict=_verdict())
    r2 = r.reject(reviewer="expert", note="fix x")
    r3 = r2.approve(reviewer="expert", note="fixed now")
    assert [e.status for e in r3.audit] == [ReviewStatus.REJECTED, ReviewStatus.APPROVED]
