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
    most_recent_by_symbol,
)
from src.research.verification import SourcedValue, VerificationStatus, verify_figure


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


def _unverifiable_fig(name="net_profit"):
    return verify_figure(name, [])   # no sources at all -> UNVERIFIABLE


def test_no_data_found_true_when_every_figure_is_unverifiable():
    # WHY (real money, workflow): live-verified with a real, plausible mistake -- typing "PAGE"
    # instead of Page Industries' actual ticker "PAGEIND" returns UNVERIFIABLE for every single
    # figure from both sources. This is the exact shape that should trigger a "check the exact
    # symbol" hint instead of the generic "insufficient data" message.
    r = Report(company="PAGE", figures=(_unverifiable_fig("net_profit"), _unverifiable_fig("revenue")))
    assert r.no_data_found is True


def test_no_data_found_true_when_figures_is_completely_empty():
    # all() on an empty tuple is True -- zero figures at all is ALSO "found nothing", not
    # something that should slip through the check unnoticed.
    r = Report(company="X", figures=())
    assert r.no_data_found is True


def test_no_data_found_false_when_anything_resolved_at_all():
    # Even a single-source or conflicting figure means the symbol IS real and has SOME data --
    # ordinary thin coverage, not the "likely wrong symbol" case.
    r = Report(company="X", figures=(_verified_fig(),))
    assert r.no_data_found is False
    r2 = Report(company="X", figures=(_conflict_fig(), _unverifiable_fig("revenue")))
    assert r2.no_data_found is False


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


def test_most_recent_by_symbol_uses_timestamp_not_dict_position():
    # WHY (regression, real money): reproduces the actual bug -- key A (no AR override) is
    # inserted first, key B (AR-override run, a different label/key) inserted after. The user
    # then goes BACK to the no-override flow, updating key A in place with a LATER created_at,
    # but A keeps its EARLIER dict position (Python dicts don't reorder on update). A naive
    # "last match seen while iterating" pick would wrongly return B (the stale one).
    reports = {
        "RELIANCE (live/yfinance + screener)":
            Report(company="RELIANCE", created_at="2026-01-03T00:00:00Z"),
        "RELIANCE (live/yfinance + screener + annual report)":
            Report(company="RELIANCE", created_at="2026-01-02T00:00:00Z"),
    }
    result = most_recent_by_symbol(reports, "RELIANCE")
    assert result.created_at == "2026-01-03T00:00:00Z"


def test_most_recent_by_symbol_filters_by_symbol_prefix():
    reports = {
        "RELIANCE (live/x)": Report(company="RELIANCE", created_at="2026-01-01T00:00:00Z"),
        "TCS (live/x)": Report(company="TCS", created_at="2026-01-05T00:00:00Z"),
    }
    result = most_recent_by_symbol(reports, "RELIANCE")
    assert result is not None and result.company == "RELIANCE"


def test_most_recent_by_symbol_none_when_no_match():
    assert most_recent_by_symbol({}, "RELIANCE") is None
    assert most_recent_by_symbol({"TCS (live/x)": Report(company="TCS")}, "RELIANCE") is None
