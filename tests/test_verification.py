from src.research.verification import (
    SourcedValue,
    VerificationStatus,
    verify_figure,
    verify_identity,
)


def _sv(value, source, loc=""):
    return SourcedValue(value=value, source_id=source, locator=loc)


def test_two_independent_sources_agree_is_verified():
    f = verify_figure("FY24 net profit (cr)", [_sv(79000, "annual_report"), _sv(79010, "screener")])
    assert f.status == VerificationStatus.VERIFIED
    assert f.is_trustworthy
    assert abs(f.value - 79005) < 1  # median of the two


def test_two_independent_sources_disagree_is_conflict():
    f = verify_figure("revenue", [_sv(100.0, "annual_report"), _sv(140.0, "moneycontrol")])
    assert f.status == VerificationStatus.CONFLICT
    assert f.value is None
    assert not f.is_trustworthy  # a conflicting number is never shown as fact


def test_single_source_is_usable_but_not_trustworthy():
    f = verify_figure("debt", [_sv(110.0, "annual_report")])
    assert f.status == VerificationStatus.SINGLE_SOURCE
    assert f.value == 110.0
    assert not f.is_trustworthy


def test_same_source_twice_is_not_independent():
    f = verify_figure("eps", [_sv(12.0, "screener", "a"), _sv(12.0, "screener", "b")])
    assert f.status == VerificationStatus.SINGLE_SOURCE  # one distinct source, not two
    assert not f.is_trustworthy


def test_no_value_is_unverifiable():
    f = verify_figure("x", [_sv(None, "annual_report")])
    assert f.status == VerificationStatus.UNVERIFIABLE and f.value is None


def test_handles_negative_values_loss():
    f = verify_figure("net loss", [_sv(-500.0, "annual_report"), _sv(-501.0, "tickertape")])
    assert f.status == VerificationStatus.VERIFIED and f.value < 0


def test_verify_identity_parts_sum_to_total():
    f = verify_identity(
        "segment revenue sums to total",
        total=_sv(1000.0, "annual_report"),
        parts=[_sv(600.0, "annual_report"), _sv(400.0, "annual_report")],
    )
    assert f.status == VerificationStatus.VERIFIED


def test_verify_identity_mismatch_is_conflict():
    f = verify_identity(
        "segment revenue sums to total",
        total=_sv(1000.0, "annual_report"),
        parts=[_sv(600.0, "annual_report"), _sv(250.0, "annual_report")],
    )
    assert f.status == VerificationStatus.CONFLICT and f.value is None
