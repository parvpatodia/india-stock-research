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


def test_two_of_three_consensus_verifies_and_names_outlier():
    # yfinance and the annual report agree; Screener is the odd one out (different definition).
    f = verify_figure("net profit", [
        SourcedValue(80000, "yfinance"),
        SourcedValue(95000, "screener"),
        SourcedValue(80100, "annual_report"),
    ])
    assert f.status == VerificationStatus.VERIFIED
    assert abs(f.value - 80050) < 200          # median of the agreeing cluster
    assert "screener" in f.note                # outlier is named/withheld


def test_three_way_all_disagree_is_conflict():
    f = verify_figure("x", [SourcedValue(100, "a"), SourcedValue(130, "b"), SourcedValue(170, "c")])
    assert f.status == VerificationStatus.CONFLICT and f.value is None


def test_chained_agreement_does_not_verify_a_pair_that_actually_disagrees():
    # WHY (real money, HIGH severity): app.py's Research tab wires yfinance + Screener + the
    # annual report (a real, live 3-source scenario, not hypothetical -- the annual report is
    # auto-added "as a third source to break ties" whenever an LLM is configured). The old
    # clustering picked the LARGEST "star" around a single pivot value (each member merely close
    # to that ONE pivot), not a true clique where EVERY member is mutually close to EVERY other
    # member. A=100, B=101.9, C=103.8 with the default 2% tolerance: A-B agree (1.9 <= 2.04) and
    # B-C agree (1.9 <= 2.08), so the old code reported "3 independent sources agree" and took
    # their median -- but A and C themselves are 3.8% apart, genuinely beyond tolerance (3.8 >
    # 2.08), and never checked against each other. That is exactly the "independent sources
    # disagree beyond tolerance" case this module exists to catch, smuggled through by chaining.
    a_c_gap_pct = abs(100.0 - 103.8) / 103.8
    assert a_c_gap_pct > 0.02   # confirms A and C genuinely disagree beyond the 2% tolerance
    f = verify_figure("net_profit", [
        SourcedValue(100.0, "yfinance"),
        SourcedValue(101.9, "screener"),
        SourcedValue(103.8, "annual_report"),
    ])
    # Must never claim all 3 agree -- that would only be true if A and C were also mutually
    # close, which they are not. A real clique here can only be the 2-source pair {A,B} or
    # {B,C} (median 100.95 or 102.85), never the old chained 3-way median of 101.9.
    assert "3 independent sources agree" not in f.note
    if f.status == VerificationStatus.VERIFIED:
        assert f.value != 101.9


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
