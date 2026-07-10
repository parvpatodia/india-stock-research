from src.analysis.framework import (
    assemble_verdict,
    earnings_quality,
    leverage_health,
    promoter_pledge,
    valuation_vs_history,
    value_if_trustworthy,
)
from src.research.report import Confidence, Leaning, QualityTier, ValuationTier
from src.research.verification import SourcedValue, verify_figure


def test_value_if_trustworthy_only_returns_verified():
    verified = verify_figure("x", [SourcedValue(10, "a"), SourcedValue(10, "b")])
    single = verify_figure("y", [SourcedValue(10, "a")])
    conflict = verify_figure("z", [SourcedValue(10, "a"), SourcedValue(20, "b")])
    assert value_if_trustworthy(verified) == 10
    assert value_if_trustworthy(single) is None      # single-source is not trustworthy
    assert value_if_trustworthy(conflict) is None     # conflict is not trustworthy
    assert value_if_trustworthy(None) is None


def test_valuation_tiers():
    assert valuation_vs_history(15, 25).verdict == "cheap"       # 60% of history
    assert valuation_vs_history(25, 25).verdict == "fair"
    assert valuation_vs_history(40, 25).verdict == "expensive"
    assert valuation_vs_history(40, 25).concern is True
    assert valuation_vs_history(None, 25).known is False
    assert valuation_vs_history(-5, 25).known is False           # loss-making


def test_earnings_quality():
    assert earnings_quality(90, 100).verdict == "strong"
    assert earnings_quality(40, 100).verdict == "weak"
    assert earnings_quality(40, 100).concern is True
    assert earnings_quality(65, 100).verdict == "mixed"
    assert earnings_quality(90, 0).known is False


def test_earnings_quality_flags_negative_operating_cash_flow_as_a_distinct_red_flag():
    # WHY (CA-level rigor, real money): a company can report a genuine profit while operating
    # cash flow is actually NEGATIVE -- cash left the business even as it claimed to make money
    # (live-verified against real filings: BHEL FY2024 net profit ~2.82B vs OCF ~-37.1B, SAIL
    # FY2023 net profit ~21.8B vs OCF ~-52.9B, VAKRANGEE FY2025 net profit ~65M vs OCF ~-188M).
    # The old code folded this into the same "weak" band as a merely-thin-but-still-positive
    # cash conversion (e.g. ratio 10%), with a "profits are only partly backed by cash" message --
    # a serious understatement of a much more severe pattern (receivables/working-capital stress,
    # or a revenue-recognition red flag), the textbook earnings-quality red flag CAs specifically
    # watch for. It must read as distinctly more serious than "weak", not the same tier.
    r = earnings_quality(-40, 100)
    assert r.concern is True
    assert r.verdict != "weak"
    assert "negative" in r.detail.lower()
    assert "red flag" in r.detail.lower()


def test_leverage_health():
    assert leverage_health(20, 100, 50, 5).verdict == "healthy"   # D/E 0.2, cover 10x
    assert leverage_health(150, 100, 10, 8).verdict == "stretched"  # D/E 1.5
    assert leverage_health(20, 100, 6, 3).verdict == "stretched"    # cover 2x < 3
    assert leverage_health(20, 0, 5, 1).known is False


def test_promoter_pledge():
    assert promoter_pledge(0).verdict == "none"
    assert promoter_pledge(10).verdict == "watch"
    assert promoter_pledge(40).verdict == "high"
    assert promoter_pledge(40).concern is True
    assert promoter_pledge(None).known is False


def test_assemble_verdict_constructive_high_confidence():
    v = assemble_verdict(
        valuation_vs_history(15, 25),  # cheap
        [earnings_quality(90, 100), leverage_health(20, 100, 50, 5), promoter_pledge(0)],
    )
    assert v.valuation == ValuationTier.CHEAP
    assert v.quality == QualityTier.STRONG
    assert v.leaning == Leaning.CONSTRUCTIVE
    assert v.confidence == Confidence.HIGH
    assert v.caveat  # always present
    assert len(v.reasons) == 4


def test_single_clean_quality_signal_is_not_strong():
    # WHY (real money): one verified clean signal, with debt (leverage) and pledge UNKNOWN, is not
    # a "strong" balance sheet. STRONG must require >=2 corroborating verified dimensions, else a
    # cheap P/E + one lucky-verified metric would read FAVORABLE on thin evidence.
    v = assemble_verdict(
        valuation_vs_history(15, 25),                       # cheap
        [earnings_quality(90, 100),                         # known, clean
         leverage_health(None, None, None, None),           # debt UNVERIFIED
         promoter_pledge(None)],                            # UNVERIFIED
    )
    assert v.quality == QualityTier.MIXED                   # not STRONG (was over-confident)
    assert v.leaning != Leaning.CONSTRUCTIVE                # cheap + non-strong -> not constructive


def test_two_clean_quality_signals_reach_strong():
    v = assemble_verdict(
        valuation_vs_history(15, 25),
        [earnings_quality(90, 100), leverage_health(20, 100, 50, 5), promoter_pledge(None)],
    )
    assert v.quality == QualityTier.STRONG                  # 2 known incl. leverage -> corroborated


def test_two_soft_signals_without_verified_debt_are_not_strong():
    # WHY (real money): earnings-quality + no-pledge are two signals, but NEITHER is solvency.
    # STRONG must be unreachable while the critical debt dimension is unverified — otherwise a
    # cheap name reads "strong balance sheet, favorable" with its debt never checked.
    v = assemble_verdict(
        valuation_vs_history(15, 30),                       # cheap
        [earnings_quality(90, 100),                         # known, clean (non-solvency)
         leverage_health(None, None, None, None),           # DEBT UNVERIFIED (critical)
         promoter_pledge(0)],                               # known, clean (non-solvency)
    )
    assert v.quality != QualityTier.STRONG                  # not strong: solvency unverified
    assert v.leaning != Leaning.CONSTRUCTIVE                # so it doesn't drive a FAVORABLE stance


def test_valuation_exposes_discount_magnitude():
    # margin of safety: current P/E vs its own median, as a ratio the ranker can weigh by depth.
    m = valuation_vs_history(15, 25)                       # 0.60x
    assert m.magnitude is not None and abs(m.magnitude - 0.6) < 1e-9


def test_assemble_carries_valuation_ratio_onto_verdict():
    v = assemble_verdict(valuation_vs_history(12, 24),     # 0.50x
                         [earnings_quality(90, 100), leverage_health(20, 100, 50, 5)])
    assert v.valuation_ratio is not None and abs(v.valuation_ratio - 0.5) < 1e-9


def test_confidence_capped_when_debt_unverified():
    # WHY (real money): leverage is the solvency dimension. You cannot be HIGH-confidence about a
    # business whose debt you have not verified, even when everything else checks out.
    v = assemble_verdict(
        valuation_vs_history(15, 25),                       # known (cheap)
        [earnings_quality(90, 100),                         # known
         leverage_health(None, None, None, None),           # DEBT UNVERIFIED (critical)
         promoter_pledge(0)],                               # known
    )
    assert v.confidence == Confidence.MEDIUM                # 3/4 known would be HIGH, but debt caps it


def test_confidence_high_when_debt_verified_and_most_known():
    v = assemble_verdict(
        valuation_vs_history(15, 25),
        [earnings_quality(90, 100), leverage_health(20, 100, 50, 5), promoter_pledge(0)],
    )
    assert v.confidence == Confidence.HIGH                  # leverage known + 4/4 -> uncapped


def test_assemble_verdict_cautious_on_concerns():
    v = assemble_verdict(
        valuation_vs_history(40, 25),  # expensive (concern)
        [earnings_quality(40, 100), leverage_health(150, 100, 10, 8), promoter_pledge(40)],
    )
    assert v.valuation == ValuationTier.EXPENSIVE
    assert v.quality == QualityTier.WEAK  # 3 concerns
    assert v.leaning == Leaning.CAUTIOUS


def test_assemble_verdict_unknown_is_low_confidence():
    v = assemble_verdict(
        valuation_vs_history(None, None),
        [earnings_quality(None, None), leverage_health(None, None, None, None),
         promoter_pledge(None)],
    )
    assert v.valuation == ValuationTier.UNKNOWN
    assert v.quality == QualityTier.UNKNOWN
    assert v.leaning == Leaning.UNKNOWN
    assert v.confidence == Confidence.LOW
    assert v.reasons == ()
