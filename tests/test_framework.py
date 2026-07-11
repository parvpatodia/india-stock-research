from src.analysis.framework import (
    assemble_verdict,
    earnings_quality,
    leverage_health,
    promoter_pledge,
    valuation_vs_history,
    value_if_trustworthy,
)
from src.analysis.sizing import Stance, stance_from_verdict
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
    assert r.severe is True                    # standalone-disqualifying, not a mild concern


def test_a_lone_negative_ocf_red_flag_reads_weak_not_mixed_so_it_is_never_suggested():
    # WHY (real money, CA-level rigor): negative operating cash flow despite a REPORTED profit is
    # the textbook quality-of-earnings red flag (BHEL FY2024, SAIL FY2023, VAKRANGEE). earnings_quality
    # already labels it a "serious red flag", but assemble_verdict collapsed every concern to a plain
    # count, so a LONE such red flag reached only MIXED quality -> NEUTRAL leaning -> a NEUTRAL stance
    # -- and rank_picks/suggest_allocation both treat NEUTRAL as ELIGIBLE. A cash-burning-but-
    # "profitable" name could therefore enter the daily research shortlist and receive an allocation,
    # even with a cheap price and clean debt. A SEVERE concern must stand on its own: WEAK quality ->
    # CAUTIOUS leaning -> UNFAVORABLE stance, excluded from every suggestion.
    v = assemble_verdict(
        valuation_vs_history(15, 25),                       # cheap (would otherwise pull constructive)
        [earnings_quality(-40, 100),                        # NEGATIVE OCF despite profit -> severe
         leverage_health(20, 100, 50, 5),                   # healthy, clean (non-concern)
         promoter_pledge(0)],                               # none
    )
    assert v.quality == QualityTier.WEAK                    # severe red flag ALONE -> WEAK, not MIXED
    assert v.leaning == Leaning.CAUTIOUS                    # leans unfavorable despite the cheap price
    assert stance_from_verdict(v) == Stance.UNFAVORABLE     # -> excluded from shortlist + allocation


def test_a_single_ordinary_concern_still_reads_mixed_not_weak():
    # Guard the boundary: a NON-severe lone concern (merely-thin cash conversion) must stay MIXED --
    # the severe path must not sweep every single concern into WEAK.
    v = assemble_verdict(
        valuation_vs_history(15, 25),                       # cheap
        [earnings_quality(40, 100),                         # "weak" (thin, but positive OCF) -> ordinary
         leverage_health(20, 100, 50, 5),                   # healthy
         promoter_pledge(0)],                               # none
    )
    assert v.quality == QualityTier.MIXED                   # one ordinary concern -> still MIXED


def test_leverage_health():
    assert leverage_health(20, 100, 50, 5).verdict == "healthy"   # D/E 0.2, cover 10x
    assert leverage_health(150, 100, 10, 8).verdict == "stretched"  # D/E 1.5
    assert leverage_health(20, 100, 6, 3).verdict == "stretched"    # cover 2x < 3
    assert leverage_health(20, 0, 5, 1).known is False


def test_leverage_health_negative_operating_profit_avoids_a_confusing_negative_cover():
    # WHY (real money, clarity): a leveraged company with an OPERATING LOSS can't service interest
    # from operations -- it must read stretched (it does), but "interest cover -2.0x" is a
    # nonsensical display (you don't "cover" interest with a negative operating profit). State
    # plainly that operating profit is negative and isn't covering interest. The VERDICT is
    # unchanged: the old negative-coverage-below-3 path already read this stretched.
    r = leverage_health(120, 100, -40, 20)     # D/E 1.2, EBIT -40, interest 20
    assert r.verdict == "stretched" and r.concern is True
    assert "-2.0x" not in r.detail and "interest cover -" not in r.detail
    assert "operating profit is negative" in r.detail and "isn't covering" in r.detail


def test_leverage_health_debt_light_operating_loss_is_not_flagged_on_coverage():
    # A near-debt-free company with an operating loss but NO interest bill must not be dragged to
    # "stretched" by the coverage lens -- there is no interest to miss.
    assert leverage_health(1, 100, -40, 0).verdict == "healthy"    # D/E 0.01, no interest


def test_interest_cover_below_one_is_a_severe_solvency_concern():
    # WHY (real money; mirrors the negative-OCF and negative-ROA severe fixes): interest cover BELOW
    # 1x means operating profit does NOT cover the interest bill -- the company services its debt from
    # reserves/asset sales/fresh borrowing, a serious solvency-distress signal, not merely "stretched".
    # As an ordinary lone concern it only reached MIXED quality -> a suggestible NEUTRAL.
    r = leverage_health(80, 100, 80, 100)             # D/E 0.8, EBIT 80 < interest 100 -> cover 0.8x
    assert r.verdict == "stretched" and r.concern is True
    assert r.severe is True
    assert "1x" in r.detail.lower() and "cover" in r.detail.lower()


def test_interest_cover_of_two_covers_tightly_and_is_not_severe():
    # The boundary: coverage of 1-3x covers the interest bill (tightly) -- an ordinary "stretched"
    # concern, NOT the severe can't-cover case, so it isn't swept into WEAK.
    r = leverage_health(20, 100, 6, 3)                # cover 2.0x -- covers, but under the 3x comfort line
    assert r.verdict == "stretched" and r.concern is True
    assert r.severe is False


def test_a_company_that_cannot_cover_interest_reads_weak_not_a_suggestible_neutral():
    # A lone can't-cover-interest concern drags quality to WEAK -> CAUTIOUS -> UNFAVORABLE, so a zombie
    # servicing debt it can't afford from operations is never surfaced on a cheap valuation.
    v = assemble_verdict(
        valuation_vs_history(15, 25),                       # cheap
        [earnings_quality(90, 100),                         # clean (positive OCF, strong)
         leverage_health(80, 100, 80, 100),                 # cover 0.8x -> cannot cover -> severe
         promoter_pledge(0)],
    )
    assert v.quality == QualityTier.WEAK
    assert v.leaning == Leaning.CAUTIOUS
    assert stance_from_verdict(v) == Stance.UNFAVORABLE


def test_promoter_pledge():
    assert promoter_pledge(0).verdict == "none"
    assert promoter_pledge(10).verdict == "watch"
    assert promoter_pledge(40).verdict == "high"
    assert promoter_pledge(40).concern is True
    assert promoter_pledge(None).known is False


def test_pledge_threshold_is_the_single_shared_constant():
    # WHY (one source of truth, real money): the "serious pledge" threshold was duplicated as
    # literals in framework and screener_source. Both derive from the constant now -- but they must
    # also AGREE at the boundary. The framework used `>` (watch AT the threshold) while the Screener
    # signal a parent reads used `>=` (serious AT the threshold), so at EXACTLY the threshold the
    # verdict metric and the wording CONTRADICTED each other on a top-tier red flag. Both treat the
    # threshold itself as the SERIOUS tier now (the conservative direction, matching what the parent
    # already sees from the Screener side).
    from src.analysis import framework
    from src.constants import PROMOTER_PLEDGE_HIGH_PCT
    from src.data.screener_source import promoter_pledge_point
    assert framework._PLEDGE_HIGH is PROMOTER_PLEDGE_HIGH_PCT
    assert promoter_pledge(PROMOTER_PLEDGE_HIGH_PCT + 1).verdict == "high"
    assert promoter_pledge(PROMOTER_PLEDGE_HIGH_PCT - 1).verdict == "watch"
    # at EXACTLY the shared threshold, the framework metric and the Screener wording must agree:
    assert promoter_pledge(PROMOTER_PLEDGE_HIGH_PCT).verdict == "high"
    assert promoter_pledge(PROMOTER_PLEDGE_HIGH_PCT).concern is True
    assert "serious red flag" in promoter_pledge_point(PROMOTER_PLEDGE_HIGH_PCT)


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


def test_all_concern_free_but_lukewarm_signals_are_not_strong():
    # WHY (real money): two concern-free dimensions that are merely MODERATE, with nothing
    # affirmatively strong, must read MIXED, not STRONG. STRONG means at least one dimension is
    # genuinely strong, not just "no red flags present". Here: moderate leverage (D/E 0.7, no
    # coverage signal), no pledge, earnings unverified -- the critical (leverage) dimension IS
    # known, so the old "N concern-free dimensions" rule wrongly reached STRONG on lukewarm data.
    v = assemble_verdict(
        valuation_vs_history(15, 25),
        [earnings_quality(None, None),            # unverified
         leverage_health(70, 100, None, None),    # D/E 0.7 -> "moderate" (concern-free, not strong)
         promoter_pledge(0)],                     # "none" (concern-free, not a strength)
    )
    assert v.quality == QualityTier.MIXED


def test_one_strong_dimension_with_a_moderate_one_still_strong():
    # Guard against over-tightening: a genuinely STRONG dimension (earnings quality) plus a
    # concern-free MODERATE leverage (the critical dimension, known) is still STRONG. The fix only
    # blocks verdicts with nothing affirmatively strong; it does not demand every dimension be strong.
    v = assemble_verdict(
        valuation_vs_history(15, 25),
        [earnings_quality(90, 100),               # "strong" (affirmative)
         leverage_health(70, 100, None, None),    # "moderate" (concern-free)
         promoter_pledge(0)],
    )
    assert v.quality == QualityTier.STRONG


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
