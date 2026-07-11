"""Threshold ordering invariants.

The analysis thresholds -- P/E cheap/expensive, the ROE/ROCE/ROA/margin good-vs-weak bands, the
bank ROA bands, margin-of-safety, the volatility ceilings, and the concentration/pledge warns --
are documented as EXPERT-TUNABLE: a human edits the constants. If a tune inverts a band (sets the
"good" ROA below the "weak" ROA, or the cheap P/E above the expensive one), every classification
built on it silently flips -- misreading real-money verdicts, with no OTHER test necessarily
failing (they exercise the bands indirectly and would fail with a confusing downstream message).
This locks the ordering relationships so a mis-tune fails loudly and specifically, right here.
"""
from src import constants as C
from src.analysis import bank_framework as B
from src.analysis import deep_metrics as D
from src.analysis import framework as F
from src.analysis import sizing as S
from src.analysis import trends as T


def test_valuation_and_ratio_bands_are_correctly_ordered():
    assert F._PE_CHEAP < 1.0 < F._PE_EXPENSIVE                      # below own history cheap, above expensive
    # the always-visible "Price:" insight must classify on the SAME thresholds as the verdict tier
    # (coupled in 569d9b9); a divergence would tell a parent "cheaper than usual" while the verdict says "fair".
    assert (D._PE_CHEAP, D._PE_EXPENSIVE) == (F._PE_CHEAP, F._PE_EXPENSIVE)
    assert D._ROE_GOOD > D._ROE_WEAK
    assert D._ROCE_GOOD > D._ROCE_WEAK
    assert D._ROA_GOOD > D._ROA_WEAK
    assert D._NETMARGIN_GOOD > D._NETMARGIN_WEAK
    assert D._OPMARGIN_GOOD > D._OPMARGIN_WEAK
    assert D._DIVIDEND_YIELD_UNUSUAL > 0


def test_bank_roa_bands_ordered_and_below_the_industrial_floor():
    assert B._ROA_STRONG > B._ROA_WEAK
    # a bank earns ~1% ROA by nature, well below the industrial "good" floor -- the whole reason a
    # lender needs its own bands (a healthy bank must not be graded on the industrial ROA scale).
    assert B._ROA_WEAK < D._ROA_GOOD


def test_margin_of_safety_and_strength_weights_are_consistent():
    assert S._MOS_DEEP < S._MOS_RICH                                # a deeper discount to history = more margin of safety
    assert abs(sum(S._STRENGTH_WEIGHTS) - 1.0) < 1e-9               # normalized so verdict_strength lands in [0,1]


def test_trend_and_cashflow_thresholds_are_correctly_ordered():
    assert T._CUM_OCF_STRONG > T._CUM_OCF_WEAK
    assert T._GROWTH_MIN > 0
    assert T._VOLATILITY_SWING < T._VOLATILITY_SWING_ABSURD         # a real swing must sit below the base-effect ceiling
    assert T._REVENUE_VOLATILITY_SWING < T._VOLATILITY_SWING_ABSURD
    assert T._CAGR_BASE_EFFECT > 0
    assert F._COVERAGE_CANNOT_COVER == 1.0                          # interest cover below 1x = does NOT cover


def test_concentration_and_pledge_thresholds_are_in_range():
    assert 0 < C.CONCENTRATION_SECTOR_WARN < 1
    assert 0 < C.CONCENTRATION_TOP_HOLDING_WARN < 1
    assert 0 < C.PROMOTER_PLEDGE_HIGH_PCT <= 100
