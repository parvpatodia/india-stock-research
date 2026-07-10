from src.analysis.deep_metrics import (
    asset_turnover,
    compute_deep_metrics,
    net_margin,
    operating_margin,
    plain_points,
    return_on_assets,
    return_on_capital,
    return_on_equity,
)
from src.analysis.framework import REAL_ESTATE_LEVERAGE_CAVEAT

CR = 1e7


def test_roe_bands():
    assert return_on_equity(20 * CR, 100 * CR).verdict == "strong"      # 20%
    assert return_on_equity(5 * CR, 100 * CR).verdict == "weak"         # 5%
    assert return_on_equity(10 * CR, 100 * CR).verdict == "moderate"    # 10%
    assert return_on_equity(10 * CR, 100 * CR).concern is False
    assert return_on_equity(5 * CR, 100 * CR).concern is True


def test_ratios_unknown_when_inputs_missing_or_nonpositive():
    assert return_on_equity(None, 100 * CR).known is False
    assert return_on_equity(10 * CR, 0).known is False                  # zero equity guarded
    assert net_margin(10 * CR, None).known is False
    assert asset_turnover(10 * CR, 0).known is False


def test_roce_and_roa_and_margins_values():
    roce = return_on_capital(30 * CR, 100 * CR, 50 * CR)                 # 30/150 = 20%
    assert roce.known and "20%" in roce.detail and roce.verdict == "strong"
    roa = return_on_assets(6 * CR, 100 * CR)                            # 6%
    assert roa.verdict == "strong"
    nm = net_margin(12 * CR, 100 * CR)                                  # 12%
    assert nm.verdict == "strong"
    om = operating_margin(4 * CR, 100 * CR)                             # 4% -> weak
    assert om.verdict == "weak" and om.concern is True


def test_bank_roa_uses_bank_bands_not_industrial():
    # WHY (regression): a healthy bank (~1.2% ROA) must not read "weak" in the plain reasons.
    v = {"net_profit": 12 * CR, "total_assets": 1000 * CR, "equity": 100 * CR}
    ind = {m.name: m for m in compute_deep_metrics(v, is_bank=False)}["Return on assets (ROA)"]
    bank = {m.name: m for m in compute_deep_metrics(v, is_bank=True)}["Return on assets (ROA)"]
    assert ind.verdict == "weak"          # 1.2% < 2% industrial floor
    assert bank.verdict == "strong"       # 1.2% >= 1.0% bank floor
    assert bank.concern is False


def test_compute_deep_metrics_bank_skips_margins():
    v = {"net_profit": 10 * CR, "equity": 100 * CR, "total_assets": 1000 * CR,
         "ebit": 30 * CR, "total_debt": 50 * CR, "revenue": 80 * CR}
    industrial = {m.name for m in compute_deep_metrics(v, is_bank=False)}
    bank = {m.name for m in compute_deep_metrics(v, is_bank=True)}
    assert "Net profit margin" in industrial and "Asset turnover" in industrial
    assert "Net profit margin" not in bank and "Asset turnover" not in bank
    assert "Return on equity (ROE)" in bank                              # banks still get ROE/ROA


def test_plain_points_are_simple_sentences_with_numbers():
    v = {"current_pe": 33.0, "median_pe": 12.0, "operating_cash_flow": 159 * CR,
         "net_profit": 100 * CR, "total_debt": 84 * CR, "equity": 100 * CR,
         "ebit": 30 * CR, "interest_expense": 3 * CR, "total_assets": 200 * CR,
         "revenue": 250 * CR}
    points = plain_points(v, compute_deep_metrics(v, is_bank=False))
    joined = " ".join(points)
    assert len(points) >= 5                                             # 5-6+ reasons
    assert "P/E 33" in joined and "pricier than usual" in joined       # price point, plain
    assert "for every ₹1 of reported profit" in joined                 # cash-quality point
    assert "D/E 0.84" in joined                                        # debt point


def test_plain_points_omit_unknowns():
    v = {"current_pe": None, "median_pe": None, "net_profit": None}    # nothing cross-verified
    assert plain_points(v, compute_deep_metrics(v)) == []


def test_plain_points_dividend_zero_is_not_a_red_flag():
    points = plain_points({"dividend_yield_pct": 0.0}, [])
    joined = " ".join(points)
    assert "no dividend" in joined
    assert "not automatically a red flag" in joined


def test_plain_points_dividend_bands_stay_neutral():
    modest = " ".join(plain_points({"dividend_yield_pct": 0.5}, []))
    moderate = " ".join(plain_points({"dividend_yield_pct": 1.6}, []))
    high = " ".join(plain_points({"dividend_yield_pct": 5.1}, []))
    assert "modest" in modest and "0.5%" in modest
    assert "moderate" in moderate and "1.6%" in moderate
    assert "high" in high and "5.1%" in high
    # WHY (real money): dividend yield is context-dependent, must never claim a direction is
    # automatically good or bad -- the same neutral framing applies at every band.
    for joined in (modest, moderate, high):
        assert "automatically good or bad" in joined


def test_plain_points_no_dividend_point_when_unknown():
    assert plain_points({}, []) == []


def test_plain_points_real_estate_debt_carries_sector_caveat_in_the_always_visible_summary():
    # WHY (real money, UI honesty): an adversarial review of the real-estate leverage caveat
    # (added last iteration) found it only reached verdict.reasons, shown inside the collapsed
    # "See the evidence" expander -- the ALWAYS-VISIBLE "Why, in plain terms" summary (this
    # function's output, report.insights) kept saying "high, worth watching" for a real developer
    # at D/E 1.09 (Prestige, live-verified) with zero sector context, so a parent could read the
    # un-caveated alarm and never open the expander that explains it's sector-normal.
    v = {"total_debt": 109 * CR, "equity": 100 * CR}   # D/E 1.09 -> "high, worth watching"
    points = plain_points(v, [], is_real_estate=True)
    joined = " ".join(points)
    assert "high, worth watching" in joined
    assert REAL_ESTATE_LEVERAGE_CAVEAT in joined


def test_plain_points_non_real_estate_debt_has_no_sector_caveat():
    v = {"total_debt": 109 * CR, "equity": 100 * CR}
    joined = " ".join(plain_points(v, [], is_real_estate=False))
    assert "high, worth watching" in joined
    assert REAL_ESTATE_LEVERAGE_CAVEAT not in joined


def test_plain_points_real_estate_moderate_debt_has_no_caveat_clutter():
    # WHY: only attach the caveat when the un-caveated wording could actually alarm a reader
    # ("high, worth watching"); a "moderate" read (e.g. Brigade's live D/E 0.93) is not itself
    # presented as a concern, so adding sector commentary there would be clutter, not honesty.
    v = {"total_debt": 93 * CR, "equity": 100 * CR}    # D/E 0.93 -> "moderate"
    joined = " ".join(plain_points(v, [], is_real_estate=True))
    assert "moderate" in joined
    assert REAL_ESTATE_LEVERAGE_CAVEAT not in joined


def test_plain_points_debt_word_matches_leverage_health_on_weak_coverage_alone():
    # WHY (real money, honesty; adversarial-review regression): D/E 0.60 alone reads "moderate",
    # but weak interest coverage (2.0x, below the 3.0x minimum) makes leverage_health -- the SAME
    # computation the Verdict's tier/concern flag is built from -- read this "stretched" overall.
    # Before this fix, plain_points recomputed its own D/E-only word, so the ALWAYS-VISIBLE "Why,
    # in plain terms" summary said "moderate" (no caveat) for the exact company the collapsed
    # evidence panel and PDF called "stretched" with a concern flag -- the same company read two
    # different ways on two supposedly-mirrored surfaces, purely because coverage (not D/E) was
    # the actual stretched signal.
    v = {"total_debt": 60 * CR, "equity": 100 * CR, "ebit": 200 * CR, "interest_expense": 100 * CR}
    joined = " ".join(plain_points(v, [], is_real_estate=True))
    assert "high, worth watching" in joined
    assert "moderate" not in joined
    assert REAL_ESTATE_LEVERAGE_CAVEAT in joined
