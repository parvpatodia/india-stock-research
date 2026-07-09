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
