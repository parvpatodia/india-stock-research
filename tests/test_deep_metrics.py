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


def test_loss_making_ratios_say_loses_not_keeps_or_earns():
    # WHY (real money, clarity for a non-expert): a loss-making company has NEGATIVE margins/returns.
    # "It keeps about Rs.-50 of profit" / "it earns about Rs.-50 a year" is a confusing double
    # negative a parent has to decode -- state plainly that the business LOSES money. The numbers
    # and the "weak" tag are unchanged; only the wording of the negative case is fixed.
    nm = net_margin(-500 * CR, 1000 * CR).detail          # -50%
    assert "loses about ₹50" in nm and "keeps about" not in nm and "-50%" in nm
    om = operating_margin(-300 * CR, 1000 * CR).detail    # -30%
    assert "loses about ₹30 at the operating level" in om and "is left as operating profit" not in om
    roe = return_on_equity(-500 * CR, 1000 * CR).detail   # -50%
    assert "loses about ₹50 a year" in roe and "earns about" not in roe
    roa = return_on_assets(-60 * CR, 1000 * CR).detail    # -6.0%
    assert "loses about ₹6.0" in roa and "earns about" not in roa
    roce = return_on_capital(-30 * CR, 100 * CR, 50 * CR).detail   # EBIT loss -> -20% on ₹150 capital
    assert "loses about ₹20 a year" in roce and "earns about" not in roce


def test_margins_over_100pct_flag_profit_exceeding_sales_rather_than_reading_strong():
    # WHY (quality of earnings, honesty; cross-verified counterpart of the other-income>100% fix):
    # when net profit (or the operating measure) EXCEEDS total sales the margin passes 100%, and
    # "net margin 200% -- strong" misrepresents it: profit that large can't come from the core sales
    # business, it's driven by other/one-off income. Reachable (a holding company whose income sits
    # below 'sales', or a big one-off divestment gain). Read it plainly, never "strong".
    nm = net_margin(200 * CR, 100 * CR)                   # net profit 2x sales
    assert nm.verdict != "strong"
    assert "larger than" in nm.detail.lower() and "sales" in nm.detail.lower()
    assert "200%" in nm.detail                            # magnitude still shown for reference
    om = operating_margin(150 * CR, 100 * CR)             # operating measure 1.5x sales
    assert om.verdict != "strong"
    assert "exceeded" in om.detail.lower()


def test_profitable_ratios_still_say_keeps_and_earns():
    # Guard: the positive-value wording (and the numbers) are untouched.
    assert "keeps about ₹8 of final profit" in net_margin(80 * CR, 1000 * CR).detail
    assert "is left as operating profit" in operating_margin(150 * CR, 1000 * CR).detail
    assert "earns about ₹20 a year" in return_on_equity(20 * CR, 100 * CR).detail
    assert "earns about ₹6.0" in return_on_assets(6 * CR, 100 * CR).detail


def test_return_ratios_use_average_denominator_when_the_prior_year_is_available():
    # WHY (CA-level rigor): profit is EARNED OVER the year, so the textbook denominator for a
    # return ratio is the AVERAGE of opening and closing capital, not the closing snapshot. Using
    # closing understates the ratio for a company that grew equity/assets during the year (retained
    # earnings, a capital raise, or a merger -- e.g. a post-merger bank whose equity jumped). When
    # the prior year is cross-verified, average it in; otherwise fall back safely to the point value.
    roe = return_on_equity(20 * CR, 100 * CR, prior_equity=80 * CR)     # avg (100+80)/2 = 90
    assert abs(20.0 / 90.0 * 100 - float(roe.detail.split("ROE ")[1].split("%")[0])) < 0.6
    assert "average" in roe.detail.lower()                              # basis disclosed
    roce = return_on_capital(30 * CR, 100 * CR, 50 * CR,
                             prior_equity=80 * CR, prior_total_debt=40 * CR)  # avg CE (150+120)/2=135
    assert "22%" in roce.detail                                          # 30/135 = 22%, not 20%
    roa = return_on_assets(6 * CR, 100 * CR, prior_total_assets=80 * CR)  # avg (100+80)/2 = 90
    assert "6.7" in roa.detail                                           # 6/90 = 6.7%, not 6.0%


def test_asset_turnover_uses_average_assets_like_roa():
    # WHY (CA-level rigor, consistency): asset turnover = sales / total assets is the same
    # flow-over-stock ratio as ROA -- sales are generated OVER the year, so the denominator should
    # be average (opening+closing) total assets, not the closing snapshot. It was left on point
    # assets when ROA moved to average; align it (a company that grew assets during the year would
    # otherwise show an artificially low turnover).
    at = asset_turnover(200 * CR, 100 * CR, prior_total_assets=80 * CR)   # avg (100+80)/2 = 90
    assert "2.22x" in at.detail and "average assets" in at.detail        # 200/90 = 2.22x, not 2.00
    assert "2.00x" in asset_turnover(200 * CR, 100 * CR).detail          # no prior -> point
    assert "average" not in asset_turnover(200 * CR, 100 * CR).detail.lower()


def test_return_ratios_fall_back_to_point_value_when_prior_is_missing_or_nonpositive():
    # a missing or non-positive prior must NOT corrupt the average -- use the closing value alone,
    # preserving the pre-existing behavior exactly.
    assert "20%" in return_on_equity(20 * CR, 100 * CR).detail                       # no prior
    assert "20%" in return_on_equity(20 * CR, 100 * CR, prior_equity=0).detail       # bad prior
    assert "20%" in return_on_equity(20 * CR, 100 * CR, prior_equity=-5 * CR).detail
    assert "average" not in return_on_equity(20 * CR, 100 * CR).detail.lower()


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


def test_plain_points_skips_industrial_debt_and_cash_lines_for_a_bank():
    # WHY (real money, sector-aware; regression exposed once bank balance sheets started parsing):
    # the industrial D/E and OCF-vs-profit lines do NOT apply to a lender -- a bank is leveraged by
    # design (routed to the ROA lens, not D/E) and its operating cash flow is dominated by
    # lending/deposit flows, so a growing bank can show negative OCF that would FALSE-flag as a
    # cash red flag. The verdict path and compute_deep_metrics already avoid these lenses for
    # banks; the always-visible plain_points summary must too, or it contradicts them.
    v = {"current_pe": 18.0, "median_pe": 16.0, "operating_cash_flow": -50 * CR,
         "net_profit": 200 * CR, "total_debt": 970 * CR, "equity": 200 * CR,
         "ebit": 100 * CR, "interest_expense": 50 * CR, "total_assets": 3000 * CR,
         "dividend_yield_pct": 1.2}
    joined = " ".join(plain_points(v, compute_deep_metrics(v, is_bank=True), is_bank=True))
    assert "D/E" not in joined                       # no industrial leverage line for a lender
    assert "backed by cash" not in joined            # no industrial cash-quality line
    assert "red flag" not in joined.lower()          # the negative-OCF false alarm must not appear
    assert "P/E 18" in joined                         # price still shown
    assert "dividend yield" in joined.lower()         # dividend still shown


def test_plain_points_still_shows_debt_and_cash_lines_for_a_non_bank():
    # regression: the industrial lines must still appear for a normal (non-bank) company
    v = {"operating_cash_flow": 90 * CR, "net_profit": 100 * CR,
         "total_debt": 84 * CR, "equity": 100 * CR, "ebit": 30 * CR, "interest_expense": 3 * CR}
    joined = " ".join(plain_points(v, [], is_bank=False))
    assert "D/E 0.84" in joined and "Cash quality:" in joined


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


def test_plain_points_unusually_high_dividend_yield_warns_about_price_fall_and_sustainability():
    # WHY (real money, honesty for income-seeking parents): a 3% and a 12% yield are NOT the same
    # story. Yield = dividend / price, so an UNUSUALLY high yield often reflects a FALLEN share price
    # as much as a generous dividend, and can flag a payout the market expects to be cut -- the
    # classic yield trap that draws income investors into a value trap. A normal 'high' yield (3-6%)
    # keeps the neutral income-vs-growth framing; an unusually high one ADDS a check-sustainability
    # caveat (non-alarmist: it says "check", and stays true even for a legitimate high-yielder whose
    # payout IS covered).
    high12 = " ".join(plain_points({"dividend_yield_pct": 12.0}, []))
    assert "12.0%" in high12
    assert "fallen share price" in high12.lower()
    assert "cut" in high12.lower() and "covered by earnings" in high12.lower()
    assert "automatically good or bad" in high12          # base neutral framing still present
    # a normal 'high' yield (5.1%) must NOT get the extra distress caveat:
    normal_high = " ".join(plain_points({"dividend_yield_pct": 5.1}, []))
    assert "fallen share price" not in normal_high.lower()


def test_plain_points_price_reads_as_valuation_not_a_share_price():
    # WHY (real money, clarity for a non-expert): the median P/E is a MULTIPLE, not the share price.
    # The old wording ("historically it traded near ₹24", "cheaper than its usual price") read the
    # P/E LEVEL as a rupee share price -- jarring and confusing next to the real ₹1,400 price a parent
    # sees for the same holding elsewhere in the app. Keep the intuitive "₹ per ₹1 of profit"
    # explanation of P/E, but frame the history and the tag as the VALUATION LEVEL versus its own
    # history, never a "price".
    price = next(p for p in plain_points({"current_pe": 18.0, "median_pe": 24.0}, [])
                 if p.startswith("Price:"))
    assert "P/E 18" in price and "₹18" in price and "₹24" in price   # the figures are still shown
    assert "traded near" not in price                                # no share-price implication
    assert "usual price" not in price and "normal price" not in price
    assert "history" in price.lower()                                # framed vs its own history
    assert "cheaper" in price                                        # 18/24 = 0.75 < 0.8 -> cheaper


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


def test_plain_points_cash_quality_word_matches_earnings_quality_on_negative_ocf():
    # WHY (real money, honesty; same class of bug as the debt-word regression below): this
    # function used to recompute its own cash-quality ratio/word independently instead of calling
    # earnings_quality() (the SAME computation the Verdict's tier/concern flag is built from) --
    # so a negative operating cash flow (net_profit 100, ocf -40, live-verified pattern: BHEL
    # FY2024, SAIL FY2023, VAKRANGEE FY2025/FY2023) fell into the same "only partly backed by
    # cash (watch this)" wording as a merely-thin-but-positive ratio, instead of the distinctly
    # more serious "red flag" earnings_quality() itself now reports for this exact pattern. The
    # ALWAYS-VISIBLE "Why, in plain terms" summary must not soften what the evidence panel calls
    # a red flag.
    v = {"net_profit": 100 * CR, "operating_cash_flow": -40 * CR}
    joined = " ".join(plain_points(v, [])).lower()
    assert "red flag" in joined
    assert "only partly backed by cash" not in joined


def test_plain_points_negative_cash_flow_reads_as_flowing_out_not_collected():
    # WHY (real money, clarity for a non-expert): when operating cash flow is NEGATIVE, the line
    # read "it actually collected a net outflow of ₹0.50 of cash" -- you don't "collect" an
    # outflow, a self-contradictory phrasing for the exact red-flag pattern this line exists to
    # warn about. Say plainly that cash flowed OUT of the business.
    joined = " ".join(plain_points({"net_profit": 100 * CR, "operating_cash_flow": -50 * CR}, []))
    assert "collected a net outflow" not in joined
    assert "flowed out of the business" in joined.lower() and "₹0.50" in joined
    # a POSITIVE cash flow still reads "collected ₹X of cash" (unchanged):
    pos = " ".join(plain_points({"net_profit": 100 * CR, "operating_cash_flow": 90 * CR}, []))
    assert "collected ₹0.90 of cash" in pos


def test_plain_points_debt_line_flags_an_operating_loss_not_a_negative_cover():
    # WHY (real money, clarity): a leveraged loss-maker showed "operating profit covers its interest
    # bill about -2x over" -- a confusing negative "cover". Say its operating profit is negative and
    # it isn't covering interest instead; the verdict word ("high, worth watching") is unchanged.
    v = {"total_debt": 120 * CR, "equity": 100 * CR, "ebit": -40 * CR, "interest_expense": 20 * CR}
    joined = " ".join(plain_points(v, []))
    assert "-2x" not in joined and "covers its interest bill about" not in joined
    assert "operating profit is negative" in joined and "isn't covering" in joined
    assert "high, worth watching" in joined      # verdict word unchanged


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
