from src.analysis.trends import (
    cagr,
    cash_conversion_quality_point,
    earnings_volatility_point,
    leverage_trend_point,
    limited_history_note,
    margins_improving,
    revenue_volatility_point,
    trend_improving,
    trend_points,
    verified_series,
)

CR = 1e7


def test_cash_conversion_quality_flags_a_persistent_profit_to_cash_gap():
    # WHY (CA-level rigor, quality of earnings): a single year's cash conversion can be lumpy
    # (working-capital timing). The multi-year cumulative view is what a professional relies on --
    # if a company keeps reporting profit but cumulative operating cash flow chronically lags, it
    # signals aggressive revenue recognition or a receivables/working-capital build-up. Built from
    # cross-verified OCF and profit series, so it's a cross-verified insight.
    ocf = {2022: 40 * CR, 2023: 45 * CR, 2024: 50 * CR}       # cum 135
    profit = {2022: 100 * CR, 2023: 110 * CR, 2024: 120 * CR}  # cum 330 -> 41%
    p = cash_conversion_quality_point(ocf, profit)
    assert p is not None
    assert "41%" in p
    assert "3 years" in p
    assert "cross-verified" in p


def test_cash_conversion_quality_confirms_cash_backed_profits():
    ocf = {2022: 95 * CR, 2023: 100 * CR, 2024: 110 * CR}      # cum 305
    profit = {2022: 100 * CR, 2023: 105 * CR, 2024: 115 * CR}  # cum 320 -> 95%
    p = cash_conversion_quality_point(ocf, profit)
    assert "95%" in p and "well backed" in p.lower()


def test_cash_conversion_quality_flags_cumulative_negative_ocf():
    # Reported profits over the years, but the business consumed cash on a cumulative basis -- the
    # strongest multi-year quality-of-earnings red flag.
    ocf = {2022: -40 * CR, 2023: -30 * CR, 2024: 10 * CR}      # cum -60
    profit = {2022: 50 * CR, 2023: 60 * CR, 2024: 70 * CR}     # cum 180
    p = cash_conversion_quality_point(ocf, profit)
    assert "negative" in p.lower() and "red flag" in p.lower()


def test_cash_conversion_quality_needs_three_profitable_years():
    ocf = {2023: 40 * CR, 2024: 50 * CR}
    profit = {2023: 100 * CR, 2024: 110 * CR}
    assert cash_conversion_quality_point(ocf, profit) is None            # only 2 common years
    loss = {2022: -100 * CR, 2023: -90 * CR, 2024: -80 * CR}             # loss-making period
    assert cash_conversion_quality_point({2022: 5 * CR, 2023: 5 * CR, 2024: 5 * CR}, loss) is None


def test_cash_conversion_quality_abstains_when_any_year_is_loss_making():
    # WHY (real money, HIGH severity; found by adversarial review): the cumulative OCF-to-profit
    # ratio is only meaningful over a CONSISTENTLY profitable period. A single loss year netting
    # against profit years can shrink cumulative profit to a near-zero residual, blowing the ratio
    # up into a nonsensical, falsely reassuring "2700% -- well backed by real cash" verdict for a
    # company whose earnings quality is actually murky (a real risk for cyclicals: steel,
    # commodities). Abstain when any year in the window is a loss; the single-year earnings-quality
    # point and the earnings-volatility caveat still cover such a period.
    ocf = {2022: 45 * CR, 2023: 45 * CR, 2024: 45 * CR}         # cum 135
    profit = {2022: -100 * CR, 2023: 50 * CR, 2024: 55 * CR}    # cum +5 -> near-zero denominator
    assert cash_conversion_quality_point(ocf, profit) is None


def test_leverage_trend_flags_a_rising_debt_to_equity_ratio():
    # WHY (real money, CA-level rigor): a single-year D/E snapshot hides whether the balance sheet
    # is getting RISKIER over time. Debt rising faster than equity (D/E climbing) is a core
    # leverage-risk signal a professional watches. Built from cross-verified debt & equity series,
    # so it's a cross-verified insight, not single-source context.
    debt = {2022: 20 * CR, 2023: 40 * CR, 2024: 60 * CR}
    equity = {2022: 100 * CR, 2023: 100 * CR, 2024: 100 * CR}
    p = leverage_trend_point(debt, equity)
    assert p is not None
    assert "risen" in p.lower()
    assert "0.20" in p and "0.60" in p          # shows both endpoints
    assert "FY2022" in p and "FY2024" in p


def test_leverage_trend_flags_deleveraging_as_a_positive():
    # Suzlon-style turnaround: D/E collapses as debt is repaid -- a genuine positive signal.
    debt = {2023: 176 * CR, 2024: 4 * CR, 2026: 6 * CR}
    equity = {2023: 100 * CR, 2024: 100 * CR, 2026: 100 * CR}
    p = leverage_trend_point(debt, equity)
    assert "fallen" in p.lower() or "deleverag" in p.lower()
    assert "1.76" in p and "0.06" in p


def test_leverage_trend_steady_when_change_is_immaterial():
    # A small absolute move on an already-low D/E is noise, not a trend -- don't alarm on it.
    debt = {2023: 4 * CR, 2024: 3 * CR, 2026: 10 * CR}   # DMART-like: 0.04 -> 0.10, tiny absolute
    equity = {2023: 100 * CR, 2024: 100 * CR, 2026: 100 * CR}
    p = leverage_trend_point(debt, equity)
    assert p is not None and "steady" in p.lower()


def test_leverage_trend_surfaces_a_hidden_mid_period_spike():
    # WHY (found by adversarial review): oldest-vs-latest alone hides a leverage episode that
    # resolved -- D/E 0.20 -> 1.50 -> 0.25 reads "roughly steady" on the endpoints, erasing a real
    # (if temporary) spike a CA would want to know about. Surface the intra-period peak.
    debt = {2022: 20 * CR, 2023: 150 * CR, 2024: 25 * CR}
    equity = {2022: 100 * CR, 2023: 100 * CR, 2024: 100 * CR}
    p = leverage_trend_point(debt, equity)
    assert "steady" in p.lower()                 # primary read is still the net endpoint position
    assert "1.50" in p and "FY2023" in p         # but the hidden spike is surfaced
    assert "spiked" in p.lower()


def test_leverage_trend_no_spike_note_when_move_is_monotonic():
    # a clean monotonic rise has its peak AT the last year -> no misleading "spiked in between"
    debt = {2022: 20 * CR, 2023: 40 * CR, 2024: 60 * CR}
    equity = {2022: 100 * CR, 2023: 100 * CR, 2024: 100 * CR}
    assert "spiked" not in leverage_trend_point(debt, equity).lower()


def test_leverage_trend_needs_two_years_with_positive_equity():
    assert leverage_trend_point({2024: 40 * CR}, {2024: 100 * CR}) is None      # one year only
    # negative-equity years are skipped (D/E is meaningless there); too few left -> None
    assert leverage_trend_point({2023: 40 * CR, 2024: 50 * CR},
                                {2023: -10 * CR, 2024: 100 * CR}) is None


def test_verified_series_keeps_only_agreeing_years():
    per_source = {
        "yfinance": {2022: 100, 2023: 110, 2024: 121},
        "screener": {2022: 100.5, 2023: 110, 2024: 200},   # 2024 disagrees
    }
    vs = verified_series(per_source)
    assert set(vs) == {2022, 2023}          # 2024 dropped (sources disagree > 2%)
    assert vs[2023] == 110


def test_verified_series_needs_two_sources_for_a_year():
    per_source = {"yfinance": {2022: 100, 2023: 110}, "screener": {2022: 100}}
    vs = verified_series(per_source)
    assert set(vs) == {2022}                # 2023 only from one source -> dropped


def test_verified_series_chained_agreement_does_not_verify_a_pair_that_disagrees():
    # WHY (real money, HIGH severity; same class of bug as verification.py's verify_figure,
    # found alongside it): a real, live 3-source scenario (app.py wires yfinance + Screener + the
    # annual report for figures_by_year too). A=100 (yfinance), B=101.9 (screener), C=103.8
    # (annual_report): A-B agree (1.9 <= 2.04) and B-C agree (1.9 <= 2.08) at the default 2%
    # tolerance, but A and C are themselves 3.8% apart -- genuinely beyond tolerance -- and were
    # never checked against each other before this fix, so this year would wrongly read as "all
    # 3 sources agree" with a chained median (101.9) instead of the genuine 2-source clique.
    a_c_gap_pct = abs(100.0 - 103.8) / 103.8
    assert a_c_gap_pct > 0.02
    per_source = {
        "yfinance": {2024: 100.0},
        "screener": {2024: 101.9},
        "annual_report": {2024: 103.8},
    }
    vs = verified_series(per_source)
    assert vs.get(2024) != 101.9


def test_cagr_basic_and_guards():
    rate, span = cagr({2020: 100, 2021: 110, 2022: 121})         # 100 -> 121 over 2 yrs = 10%/yr
    assert abs(rate - 10.0) < 1e-9 and span == 2
    assert cagr({2021: 100, 2022: 110}) is None                  # <3 years
    assert cagr({2020: -5, 2021: 10, 2022: 20}) is None          # non-positive endpoint


def test_trend_points_growth_and_margin_direction():
    rev = {2020: 100 * CR, 2021: 110 * CR, 2022: 121 * CR}       # ~10%/yr
    prof = {2020: 10 * CR, 2021: 12 * CR, 2022: 15 * CR}         # faster than sales
    pts = trend_points(rev, prof)
    joined = " ".join(pts)
    assert "sales have been growing" in joined
    assert "profit has been growing" in joined
    assert "margins have been improving" in joined
    assert "outpaced sales" in joined                           # direction-agnostic margin wording


def test_trend_points_shrinking_shows_positive_magnitude_not_a_double_negative():
    # WHY (real money, clarity): a declining company's CAGR is negative, so "shrinking about -13%
    # a year" is a confusing double negative -- the word "shrinking" already carries the direction.
    # Show the magnitude (13%), never "-13%".
    rev = {2021: 200 * CR, 2022: 170 * CR, 2023: 150 * CR}       # ~-13%/yr
    prof = {2021: 40 * CR, 2022: 30 * CR, 2023: 22 * CR}          # ~-26%/yr
    joined = " ".join(trend_points(rev, prof))
    assert "shrinking about -" not in joined
    assert "shrinking about 13%" in joined and "shrinking about 26%" in joined


def test_margin_direction_wording_is_not_grown_when_the_company_is_declining():
    # WHY (real money, clarity/correctness): when BOTH sales and profit are FALLING, "Profit has
    # grown slower than sales" is wrong -- profit didn't grow, it shrank faster. Use direction-
    # agnostic wording ("lagged sales") so the sentence is true whether the business grew or shrank.
    rev = {2021: 200 * CR, 2022: 180 * CR, 2023: 160 * CR}       # ~-11%/yr
    prof = {2021: 40 * CR, 2022: 30 * CR, 2023: 20 * CR}          # ~-29%/yr, margin compressed
    joined = " ".join(trend_points(rev, prof))
    assert "margins have been under pressure" in joined
    assert "has grown slower" not in joined
    assert "lagged sales" in joined


def test_trend_points_margin_wording_when_both_lines_shrink_is_not_a_positive_headline():
    # WHY (real money, clarity -- pairs with trend_improving's melting-ice-cube guard, 1b9dd68):
    # when sales AND profit are BOTH shrinking, margins_improving is True (profit fell slower), and
    # the prose printed "Profit has outpaced sales, so margins have been improving." -- a positive-
    # sounding headline directly under two "...has been shrinking..." lines. That reads as a
    # contradiction (a shrinking profit did not "outpace" anything) and spins a contracting business
    # as improving, now also disagreeing with the withheld positive flag. The margin fact (it
    # widened) is still stated, but framed honestly as managing decline, not progress.
    rev = {2020: 200 * CR, 2021: 150 * CR, 2022: 110 * CR}      # sales down ~26%/yr
    prof = {2020: 20 * CR, 2021: 17 * CR, 2022: 15 * CR}        # profit down ~13%/yr (slower)
    joined = " ".join(trend_points(rev, prof))
    assert margins_improving(rev, prof) is True                 # margins did widen (profit fell slower)
    assert "outpaced sales" not in joined                       # not a positive "outpaced" headline
    assert "margins have been improving" not in joined          # nor a bare "improving" for a shrinking biz
    assert "shrunk more slowly than sales" in joined            # honest: profit fell slower, margins widened
    assert "managing decline" in joined                         # framed as contraction, not progress


def test_trend_points_empty_when_insufficient_history():
    assert trend_points({2022: 100}, {2022: 10}) == []           # too few years


def test_trend_points_suppresses_margin_direction_when_a_cagr_is_a_base_effect():
    # WHY (real money, consistency): the margin-direction claim compares profit vs sales growth. When
    # EITHER is a >100%/yr base-effect CAGR (a trough recovery), that comparison is dominated by the
    # base effect, not a real margin trend -- and stating "margins have been improving" flatly
    # contradicts the base-effect caveat the growth line now prints for the SAME numbers. Suppress the
    # margin-direction line in that case; a normal-growth pair still gets it.
    prof = {2021: 5 * CR, 2022: 50 * CR, 2023: 200 * CR, 2024: 500 * CR}       # ~364%/yr base effect
    rev = {2021: 1000 * CR, 2022: 1200 * CR, 2023: 1400 * CR, 2024: 1600 * CR}  # ~17%/yr
    joined = " ".join(trend_points(rev, prof))
    assert "margins have been improving" not in joined
    assert "margins have been under pressure" not in joined
    normal = " ".join(trend_points({2020: 100 * CR, 2021: 110 * CR, 2022: 121 * CR},
                                    {2020: 10 * CR, 2021: 12 * CR, 2022: 15 * CR}))
    assert "margins have been improving" in normal            # a normal pair still gets the claim


def test_trend_points_qualifies_an_absurd_low_base_cagr():
    # WHY (real money, honesty): a cyclical/turnaround whose EARLIEST cross-verified year was a trough
    # (tiny positive profit) shows an astronomical CAGR off that low base -- "growing 364% a year"
    # reads as a bug and overstates a sustainable trend (no business compounds >100%/yr for years, and
    # a genuine small-company hyper-grower still isn't sustaining that rate). Above 100%/yr, describe
    # it qualitatively instead of quoting the absurd precise rate; the year-by-year swing caveat and
    # the raw figures still convey the real picture. A NORMAL CAGR keeps its precise rate.
    prof = {2021: 5 * CR, 2022: 50 * CR, 2023: 200 * CR, 2024: 500 * CR}       # trough -> ~364%/yr
    rev = {2021: 1000 * CR, 2022: 1200 * CR, 2023: 1400 * CR, 2024: 1600 * CR}  # ~17%/yr, normal
    joined = " ".join(trend_points(rev, prof))
    assert "364% a year" not in joined                            # the absurd precise rate is gone
    assert "low or one-off starting year" in joined               # qualified as a base effect instead
    assert "17% a year" in joined                                 # a normal CAGR still shows its rate


def test_limited_history_note_thresholds():
    # WHY (real money, rigor): fewer than the 3 years a trend needs is a SHORT track record. The
    # note fires for 1-2 years, is silent at >=3 (a real trend is available), and silent at 0
    # (nothing cross-verified -> handled as no-data, not a "short history" caveat).
    assert limited_history_note(1) is not None and "1 year" in limited_history_note(1)     # singular
    assert limited_history_note(2) is not None and "2 years" in limited_history_note(2)     # plural
    assert "short track record" in limited_history_note(2)
    assert limited_history_note(3) is None
    assert limited_history_note(0) is None


# --- structured trend_improving signal (decoupled from the UI prose) ---

def test_trend_improving_true_on_growing_sales():
    rev = {2020: 100 * CR, 2021: 110 * CR, 2022: 121 * CR}       # ~10%/yr > 3% floor
    flat = {2020: 10 * CR, 2021: 10 * CR, 2022: 10 * CR}
    assert trend_improving(rev, flat) is True


def test_trend_improving_true_on_improving_margins_even_if_sales_flat():
    rev = {2020: 100 * CR, 2021: 101 * CR, 2022: 102 * CR}       # ~1%/yr (below growth floor)
    prof = {2020: 10 * CR, 2021: 12 * CR, 2022: 15 * CR}         # profit faster -> margins up
    assert trend_improving(rev, prof) is True


def test_trend_improving_false_on_flat_and_on_thin_history():
    flat = {2020: 100 * CR, 2021: 100 * CR, 2022: 100 * CR}
    assert trend_improving(flat, flat) is False
    assert trend_improving({2022: 100 * CR}, {2022: 10 * CR}) is False   # <3 yrs -> no signal


def test_trend_improving_false_when_shrinking():
    rev = {2020: 121 * CR, 2021: 110 * CR, 2022: 100 * CR}       # declining
    prof = {2020: 15 * CR, 2021: 12 * CR, 2022: 10 * CR}
    assert trend_improving(rev, prof) is False


def test_trend_improving_false_when_both_lines_shrink_even_if_margins_widen():
    # WHY (real money, the "melting ice cube" value trap): a business whose sales AND profit have
    # BOTH compounded DOWNWARD over the window is contracting. When profit shrinks SLOWER than sales,
    # margins_improving is True (profit "outpaced" sales), which short-circuited trend_improving to
    # True -- handing a shrinking business the same +1 in the suggestion score as a growing one, the
    # mirror of the sales-up/profit-down trap already guarded. Widening margins on a falling top line
    # is managing decline, not an improving trend; no long-term investor credits it. trend_points
    # still prints the honest sub-signals (both shrinking; margins improving); only the OVERALL
    # positive flag is withheld, so the flag never claims "improving" for a contracting business.
    rev = {2020: 200 * CR, 2021: 150 * CR, 2022: 110 * CR}      # sales down ~26%/yr
    prof = {2020: 20 * CR, 2021: 17 * CR, 2022: 15 * CR}        # profit down ~13%/yr (slower)
    assert margins_improving(rev, prof) is True                 # profit outpaced sales -> margins widened
    assert trend_improving(rev, prof) is False                  # but the business is CONTRACTING
    assert sum("shrink" in p for p in trend_points(rev, prof)) >= 2   # prose still flags both lines down


def test_trend_improving_false_when_revenue_grows_but_profit_shrinks():
    # WHY (real money, value trap / unprofitable growth): sales compounding while PROFIT actually
    # SHRINKS is the textbook top-line-at-any-cost value trap -- a value investor reads it as
    # deterioration, not improvement. Crediting it +1 in the suggestion score also flatly
    # contradicts the "profit shrinking / margins under pressure" lines trend_points prints for the
    # SAME numbers, breaking the flag<->prose agreement this signal promises. Revenue growth no
    # longer rescues a falling bottom line; profit growth or a real margin expansion still stands.
    rev = {2020: 100 * CR, 2021: 110 * CR, 2022: 121 * CR}      # sales up ~10%/yr
    prof = {2020: 20 * CR, 2021: 15 * CR, 2022: 11 * CR}        # profit DOWN ~26%/yr
    assert margins_improving(rev, prof) is False
    assert trend_improving(rev, prof) is False                 # was True on the revenue leg alone


def test_trend_improving_true_on_profit_growth_even_if_margins_compress():
    # WHY: a bottom line compounding above the floor is genuine improvement and stands on its own,
    # even when revenue grew FASTER (so margins technically compressed). Only revenue growth with a
    # FALLING bottom line is the value trap; a rising bottom line is not, so it must still count.
    rev = {2020: 100 * CR, 2021: 130 * CR, 2022: 169 * CR}     # sales up ~30%/yr
    prof = {2020: 10 * CR, 2021: 11.5 * CR, 2022: 13.2 * CR}   # profit up ~15%/yr (real growth)
    assert margins_improving(rev, prof) is False               # slower than sales -> margins compressed
    assert trend_improving(rev, prof) is True                  # profit growth stands on its own


# --- margin direction must compare revenue vs profit growth over the SAME period ---

def test_margin_direction_not_claimed_when_windows_do_not_overlap_enough():
    # WHY (real money, CA-level correctness): revenue cross-verified FY2019-FY2021, profit only
    # FY2021-FY2023. They share ONE year (2021), so there is no >=3-year common window to compare
    # growth over. The end-to-end margin identity (margin_last/margin_first = profit_ratio /
    # revenue_ratio) only holds when both cover the SAME first and last year; pitting profit's
    # 2021-2023 rise against revenue's 2019-2021 rise is two different eras. No margin claim.
    rev = {2019: 100 * CR, 2020: 110 * CR, 2021: 121 * CR}
    prof = {2021: 10 * CR, 2022: 13 * CR, 2023: 17 * CR}
    assert margins_improving(rev, prof) is None
    assert "margin" not in " ".join(trend_points(rev, prof))


def test_trend_improving_not_faked_by_comparing_mismatched_windows():
    # Sales cross-verified FY2015-FY2021 (declining), profit only FY2021-FY2023 (barely up).
    # Neither is "growing" on its own, and the windows overlap in a single year, so there is no
    # valid common period to judge margins over. The old code compared profit's recent 2021-2023
    # growth against sales' 2015-2021 DECLINE -- fabricating a "margins improving" signal for two
    # different eras that would wrongly bump this name up the suggestion ranking.
    rev = {2015: 200 * CR, 2018: 150 * CR, 2021: 100 * CR}
    prof = {2021: 10 * CR, 2022: 10 * CR, 2023: 10.2 * CR}
    assert trend_improving(rev, prof) is False


def test_margin_direction_uses_common_window_and_ignores_interior_gaps():
    # Revenue cross-verified only FY2020/FY2022/FY2024 (FY2021, FY2023 didn't agree across
    # sources), profit in all of FY2020-FY2024. The endpoints match, so margins ARE comparable
    # over the common window despite revenue's interior gaps -- a same-year-SET guard would wrongly
    # suppress this; a common-window comparison keeps the real signal.
    rev = {2020: 100 * CR, 2022: 110 * CR, 2024: 121 * CR}       # ~5%/yr
    prof = {2020: 10 * CR, 2021: 11 * CR, 2022: 13 * CR, 2023: 15 * CR, 2024: 18 * CR}  # faster
    assert margins_improving(rev, prof) is True
    assert "margins have been improving" in " ".join(trend_points(rev, prof))


# --- earnings_volatility_point: no blind spots for cyclical/lumpy-revenue businesses ---

def test_earnings_volatility_flags_a_real_cyclical_swing():
    # WHY: live-verified against real JSW Steel data (a genuine cyclical steel producer): profit
    # swung +115% then -61% year over year. A single year's ROE/margin here would be badly
    # misleading -- 2024 alone would look like a standout year, 2025 alone mediocre, purely from
    # steel-cycle timing, not a change in the underlying business.
    profit = {2023: 4142 * CR, 2024: 8892 * CR, 2025: 3498 * CR}
    point = earnings_volatility_point(profit)
    assert point is not None
    assert "swung" in point.lower() or "volatil" in point.lower()


def test_earnings_volatility_names_the_pe_valuation_as_the_cyclical_trap():
    # WHY (real money, margin-of-safety / the cyclical P/E value trap): the caveat already warned a
    # single year's ROE/margin can mislead for a cyclical, but it OMITTED the P/E -- yet the valuation
    # is computed from the SAME latest-year earnings, so a LOW P/E on a cyclical at PEAK earnings reads
    # "cheap" while actually being the classic value trap (earnings about to normalize DOWN), the
    # single most important cyclical mistake a non-expert makes. The caveat must name the P/E, not
    # just ROE/margin, and explain the inverse P/E-vs-profit relationship in both directions.
    profit = {2023: 4142 * CR, 2024: 8892 * CR, 2025: 3498 * CR}    # real JSW Steel cyclical swing
    point = earnings_volatility_point(profit)
    assert point is not None
    assert "p/e" in point.lower()                                   # valuation named, not just ROE/margin
    assert "trap" in point.lower() and "peak" in point.lower()      # the low-P/E-at-peak trap
    assert "mid-cycle" in point.lower()                             # the corrective: normalize earnings


def test_earnings_volatility_silent_for_a_smooth_grower():
    # Live-verified against real TCS data: consistent ~1-9%/yr growth, no cyclical swing.
    profit = {2023: 42225 * CR, 2024: 46004 * CR, 2025: 48675 * CR, 2026: 49332 * CR}
    assert earnings_volatility_point(profit) is None


def test_earnings_volatility_needs_at_least_two_yoy_growth_points():
    assert earnings_volatility_point({2024: 100 * CR}) is None            # 1 year, no growth rate
    assert earnings_volatility_point({2023: 100 * CR, 2024: 110 * CR}) is None  # only 1 growth rate


def test_earnings_volatility_guards_against_a_zero_base_year():
    # A year with zero profit can't produce a meaningful % growth rate off it; must not crash.
    profit = {2022: 0.0, 2023: 100 * CR, 2024: 50 * CR}
    assert earnings_volatility_point(profit) is None   # <2 usable growth points after the guard


def test_earnings_volatility_uses_qualitative_wording_across_a_loss_to_profit_crossing():
    # WHY (real money, honesty): a year-over-year growth % is ill-defined through a SIGN CHANGE --
    # a small loss between profit years explodes the rate into an absurd, alarming figure
    # (live-repro: "a 20201-percentage-point range"). When profit crossed between losses and
    # profits, fire the SAME earning-power caveat but phrase it qualitatively, never quoting a
    # nonsensical percentage a parent would read as a typo.
    point = earnings_volatility_point({2023: 100 * CR, 2024: -1 * CR, 2025: 200 * CR})
    assert point is not None
    assert "losses and profits" in point.lower()
    assert "percentage-point" not in point          # no absurd computed % shown
    assert "20201" not in point
    # a genuine turnaround (losses then a profit) is also worded qualitatively
    tp = earnings_volatility_point({2023: -100 * CR, 2024: -50 * CR, 2025: 50 * CR})
    assert tp is not None and "losses and profits" in tp.lower()


def test_earnings_volatility_keeps_percentage_wording_for_all_positive_cyclicals():
    # regression: an all-positive cyclical (no sign change) still gets the precise pp-range wording
    point = earnings_volatility_point({2023: 4142 * CR, 2024: 8892 * CR, 2025: 3498 * CR})
    assert point is not None and "percentage-point range" in point


# --- revenue_volatility_point: fills a real gap earnings_volatility_point cannot ---

def test_revenue_volatility_flags_lumpy_project_based_revenue():
    # WHY: live-verified against real Brigade Enterprises data (a real-estate developer): 4
    # cross-verified REVENUE years swinging sharply, but only 1 cross-verified PROFIT year
    # (percentage-of-completion accounting makes profit recognition lumpier and harder to
    # cross-verify), so earnings_volatility_point can NEVER fire for this name -- revenue data
    # alone must be able to surface the real lumpiness.
    revenue = {2022: 1000 * CR, 2023: 1387 * CR, 2024: 1050 * CR, 2025: 1450 * CR}
    point = revenue_volatility_point(revenue)
    assert point is not None
    assert "revenue" in point.lower() and ("swung" in point.lower() or "volatil" in point.lower())


def test_revenue_volatility_silent_for_a_smooth_grower():
    revenue = {2023: 100 * CR, 2024: 108 * CR, 2025: 115 * CR, 2026: 120 * CR}
    assert revenue_volatility_point(revenue) is None


def test_revenue_volatility_uses_a_lower_threshold_than_profit():
    # WHY: operating leverage means revenue swings LESS than profit for the same underlying
    # volatility (live-verified: JSW Steel's PROFIT swung 175pp but its REVENUE only 13pp), so
    # reusing profit's 40pp threshold for revenue would miss genuine project-based lumpiness.
    # These are the EXACT real, cross-verified swing magnitudes for three independent real-estate
    # developers (Brigade 38.8pp, DLF 38.0pp, Sobha 37.2pp) -- all comfortably under the 40pp
    # profit threshold, so a shared threshold would have silently missed all three real cases.
    for real_swing_pct in (38.8, 38.0, 37.2):
        # One flat year (0% growth) then one year growing by the target swing: the max-min
        # spread across those two growth legs equals exactly real_swing_pct.
        rev = {2022: 1000 * CR, 2023: 1000 * CR, 2024: 1000 * (1 + real_swing_pct / 100) * CR}
        point = revenue_volatility_point(rev)
        assert point is not None, f"{real_swing_pct}pp swing should fire under the 25pp threshold"


def test_trend_points_prefers_profit_volatility_when_both_swing():
    # WHY (avoid repetitive messaging): when BOTH profit and revenue swing sharply (live-verified
    # pattern seen in DLF), show only ONE volatility caveat, not two near-duplicate sentences.
    # Profit (the bottom line) is the more decision-relevant one and takes priority.
    rev = {2022: 1000 * CR, 2023: 1380 * CR, 2024: 1050 * CR}     # also swings
    prof = {2022: 100 * CR, 2023: 215 * CR, 2024: 84 * CR}         # swings even more
    pts = trend_points(rev, prof)
    volatility_pts = [p for p in pts if "swung sharply" in p]
    assert len(volatility_pts) == 1
    assert "Profit" in volatility_pts[0]


def test_trend_points_falls_back_to_revenue_volatility_when_profit_data_is_thin():
    # The actual Brigade-shaped case: profit has too few cross-verified years to judge volatility
    # at all, but revenue has enough and swings -- the reader should still see SOMETHING, not
    # nothing, about the real lumpiness the data shows.
    rev = {2022: 1000 * CR, 2023: 1387 * CR, 2024: 1050 * CR, 2025: 1450 * CR}
    prof = {2024: 90 * CR}                                          # only 1 year -> no signal
    pts = trend_points(rev, prof)
    volatility_pts = [p for p in pts if "swung sharply" in p]
    assert len(volatility_pts) == 1
    assert "Revenue" in volatility_pts[0]


def test_trend_points_does_not_fall_back_to_revenue_when_profit_is_confirmed_smooth():
    # WHY (regression, adversarial review): the fallback must trigger ONLY when profit data is
    # too THIN to judge (see the test above), not merely because profit turned out smooth. A
    # confirmed-smooth bottom line despite one lumpy revenue year is a reasonable case to NOT
    # caveat at all -- the business absorbed the swing before it reached earnings, and showing
    # "steady profit growth" right next to "but revenue swung sharply, don't trust a single year"
    # reads as contradictory guidance about the same business.
    rev = {2022: 1000 * CR, 2023: 1300 * CR, 2024: 1000 * CR}   # a real ~30pp swing (one lumpy yr)
    prof = {2022: 100 * CR, 2023: 108 * CR, 2024: 118 * CR, 2025: 127 * CR}  # steady ~8%/yr, ample data
    pts = trend_points(rev, prof)
    assert not any("swung sharply" in p for p in pts)   # profit had enough data and is smooth


# --- pipeline integration: the leverage trend reaches report.insights (offline) ---

def _rising_leverage_sources():
    """Two agreeing sources whose debt/equity series show D/E climbing 0.20 -> 0.60, plus a
    couple of other cross-verifying figures so the report builds normally."""
    from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource

    series = {
        "total_debt": {2022: 20 * CR, 2023: 40 * CR, 2024: 60 * CR},
        "equity": {2022: 100 * CR, 2023: 100 * CR, 2024: 100 * CR},
        "net_profit": {2022: 100 * CR, 2023: 110 * CR, 2024: 120 * CR},
        "operating_cash_flow": {2022: 40 * CR, 2023: 45 * CR, 2024: 50 * CR},  # cum 135/330 = 41%
    }
    scalar = {"total_debt": 60 * CR, "equity": 100 * CR, "net_profit": 120 * CR,
              "operating_cash_flow": 50 * CR}

    class _FakeSrc(FigureSource):
        def __init__(self, sid):
            self.source_id = sid

        def figures(self, symbol):
            return {n: scalar.get(n) for n in FRAMEWORK_FIGURES}

        def figures_by_year(self, symbol):
            return series

    return [_FakeSrc("yfinance"), _FakeSrc("screener")]


def test_leverage_trend_reaches_report_insights_for_a_non_bank(monkeypatch):
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    report = pipeline.build_report_for_symbol("TESTCO", _rising_leverage_sources())
    assert any("Leverage (debt/equity) has risen" in i for i in report.insights)


def test_leverage_trend_is_skipped_for_a_bank(monkeypatch):
    # WHY: a bank/NBFC is leveraged by design, so a rising D/E is its business model, not a risk
    # signal -- it must not surface as a leverage-risk insight (same reason banks use the ROA lens).
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Banks - Regional")
    report = pipeline.build_report_for_symbol("TESTBANK", _rising_leverage_sources())
    assert not any("Leverage (debt/equity)" in i for i in report.insights)


def test_cash_conversion_quality_reaches_report_insights_for_a_non_bank(monkeypatch):
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    report = pipeline.build_report_for_symbol("TESTCO", _rising_leverage_sources())
    assert any("cumulative operating cash flow was only 41%" in i for i in report.insights)


def test_cash_conversion_quality_is_skipped_for_a_bank(monkeypatch):
    # A bank's operating cash flow is dominated by lending/deposit flows, not the industrial
    # profit-to-cash relationship this measures -- must not surface as a quality-of-earnings flag.
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Banks - Regional")
    report = pipeline.build_report_for_symbol("TESTBANK", _rising_leverage_sources())
    assert not any("cumulative operating cash flow" in i for i in report.insights)


def _growing_equity_sources():
    """Two agreeing sources: net profit 20cr on equity that GREW 80->100cr over two years, so
    average equity (90) differs from closing equity (100) -- ROE 22.2% vs 20% on point."""
    from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource

    series = {
        "net_profit": {2023: 18 * CR, 2024: 20 * CR},
        "equity": {2023: 80 * CR, 2024: 100 * CR},
        "total_assets": {2023: 160 * CR, 2024: 200 * CR},
    }
    scalar = {"net_profit": 20 * CR, "equity": 100 * CR, "total_assets": 200 * CR}

    class _FakeSrc(FigureSource):
        def __init__(self, sid):
            self.source_id = sid

        def figures(self, symbol):
            return {n: scalar.get(n) for n in FRAMEWORK_FIGURES}

        def figures_by_year(self, symbol):
            return series

    return [_FakeSrc("yfinance"), _FakeSrc("screener")]


def test_return_ratios_use_average_equity_end_to_end(monkeypatch):
    # WHY (CA-level rigor): the pipeline must feed the cross-verified prior-year balance into the
    # ratio suite so ROE is measured on AVERAGE equity, not the closing snapshot -- otherwise a
    # company that grew its equity during the year reads a too-low ROE.
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    report = pipeline.build_report_for_symbol("GROWCO", _growing_equity_sources())
    roe_line = next((i for i in report.insights if "ROE" in i), "")
    assert "on average equity" in roe_line       # averaged, not the closing snapshot
    assert "ROE 22%" in roe_line                  # 20 / ((80+100)/2) = 22.2%, not 20%


def _two_sources_from_series(fseries):
    """Two agreeing sources built from an explicit {figure: {year: value}} map (scalar = each
    figure's latest year), for exercising the pipeline's prior-year / averaging wiring offline."""
    from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource
    scalar = {fig: ys[max(ys)] for fig, ys in fseries.items() if ys}

    class _S(FigureSource):
        def __init__(self, sid):
            self.source_id = sid

        def figures(self, symbol):
            return {n: scalar.get(n) for n in FRAMEWORK_FIGURES}

        def figures_by_year(self, symbol):
            return fseries

    return [_S("yfinance"), _S("screener")]


def test_roce_averaging_gated_when_equity_and_debt_latest_years_diverge(monkeypatch):
    # WHY (found by adversarial review): ROCE averages equity + debt; if their latest cross-verified
    # years diverge (yfinance often leaves the newest Total Debt empty), pairing their independent
    # openings would blend an equity (Y, Y-1) window with a debt (Y-1, Y-2) window while labeling it
    # a clean "average capital". Gate it: divergent coverage -> point capital, no misleading label.
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    divergent = {
        "net_profit": {2023: 18 * CR, 2024: 20 * CR},
        "ebit": {2023: 25 * CR, 2024: 30 * CR},
        "equity": {2023: 80 * CR, 2024: 100 * CR},         # latest FY2024
        "total_debt": {2022: 30 * CR, 2023: 40 * CR},      # latest FY2023 -- diverges
    }
    insights = pipeline.build_report_for_symbol("DIVCO", _two_sources_from_series(divergent)).insights
    roce = next((i for i in insights if "ROCE" in i), "")
    roe = next((i for i in insights if "ROE" in i), "")
    assert roce and "on average capital" not in roce       # point capital when years diverge
    assert "on average equity" in roe                        # ROE (single figure) still averages


def test_limited_history_caveat_reaches_insights_for_a_thin_track_record(monkeypatch):
    # WHY (real money, rigor): a recently-listed company (only 1-2 cross-verified years) shows
    # latest-year ratios and can read HIGH-confidence FAVORABLE, since the verdict's confidence
    # counts how many METRICS cross-verified, not how many YEARS. Surface an explicit "short track
    # record" caveat so a single flattering year isn't over-read as a proven record. An established
    # company (>=3 cross-verified years) gets no such caveat.
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    thin = {"net_profit": {2024: 20 * CR, 2025: 24 * CR},           # only 2 cross-verified years
            "equity": {2024: 100 * CR, 2025: 110 * CR},
            "revenue": {2024: 200 * CR, 2025: 240 * CR}}
    thin_insights = pipeline.build_report_for_symbol("NEWCO", _two_sources_from_series(thin)).insights
    assert any("short track record" in i for i in thin_insights)
    rich = {"net_profit": {2022: 16 * CR, 2023: 18 * CR, 2024: 20 * CR, 2025: 24 * CR},
            "equity": {2022: 90 * CR, 2023: 95 * CR, 2024: 100 * CR, 2025: 110 * CR},
            "revenue": {2022: 160 * CR, 2023: 180 * CR, 2024: 200 * CR, 2025: 240 * CR}}
    rich_insights = pipeline.build_report_for_symbol("OLDCO", _two_sources_from_series(rich)).insights
    assert not any("short track record" in i for i in rich_insights)


def test_roce_averaging_applies_when_equity_and_debt_share_latest_year(monkeypatch):
    import src.pipeline as pipeline
    monkeypatch.setattr(pipeline, "compute_median_pe", lambda s: None, raising=False)
    from src.analysis import bank_framework
    monkeypatch.setattr(bank_framework, "_yfinance_industry", lambda s: "Auto Components")
    aligned = {
        "net_profit": {2023: 18 * CR, 2024: 20 * CR},
        "ebit": {2023: 25 * CR, 2024: 30 * CR},
        "equity": {2023: 80 * CR, 2024: 100 * CR},
        "total_debt": {2023: 30 * CR, 2024: 40 * CR},      # same latest year as equity -> clean
    }
    insights = pipeline.build_report_for_symbol("ALGNCO", _two_sources_from_series(aligned)).insights
    roce = next((i for i in insights if "ROCE" in i), "")
    assert "on average capital" in roce
