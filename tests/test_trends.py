from src.analysis.trends import (
    cagr,
    cash_conversion_quality_point,
    earnings_volatility_point,
    leverage_trend_point,
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


def test_cash_conversion_quality_needs_three_years_and_positive_cumulative_profit():
    ocf = {2023: 40 * CR, 2024: 50 * CR}
    profit = {2023: 100 * CR, 2024: 110 * CR}
    assert cash_conversion_quality_point(ocf, profit) is None            # only 2 common years
    loss = {2022: -100 * CR, 2023: -90 * CR, 2024: -80 * CR}             # cumulative loss period
    assert cash_conversion_quality_point({2022: 5 * CR, 2023: 5 * CR, 2024: 5 * CR}, loss) is None


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


def test_trend_points_empty_when_insufficient_history():
    assert trend_points({2022: 100}, {2022: 10}) == []           # too few years


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
