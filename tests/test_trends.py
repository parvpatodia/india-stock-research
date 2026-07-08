from src.analysis.trends import cagr, trend_points, verified_series

CR = 1e7


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
