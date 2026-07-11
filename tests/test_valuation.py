import pandas as pd

from src.analysis import valuation
from src.analysis.valuation import (
    _price_by_fiscal_year,
    compute_median_pe,
    median_pe_from_annual_shares,
    median_pe_from_annuals,
    median_pe_from_eps,
)
from src.pipeline import build_company_report
from src.research.report import ValuationTier
from src.research.verification import SourcedValue


def test_compute_median_pe_uses_split_adjusted_not_dividend_adjusted_prices(monkeypatch):
    # WHY (real money, data quality; live-verified on ITC/RELIANCE/NESTLEIND): historical median
    # P/E must pair prices with yfinance's income_stmt EPS, which is retroactively restated to the
    # CURRENT split-adjusted share basis. yfinance's Close is ALWAYS split-adjusted; the
    # auto_adjust flag only toggles the DIVIDEND adjustment. The old default (auto_adjust=True)
    # additionally applied dividends, understating past prices (~11% over 3y on a ~3%-yield name
    # like ITC) -> historical P/E read low -> the "cheap vs its own history" call biased toward
    # "expensive". compute_median_pe must fetch auto_adjust=False (split-adjusted, NOT
    # dividend-adjusted) -- the same basis the EPS is on.
    import yfinance as yf
    idx = pd.to_datetime(["2023-03-31", "2024-03-31"])
    div_unadjusted = pd.DataFrame({"Close": [120.0, 140.0]}, index=idx)  # auto_adjust=False
    div_adjusted = pd.DataFrame({"Close": [100.0, 130.0]}, index=idx)    # auto_adjust=True (lower)
    income = pd.DataFrame(
        {pd.Timestamp("2024-03-31"): [10.0, 1000.0], pd.Timestamp("2023-03-31"): [10.0, 900.0]},
        index=["Diluted EPS", "Net Income"])

    class FakeTicker:
        @property
        def income_stmt(self):
            return income

        @property
        def info(self):
            return {}

        def history(self, period=None, auto_adjust=True, **kw):
            return div_unadjusted if auto_adjust is False else div_adjusted

    monkeypatch.setattr(yf, "Ticker", lambda *a, **k: FakeTicker())
    m = compute_median_pe("X")
    # dividend-UNadjusted prices 120/140 over EPS 10/10 -> P/E 12 and 14 -> median 13.0, NOT the
    # dividend-adjusted 100/130 -> P/E 10 and 13 -> median 11.5 the old default produced.
    assert m == 13.0


def test_price_by_fiscal_year_uses_actual_period_end_not_hardcoded_march():
    # WHY (rigor): a Dec-FY-end company's December EPS must be priced at December, not a hardcoded
    # 31 March, or its median P/E (which now feeds the margin-of-safety ranking) is distorted.
    idx = pd.to_datetime(["2023-11-30", "2023-12-31", "2024-02-29", "2024-03-31", "2024-06-30"])
    hist = pd.DataFrame({"Close": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=idx)
    year_ends = {2023: pd.Timestamp("2023-12-31"), 2024: pd.Timestamp("2024-03-31")}
    prices = _price_by_fiscal_year(hist, year_ends)
    assert prices[2023] == 110.0        # close on the Dec period-end, not a March price
    assert prices[2024] == 130.0        # close on 31 Mar 2024


def test_price_by_fiscal_year_takes_last_close_on_or_before_end():
    idx = pd.to_datetime(["2024-03-15", "2024-03-28"])   # no close exactly on 31 Mar
    hist = pd.DataFrame({"Close": [200.0, 210.0]}, index=idx)
    prices = _price_by_fiscal_year(hist, {2024: pd.Timestamp("2024-03-31")})
    assert prices[2024] == 210.0        # most recent close on/before the period-end


def test_median_pe_from_annuals():
    # eps 2024=10, 2023=9; pe 200, 200 -> median 200
    assert median_pe_from_annuals({2024: 100, 2023: 90}, {2024: 2000, 2023: 1800}, 10) == 200


def test_median_pe_needs_two_years():
    assert median_pe_from_annuals({2024: 100}, {2024: 2000}, 10) is None


def test_median_pe_no_shares_is_none():
    assert median_pe_from_annuals({2024: 100, 2023: 90}, {2024: 2000, 2023: 1800}, 0) is None


def test_median_pe_skips_loss_years():
    # 2023 loss -> skipped, only 2024 remains -> < 2 usable -> None
    assert median_pe_from_annuals({2024: 100, 2023: -50}, {2024: 2000, 2023: 1800}, 10) is None


def test_median_pe_from_eps_uses_period_eps():
    # price/eps per year: 10, 12, 15 -> median 12. Uses each year's own EPS (no dilution error).
    assert median_pe_from_eps({2022: 100, 2023: 120, 2024: 150},
                              {2022: 10, 2023: 10, 2024: 10}) == 12
    assert median_pe_from_eps({2022: 100}, {2022: 10}) is None           # <2 years
    assert median_pe_from_eps({2022: 100, 2023: 120},
                              {2022: -5, 2023: 10}) is None               # only 1 positive-EPS yr


def test_median_pe_from_annual_shares_uses_each_years_own_share_count():
    # WHY (real money, margin-of-safety accuracy): a diluting company had FEWER shares in the past,
    # so its past EPS was HIGHER than net_profit / TODAY's shares. Using each year's OWN weighted
    # share count (yfinance's "Average Shares" rows) computes the correct per-year EPS -> the correct
    # historical median P/E. The current-shares fallback (median_pe_from_annuals) instead understates
    # past EPS, inflating the historical median so the stock reads CHEAPER than it was -- the
    # dangerous direction for a margin-of-safety call.
    net_profit = {2022: 1000.0, 2023: 1200.0, 2024: 1500.0}
    price = {2022: 100.0, 2023: 120.0, 2024: 150.0}
    shares = {2022: 100.0, 2023: 110.0, 2024: 125.0}             # share count grew (dilution)
    m = median_pe_from_annual_shares(net_profit, price, shares)  # EPS 10/10.9/12 -> P/E 10/11/12.5
    assert round(m, 2) == 11.0
    biased = median_pe_from_annuals(net_profit, price, 125.0)    # current-shares fallback
    assert biased > m                                           # reads a higher median (looks cheaper)


def test_median_pe_from_annual_shares_guards_bad_years():
    # A year missing its share count, or with a non-positive share/profit/price, is skipped; <2 -> None.
    assert median_pe_from_annual_shares({2024: 100.0}, {2024: 2000.0}, {2024: 10.0}) is None   # 1 year
    assert median_pe_from_annual_shares(
        {2023: 100.0, 2024: 100.0}, {2023: 2000.0, 2024: 2000.0},
        {2023: 0.0, 2024: 10.0}) is None            # zero shares in 2023 -> only 1 usable year


def test_compute_median_pe_prefers_per_year_shares_over_current_shares(monkeypatch):
    # WHY (real money): with no per-year EPS row, prefer per-year AVERAGE SHARES (correct per-year
    # EPS, no dilution bias) over the current-shares fallback. A diluting company must not read
    # cheaper than it was because past profit got divided by today's larger share count.
    import yfinance as yf
    idx = pd.to_datetime(["2023-03-31", "2024-03-31"])
    hist = pd.DataFrame({"Close": [100.0, 150.0]}, index=idx)
    income = pd.DataFrame(
        {pd.Timestamp("2024-03-31"): [1500.0, 125.0], pd.Timestamp("2023-03-31"): [1000.0, 100.0]},
        index=["Net Income", "Diluted Average Shares"])   # NO EPS row; per-year shares present

    class FakeTicker:
        @property
        def income_stmt(self):
            return income

        @property
        def info(self):
            return {"sharesOutstanding": 125.0}   # the biased current-shares fallback basis

        def history(self, period=None, auto_adjust=True, **kw):
            return hist

    monkeypatch.setattr(yf, "Ticker", lambda *a, **k: FakeTicker())
    # per-year: 2023 EPS 1000/100=10 -> P/E 10; 2024 EPS 1500/125=12 -> P/E 12.5; median 11.25.
    # The current-shares (125) fallback would read 12.5 for both years -> median 12.5 (cheaper-looking).
    assert compute_median_pe("X") == 11.25


def _cur_pe(value):
    return {"current_pe": [SourcedValue(value, "a"), SourcedValue(value, "b")]}  # cross-verified


def test_valuation_cheap_with_computed_median():
    r = build_company_report("X", _cur_pe(18.0), median_pe=25.0)  # 0.72x -> cheap
    assert r.verdict.valuation == ValuationTier.CHEAP


def test_valuation_expensive_with_computed_median():
    r = build_company_report("X", _cur_pe(40.0), median_pe=25.0)  # 1.6x -> expensive
    assert r.verdict.valuation == ValuationTier.EXPENSIVE


def test_valuation_unknown_without_median():
    r = build_company_report("X", _cur_pe(18.0))  # no median -> unknown
    assert r.verdict.valuation == ValuationTier.UNKNOWN


def test_dividend_yield_cross_verifies_with_wider_tolerance():
    # WHY: live-verified real-world cross-provider variance for dividend yield (2-17% typical), so
    # this figure gets a wider band; a genuinely large disagreement still correctly conflicts (see
    # the next test).
    figs = {"dividend_yield_pct": [SourcedValue(5.0, "yfinance"), SourcedValue(5.8, "screener")]}
    r = build_company_report("X", figs)  # 16% apart -- would CONFLICT at the 2% default
    fig = next(f for f in r.figures if f.name == "dividend_yield_pct")
    assert fig.is_trustworthy
    assert fig.value == 5.4                          # median of the agreeing cluster


def test_current_pe_cross_verifies_with_a_wider_tolerance_than_2pct():
    # WHY (real money, live-verified across RELIANCE 3.4% / TCS 6.8% / HDFCBANK 9.3%): yfinance
    # trailingPE and Screener's Stock P/E legitimately differ several percent (different
    # trailing-EPS windows, consolidated-vs-standalone basis, price snapshot), NOT the parse/scale
    # errors the 2% default guards. At 2% the current P/E was CONFLICTing for major stocks,
    # withholding the WHOLE valuation tier (the core margin-of-safety signal) as "unknown".
    figs = {"current_pe": [SourcedValue(18.41, "yfinance"), SourcedValue(16.70, "screener")]}
    r = build_company_report("X", figs)   # ~9.3% apart -- would CONFLICT at the 2% default
    fig = next(f for f in r.figures if f.name == "current_pe")
    assert fig.is_trustworthy


def test_current_pe_still_conflicts_on_a_gross_error():
    # A scale/parse error (e.g. 22.7 misread as 227) is ~10x off -- must still be withheld, never
    # blended into a fabricated "valuation" the wider band was meant to enable.
    figs = {"current_pe": [SourcedValue(22.7, "yfinance"), SourcedValue(227.0, "screener")]}
    r = build_company_report("X", figs)
    fig = next(f for f in r.figures if f.name == "current_pe")
    assert not fig.is_trustworthy


def test_dividend_yield_still_conflicts_when_wildly_different():
    # Real case (TCS, live-verified): yfinance 6.03% vs Screener 3.12%, a 48% gap, likely a
    # special-dividend timing difference between providers. Even the wider 25% band must not
    # force this to verify; a genuinely disputed figure stays withheld, not averaged.
    figs = {"dividend_yield_pct": [SourcedValue(6.03, "yfinance"), SourcedValue(3.12, "screener")]}
    r = build_company_report("X", figs)
    fig = next(f for f in r.figures if f.name == "dividend_yield_pct")
    assert not fig.is_trustworthy
