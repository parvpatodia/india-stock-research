import pandas as pd

from src.analysis.valuation import (
    _price_by_fiscal_year,
    median_pe_from_annuals,
    median_pe_from_eps,
)
from src.pipeline import build_company_report
from src.research.report import ValuationTier
from src.research.verification import SourcedValue


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
    # WHY: live-verified real-world cross-provider variance for dividend yield (2-17% typical,
    # unlike the 2% default used for cleaner ratios like P/E), so this figure alone gets a wider
    # band; a genuinely large disagreement still correctly conflicts (see the next test).
    figs = {"dividend_yield_pct": [SourcedValue(5.0, "yfinance"), SourcedValue(5.8, "screener")]}
    r = build_company_report("X", figs)  # 16% apart -- would CONFLICT at the 2% default
    fig = next(f for f in r.figures if f.name == "dividend_yield_pct")
    assert fig.is_trustworthy
    assert fig.value == 5.4                          # median of the agreeing cluster


def test_dividend_yield_still_conflicts_when_wildly_different():
    # Real case (TCS, live-verified): yfinance 6.03% vs Screener 3.12%, a 48% gap, likely a
    # special-dividend timing difference between providers. Even the wider 25% band must not
    # force this to verify; a genuinely disputed figure stays withheld, not averaged.
    figs = {"dividend_yield_pct": [SourcedValue(6.03, "yfinance"), SourcedValue(3.12, "screener")]}
    r = build_company_report("X", figs)
    fig = next(f for f in r.figures if f.name == "dividend_yield_pct")
    assert not fig.is_trustworthy
