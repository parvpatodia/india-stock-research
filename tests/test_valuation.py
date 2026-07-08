from src.analysis.valuation import median_pe_from_annuals
from src.pipeline import build_company_report
from src.research.report import ValuationTier
from src.research.verification import SourcedValue


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
