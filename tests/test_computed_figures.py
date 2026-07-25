"""Compute-don't-generate seam (SPEC v4 §2 decision #1 / §3).

Every derived number (YoY growth, a margin, a CAGR) is computed in deterministic Python from
TYPED NumericRecords and handed to the LLM only to phrase -- the model never divides two raw
numbers itself. These tests pin the arithmetic, the reuse of analysis.trends.cagr, the typed-
record seam, and the money-math guards (a non-finite input / divide-through-zero withholds,
never fabricates)."""
import math

from src.analysis.trends import cagr
from src.research.computed_figures import (
    ComputedFigure,
    cagr_from_records,
    cagr_pct,
    growth_between,
    margin_between,
    ratio_pct,
    yoy_growth_pct,
)
from src.research.numeric_records import PERCENT, NumericRecord, extract_records


# --- pure primitives ---------------------------------------------------------------------------

def test_yoy_growth_pct_basic():
    assert yoy_growth_pct(100.0, 112.0) == 12.0
    assert yoy_growth_pct(100.0, 88.0) == -12.0


def test_yoy_growth_pct_withholds_on_zero_base_or_non_finite():
    # growth off a (near-)zero base is undefined -- never divide by it
    assert yoy_growth_pct(0.0, 100.0) is None
    assert yoy_growth_pct(1e-12, 100.0) is None
    assert yoy_growth_pct(float("nan"), 100.0) is None
    assert yoy_growth_pct(100.0, float("inf")) is None


def test_ratio_pct_basic_and_guards():
    assert ratio_pct(20.0, 200.0) == 10.0            # a 10% margin
    assert ratio_pct(50.0, 0.0) is None              # margin on zero sales is undefined
    assert ratio_pct(float("inf"), 200.0) is None


def test_cagr_pct_delegates_to_trends_cagr():
    series = {2021: 100.0, 2022: 110.0, 2023: 121.0}   # endpoints 100->121 over 2 yrs = 10%/yr
    # reuse: the seam's CAGR must be byte-identical to the Research tab's tested trend math
    assert cagr_pct(series) == cagr(series)
    rate, span = cagr_pct(series)
    assert round(rate, 4) == 10.0 and span == 2


def test_cagr_pct_withholds_like_trends_cagr():
    assert cagr_pct({2023: 100.0}) is None           # <3 years
    assert cagr_pct({2021: -1.0, 2022: 2.0, 2023: 3.0}) is None  # non-positive endpoint


# --- typed-record adapters ---------------------------------------------------------------------

def test_growth_between_records_uses_absolute_normalized_magnitudes():
    prev = extract_records("Net profit was Rs 66,700 crore.")[0]
    curr = extract_records("Net profit was Rs 73,670 crore.")[0]
    fig = growth_between(prev, curr)
    assert isinstance(fig, ComputedFigure)
    assert fig.unit == PERCENT
    assert round(fig.value, 2) == round((73670 - 66700) / 66700 * 100, 2)
    # provenance: the exact absolute magnitudes it computed from, for the LLM to cite, not re-derive
    assert fig.inputs == (66700.0 * 1e7, 73670.0 * 1e7)


def test_growth_between_refuses_mismatched_units():
    rupees = extract_records("Net profit was Rs 66,700 crore.")[0]
    percent = extract_records("ROE was 22.5%.")[0]
    # never compute growth of a rupee figure against a percent
    assert growth_between(rupees, percent) is None


def test_margin_between_records_is_part_over_whole_percent():
    profit = extract_records("Net profit was Rs 73,670 crore.")[0]
    revenue = extract_records("Revenue was Rs 9,00,000 crore.")[0]
    fig = margin_between(profit, revenue)
    assert fig is not None and fig.unit == PERCENT
    assert round(fig.value, 2) == round(73670 / 900000 * 100, 2)


def test_margin_between_requires_two_rupee_figures():
    profit = extract_records("Net profit was Rs 73,670 crore.")[0]
    percent = extract_records("ROE was 22.5%.")[0]
    assert margin_between(profit, percent) is None


def test_cagr_from_records_reuses_trends_cagr_across_fiscal_years():
    records = [
        NumericRecord(100.0, "100 crore", "rupees", "crore", "INR", "FY2021", None, "ar", "ar#0"),
        NumericRecord(110.0, "110 crore", "rupees", "crore", "INR", "FY2022", None, "ar", "ar#1"),
        NumericRecord(121.0, "121 crore", "rupees", "crore", "INR", "FY2023", None, "ar", "ar#2"),
    ]
    fig = cagr_from_records(records)
    assert fig is not None
    assert round(fig.value, 4) == 10.0                 # endpoints 100->121 over 2 yrs = 10%/yr
    # withholds when there is no multi-year history
    assert cagr_from_records(records[:1]) is None


def test_computed_figure_value_is_always_finite():
    # the seam must never hand the model a NaN/inf to phrase as a real figure
    for fig in (growth_between(extract_records("Rs 100 crore")[0],
                               extract_records("Rs 110 crore")[0]),
                margin_between(extract_records("Rs 20 crore")[0],
                               extract_records("Rs 200 crore")[0])):
        assert fig is not None and math.isfinite(fig.value)
