"""Typed numeric records: extract each figure from chunk text as a structured, provenance-
tagged record and normalize its scale to an absolute value deterministically. Fixtures mirror
real Indian annual-report/table text (crore/lakh units). Offline; no network."""
import math

import pytest

from src.research.numeric_records import (
    SCALE_FACTORS,
    NumericRecord,
    extract_records,
    find_record,
    number_key,
)


# --- scale normalization (the seam W3 builds on) ---

def test_crore_figure_normalizes_to_absolute():
    recs = extract_records("Net profit was Rs 73,670 crore for the year.")
    assert len(recs) == 1
    r = recs[0]
    assert r.value == 73670.0
    assert r.scale == "crore"
    assert r.unit == "rupees"
    assert r.currency == "INR"
    assert r.absolute_value == 73670.0 * SCALE_FACTORS["crore"]
    assert r.absolute_value == 73670.0 * 1e7


def test_lakh_and_million_scales():
    recs = extract_records("Other income Rs 4,50,000 lakh and R&D spend 120 million.")
    by_scale = {r.scale: r for r in recs}
    assert by_scale["lakh"].absolute_value == 450000.0 * 1e5
    assert by_scale["million"].absolute_value == 120.0 * 1e6


def test_cr_abbreviation_is_crore():
    recs = extract_records("Total debt stood at 3,02,000 cr.")
    assert recs[0].scale == "crore"
    assert recs[0].absolute_value == 302000.0 * 1e7


def test_percent_is_unit_percent_scale_none():
    recs = extract_records("ROE improved to 22.5% this year.")
    r = recs[0]
    assert r.unit == "percent"
    assert r.scale == "none"
    assert r.value == 22.5
    assert r.absolute_value == 22.5  # a percent is never multiplied by a rupee scale


def test_ratio_x_suffix():
    recs = extract_records("The stock trades at 18.2x trailing earnings.")
    ratios = [r for r in recs if r.unit == "ratio"]
    assert len(ratios) == 1
    assert ratios[0].value == 18.2
    assert ratios[0].scale == "none"


def test_basis_points():
    recs = extract_records("Net interest margin expanded 150 bps.")
    assert recs[0].unit == "bps"
    assert recs[0].value == 150.0
    assert recs[0].scale == "none"


def test_parenthesised_number_is_negative():
    recs = extract_records("Exceptional items (1,234) crore dragged the profit down.")
    r = recs[0]
    assert r.value == -1234.0
    assert r.absolute_value == -1234.0 * 1e7


def test_currency_prefixed_number_without_scale_is_absolute_rupees():
    recs = extract_records("Cash on hand was Rs 5,000 at year end.")
    r = recs[0]
    assert r.unit == "rupees"
    assert r.scale == "none"
    assert r.value == 5000.0
    assert r.absolute_value == 5000.0


# --- provenance ---

def test_provenance_is_attached():
    recs = extract_records(
        "Revenue Rs 9,00,000 crore.", period="FY2024", company="RELIANCE",
        source_doc="annual_report", locator="annual_report#3")
    r = recs[0]
    assert r.period == "FY2024"
    assert r.company == "RELIANCE"
    assert r.source_doc == "annual_report"
    assert r.locator == "annual_report#3"


# --- caption-scale (table cells with no inline unit inherit the table's declared scale) ---

def test_caption_scale_types_bare_table_cells():
    recs = extract_records("Revenue from operations   9,00,000   7,92,000",
                           default_scale="crore", currency="INR")
    vals = sorted(r.absolute_value for r in recs)
    assert vals == [792000.0 * 1e7, 900000.0 * 1e7]
    assert all(r.scale == "crore" and r.unit == "rupees" for r in recs)


def test_bare_year_not_mined_even_under_caption_scale():
    recs = extract_records("For the year 2024 the figure was 12,345",
                           default_scale="crore", currency="INR")
    keys = {number_key(r.raw_string) for r in recs}
    assert "12345" in keys
    assert "2024" not in keys  # a 4-digit year is metadata, not a figure


def test_fiscal_year_range_not_mined_as_negative():
    # "2024-25" is a fiscal-year range, not a figure and not a "-25 crore" negative
    recs = extract_records("In 2024-25 crore projects were commissioned.",
                           default_scale="crore", currency="INR")
    assert recs == []


def test_fy_and_iso_date_metadata_not_mined():
    recs = extract_records("As on 2026-03-31 (FY2024) net profit Rs 73,670 crore.")
    keys = {number_key(r.raw_string) for r in recs}
    assert keys == {"73670"}


def test_no_caption_bare_numbers_are_skipped():
    # without an inline unit and without a chunk scale, a bare number is not a figure
    assert extract_records("The annual meeting had 250 attendees.") == []


# --- bad-input regression: one hard rejection per class (money-math lesson) ---

def test_nan_value_rejected_hard():
    with pytest.raises(ValueError):
        NumericRecord(value=float("nan"), raw_string="x", unit="rupees", scale="crore",
                      currency=None, period=None, company=None, source_doc=None, locator=None)


def test_inf_value_rejected_hard():
    with pytest.raises(ValueError):
        NumericRecord(value=float("inf"), raw_string="x", unit="rupees", scale="crore",
                      currency=None, period=None, company=None, source_doc=None, locator=None)


def test_unknown_scale_rejected():
    with pytest.raises(ValueError):
        NumericRecord(value=1.0, raw_string="x", unit="rupees", scale="zillion",
                      currency=None, period=None, company=None, source_doc=None, locator=None)


def test_overflow_on_normalization_rejected():
    # a value that is finite but overflows to inf once scaled must not become a record
    with pytest.raises(ValueError):
        NumericRecord(value=1e308, raw_string="x", unit="rupees", scale="crore",
                      currency=None, period=None, company=None, source_doc=None, locator=None)


def test_garbage_number_text_yields_no_records():
    assert extract_records("Rs .. -- crore of nothing; price Rs.") == []


def test_empty_or_blank_text_yields_no_records():
    assert extract_records("") == []
    assert extract_records("   \n  \t ") == []


# --- find_record: cite a number to its exact record; no record -> nothing ---

def test_find_record_matches_by_numeric_key_ignoring_commas():
    recs = extract_records("Net profit Rs 73,670 crore.")
    assert find_record("73670", recs) is recs[0]
    assert find_record("73,670", recs) is recs[0]
    assert find_record("999", recs) is None


def test_find_record_resolves_large_absolute_rupee_figure():
    # a 12-digit absolute rupee figure must resolve by its as-written digits, with no scientific-
    # notation round-trip mangling the key
    recs = extract_records("Net profit was Rs 958,000,000,000 for the year.")
    r = find_record("958000000000", recs)
    assert r is not None
    assert r.value == 958000000000.0
    assert find_record("958,000,000,000", recs) is r


def test_find_record_keeps_decimal_distinct():
    recs = extract_records("Margin was 12.34% and revenue was Rs 1,234 crore.")
    # 12.34 must not resolve to the 1,234 record 100x its size (decimal preserved in the key)
    pct = find_record("12.34", recs)
    rupees = find_record("1234", recs)
    assert pct is not None and pct.unit == "percent"
    assert rupees is not None and rupees.unit == "rupees"
    assert pct is not rupees
