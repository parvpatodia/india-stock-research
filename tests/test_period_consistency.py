"""The period-awareness guard (H2, SPEC v4 §2.2 "metadata filtering ... stops FY23/FY24 mixing").

numbers_grounded, numbers_record_backed and numbers_unit_consistent all match a figure by its
DIGITS (and, for the last, its scale word) -- none of them look at the fiscal PERIOD. So when the
question targets a specific year, a model can surface a DIFFERENT year's figure (FY2023's 500 crore
as the FY2024 answer) and it resolves to a real FY2023 record and passes every prior numeric check.
numbers_period_consistent closes that hole: a material number must be backed by a same-digit record
whose period MATCHES the target or is UNTAGGED. Conservative like the other numeric guards -- it
only ever DOWNGRADES, an untagged record (unknown period != wrong period) is never excluded, and
with no target period it is inert.
"""
from src.research.grounded_analyst import numbers_period_consistent
from src.research.numeric_records import extract_records


def _rec(text: str, period: str | None):
    return extract_records(text, period=period)


def test_no_target_period_is_inert():
    # no fiscal year in the question -> the guard must not touch anything (normal questions unchanged)
    records = _rec("Net profit for the year was Rs 500 crore.", "FY2023")
    assert numbers_period_consistent("Net profit was Rs 500 crore.", records, None) is True


def test_figure_from_a_different_explicit_period_is_flagged():
    # the only same-digit record is tagged FY2023, but the question targets FY2024 -> the figure
    # cannot be presented as the FY2024 answer
    records = _rec("Net profit for the year was Rs 500 crore.", "FY2023")
    assert numbers_period_consistent("Net profit was Rs 500 crore.", records, "FY2024") is False


def test_figure_from_the_target_period_is_consistent():
    records = _rec("Net profit for the year was Rs 620 crore.", "FY2024")
    assert numbers_period_consistent("Net profit was Rs 620 crore.", records, "FY2024") is True


def test_untagged_record_is_never_excluded():
    # conservatism (real money): an UNTAGGED record (unknown period) must not be withheld -- unknown
    # period is not a wrong period, and blinding the tool to untagged data is worse than a caution
    records = _rec("Net profit for the year was Rs 500 crore.", None)
    assert numbers_period_consistent("Net profit was Rs 500 crore.", records, "FY2024") is True


def test_a_target_period_record_present_prevents_a_false_positive():
    # the same digits exist for BOTH FY2023 and FY2024; a figure that matches the FY2024 record must
    # stay consistent (no false downgrade just because a same-digit FY2023 record also exists)
    records = (_rec("Rs 500 crore.", "FY2023") + _rec("Rs 500 crore.", "FY2024"))
    assert numbers_period_consistent("Net profit was Rs 500 crore.", records, "FY2024") is True


def test_no_matching_digit_record_is_consistent():
    # nothing in the corpus shares the claim's digits -> record-backed handles presence, not this guard
    records = _rec("Net profit for the year was Rs 500 crore.", "FY2023")
    assert numbers_period_consistent("Revenue was Rs 900 crore.", records, "FY2024") is True


def test_no_material_number_is_consistent():
    records = _rec("Net profit for the year was Rs 500 crore.", "FY2023")
    assert numbers_period_consistent("The company grew across segments.", records, "FY2024") is True


def test_unparseable_target_period_is_inert():
    # a target with no 4-digit year to compare against must not downgrade everything (fail safe open)
    records = _rec("Net profit for the year was Rs 500 crore.", "FY2023")
    assert numbers_period_consistent("Net profit was Rs 500 crore.", records, "FY") is True
