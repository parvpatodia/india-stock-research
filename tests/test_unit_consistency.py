"""The unit-trap guard: a claim's explicit rupee scale (crore/lakh/million) must agree with the
typed record carrying the same digits. numbers_grounded and numbers_record_backed both match on
DIGITS ONLY (the scale word is dropped), so '500 crore' in the source and '500 lakh' in the claim
-- a 100x error -- pass every prior numeric check. This guard closes that hole. Conservative: it
only fires on an EXPLICIT scale conflict, so a legitimately-scaled quote is never downgraded.
"""
from src.research.grounded_analyst import numbers_unit_consistent
from src.research.numeric_records import extract_records


def _records(text: str):
    return extract_records(text)


def test_same_digits_conflicting_scale_is_flagged():
    records = _records("Net profit for the year was Rs 500 crore.")
    # a claim that restates the SAME digits as lakh (100x smaller) is a unit trap
    assert numbers_unit_consistent("Net profit was Rs 500 lakh.", records) is False


def test_matching_scale_is_consistent():
    records = _records("Net profit for the year was Rs 500 crore.")
    assert numbers_unit_consistent("Net profit was Rs 500 crore.", records) is True


def test_bare_number_without_a_scale_word_is_not_a_unit_trap():
    # a claim stating a bare "500" carries no scale to conflict; numbers_record_backed covers presence
    records = _records("Net profit for the year was Rs 500 crore.")
    assert numbers_unit_consistent("Net profit was 500.", records) is True


def test_percent_figure_is_never_a_rupee_unit_trap():
    records = _records("ROE was 22.5% and net profit Rs 500 crore.")
    assert numbers_unit_consistent("ROE was 22.5%.", records) is True


def test_no_matching_digit_record_is_consistent():
    # nothing in the corpus shares the claim's digits -> record-backed handles it, not this guard
    records = _records("Net profit for the year was Rs 500 crore.")
    assert numbers_unit_consistent("Revenue was Rs 900 crore.", records) is True


def test_a_matching_scale_record_present_prevents_a_false_positive():
    # both a crore AND a lakh record with the same digits exist; a crore claim agrees with one of
    # them, so it must stay consistent (no false downgrade)
    records = _records("Segment A Rs 500 crore. Prior note Rs 500 lakh.")
    assert numbers_unit_consistent("Segment A was Rs 500 crore.", records) is True
