"""W5 unit-normalized numeric-exact-match metric (SPEC v4 §4 India golden set).

Over a small golden set of Indian annual-report/table fixtures with known ground-truth figures,
every extracted NumericRecord must match ground truth EXACTLY after scale normalization
(crore->1e7, lakh->1e5, million->1e6). A crore-vs-million mismatch MUST fail -- that is the point.
Deterministic, offline: fixtures are inline text ingested through the real DocumentStore.
"""
from src.eval.numeric_match import (
    GOLDEN_FIXTURES,
    GoldenFigure,
    GoldenFixture,
    numeric_exact_match,
)


def test_golden_set_matches_ground_truth_exactly():
    res = numeric_exact_match(GOLDEN_FIXTURES)
    assert res.total >= 6, "golden set should carry several known figures"
    assert res.exact_match_rate == 1.0, [m.detail for m in res.mismatches]
    assert res.matched == res.total


def test_crore_ground_truth_answered_as_million_fails_the_metric():
    # The core discipline: a value the source states in CRORE, if its ground truth is (wrongly)
    # expressed as MILLION, is a 10x error and MUST fail the metric -- proving it is not vacuous.
    fixture = GoldenFixture(
        name="crore_vs_million",
        text="Net profit for the year was Rs 73,670 crore.",
        company="ACME",
        figures=(
            # WRONG expectation: 73,670 * 1e6 (million) instead of the true 1e7 (crore).
            GoldenFigure(label="net_profit", raw_number="73,670",
                         expected_absolute=73670.0 * 1e6),
        ),
    )
    res = numeric_exact_match((fixture,))
    assert res.exact_match_rate == 0.0
    assert res.matched == 0
    mism = res.mismatches[0]
    assert mism.found_absolute == 73670.0 * 1e7  # the real (crore) magnitude, != the million claim


def test_lakh_scale_normalizes_exactly():
    fixture = GoldenFixture(
        name="lakh",
        text="Other income was Rs 4,50,000 lakh for the year.",
        company="ACME",
        figures=(GoldenFigure(label="other_income", raw_number="4,50,000",
                              expected_absolute=450000.0 * 1e5),),
    )
    res = numeric_exact_match((fixture,))
    assert res.exact_match_rate == 1.0


def test_period_keyed_figure_does_not_match_a_different_periods_value():
    # An FY2023 record must not satisfy an FY2024 golden figure of the same digits (period mixing
    # caught at the record layer). The FY2023 doc carries 500 crore; asking for FY2024/500 misses.
    fixture = GoldenFixture(
        name="period",
        text="Net profit for the year was Rs 500 crore.",
        company="ACME",
        fiscal_period="FY2023",
        figures=(GoldenFigure(label="np_wrong_period", raw_number="500",
                              expected_absolute=500.0 * 1e7, period="FY2024"),),
    )
    res = numeric_exact_match((fixture,))
    assert res.matched == 0            # no FY2024 record with these digits exists
    assert res.exact_match_rate == 0.0
