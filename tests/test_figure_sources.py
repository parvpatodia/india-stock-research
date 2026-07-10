from src.data.figure_sources import (
    FRAMEWORK_FIGURES,
    PERCENT_FIGURES,
    RATIO_FIGURES,
    FigureSource,
)
from src.pipeline import build_report_for_symbol, gather_figures
from src.research.report import Confidence, Leaning, QualityTier, ValuationTier
from src.research.verification import VerificationStatus

GOOD = {
    "current_pe": 18, "median_pe": 24, "operating_cash_flow": 88000, "net_profit": 79000,
    "total_debt": 30000, "equity": 200000, "ebit": 110000, "interest_expense": 5000,
    "promoter_pledge_pct": 0,
}


class FakeSource(FigureSource):
    def __init__(self, source_id: str, data: dict):
        self.source_id = source_id
        self._data = data

    def figures(self, symbol: str) -> dict[str, float | None]:
        return {name: self._data.get(name) for name in FRAMEWORK_FIGURES}


def test_ratio_and_percent_figures_are_disjoint_and_within_framework_figures():
    # WHY: the ONE shared classification every unit-label/format call site relies on (the Ask
    # tab's document, the expert-correction UI, ...) -- must never overlap or name a figure that
    # doesn't exist, or a label/format would silently apply to the wrong kind of number.
    assert RATIO_FIGURES.isdisjoint(PERCENT_FIGURES)
    assert RATIO_FIGURES <= set(FRAMEWORK_FIGURES)
    assert PERCENT_FIGURES <= set(FRAMEWORK_FIGURES)
    assert RATIO_FIGURES == {"current_pe", "median_pe"}
    assert PERCENT_FIGURES == {"promoter_pledge_pct", "dividend_yield_pct"}


def test_gather_merges_by_source():
    figs = gather_figures("X", [FakeSource("a", GOOD), FakeSource("b", GOOD)])
    assert len(figs["net_profit"]) == 2
    assert {sv.source_id for sv in figs["net_profit"]} == {"a", "b"}


def test_two_agreeing_sources_crossverify_high_confidence():
    r = build_report_for_symbol("X", [FakeSource("a", GOOD), FakeSource("b", GOOD)])
    net_profit = next(f for f in r.figures if f.name == "net_profit")
    assert net_profit.is_trustworthy
    assert r.verdict.valuation == ValuationTier.CHEAP
    assert r.verdict.leaning == Leaning.CONSTRUCTIVE
    assert r.verdict.confidence == Confidence.HIGH


def test_single_source_is_not_trustworthy_low_confidence():
    # WHY: honesty. One source cannot be cross-verified, so nothing is trusted and the verdict
    # stays low-confidence/unknown rather than pretending to be sure.
    r = build_report_for_symbol("X", [FakeSource("a", GOOD)])
    net_profit = next(f for f in r.figures if f.name == "net_profit")
    assert not net_profit.is_trustworthy
    assert r.verdict.valuation == ValuationTier.UNKNOWN
    assert r.verdict.quality == QualityTier.UNKNOWN
    assert r.verdict.confidence == Confidence.LOW


def test_two_disagreeing_sources_conflict():
    r = build_report_for_symbol("X", [FakeSource("a", GOOD), FakeSource("b", {**GOOD, "net_profit": 50000})])
    net_profit = next(f for f in r.figures if f.name == "net_profit")
    assert net_profit.status == VerificationStatus.CONFLICT
    assert not net_profit.is_trustworthy
