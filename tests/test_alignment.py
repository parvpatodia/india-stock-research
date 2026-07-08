from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource
from src.pipeline import (
    _latest_common_year,
    build_report_for_symbol,
    gather_aligned_figures,
)


class FakeYearSource(FigureSource):
    def __init__(self, source_id, series, scalar=None):
        self.source_id = source_id
        self._series = series
        self._scalar = scalar or {}

    def figures(self, symbol):
        return {name: self._scalar.get(name) for name in FRAMEWORK_FIGURES}

    def figures_by_year(self, symbol):
        return self._series


def test_latest_common_year_prefers_latest_common():
    assert _latest_common_year({"a": {2024: 1, 2023: 2}, "b": {2025: 3, 2024: 4}}) == 2024


def test_latest_common_year_falls_back_to_latest_when_none_common():
    assert _latest_common_year({"a": {2023: 1}, "b": {2025: 2}}) == 2025


def test_latest_common_year_empty():
    assert _latest_common_year({}) is None


def test_alignment_fixes_period_mismatch():
    # A's latest is FY2024; B's latest is FY2025. Their latest values differ, but FY2024 agrees.
    a = FakeYearSource("a", {"net_profit": {2024: 80000, 2023: 75000}})
    b = FakeYearSource("b", {"net_profit": {2025: 95000, 2024: 80100}})
    merged = gather_aligned_figures("X", [a, b])
    assert sorted(sv.value for sv in merged["net_profit"]) == [80000, 80100]  # both took FY2024
    net_profit = next(f for f in build_report_for_symbol("X", [a, b]).figures
                      if f.name == "net_profit")
    assert net_profit.is_trustworthy  # cross-verified at the common year (was a conflict before)


def test_point_figures_compared_as_is():
    a = FakeYearSource("a", {"net_profit": {2024: 80000}}, scalar={"current_pe": 18.0})
    b = FakeYearSource("b", {"net_profit": {2024: 80100}}, scalar={"current_pe": 18.1})
    merged = gather_aligned_figures("X", [a, b])
    assert len(merged["current_pe"]) == 2
    assert len(merged["net_profit"]) == 2


def test_scalar_only_source_fills_when_under_two_aligned():
    year_src = FakeYearSource("yf", {"net_profit": {2024: 80000}})
    scalar_src = FakeYearSource("ar", {}, scalar={"net_profit": 80200})  # no series, scalar only
    merged = gather_aligned_figures("X", [year_src, scalar_src])
    assert {sv.source_id for sv in merged["net_profit"]} == {"yf", "ar"}
