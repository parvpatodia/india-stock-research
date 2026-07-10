import pandas as pd

from src.data.figure_sources import (
    FRAMEWORK_FIGURES,
    PERCENT_FIGURES,
    RATIO_FIGURES,
    FigureSource,
    YFinanceFigureSource,
    format_figure_value,
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


def test_format_figure_value_by_unit():
    # WHY (real money): a bare '25.00' is genuinely ambiguous between a 25% pledge and Rs.25.
    # Every display site (Research tab table, PDF export, Ask tab document) must agree on this.
    assert format_figure_value("current_pe", 18.2) == "18.2x"
    assert format_figure_value("median_pe", 24.0) == "24.0x"
    assert format_figure_value("promoter_pledge_pct", 25.0) == "25.0%"
    assert format_figure_value("dividend_yield_pct", 0.47) == "0.5%"
    assert format_figure_value("net_profit", 79000.0) == "₹79,000.00"


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


class _FakeTicker:
    def __init__(self, income_stmt):
        self.info = {}
        self.income_stmt = income_stmt
        self.balance_sheet = pd.DataFrame()
        self.cashflow = pd.DataFrame()


def test_ebit_uses_pretax_and_interest_from_the_same_fiscal_period(monkeypatch):
    # WHY (real money): Pretax Income and Interest Expense are looked up independently via
    # _latest(), each returning ITS OWN most-recent non-NaN period. If one row has a gap the
    # other doesn't (yfinance income statements are not always fully populated for the latest
    # period across every row), the two "latest" values can come from DIFFERENT fiscal years --
    # e.g. this year's Interest Expense (100) plus LAST year's Pretax Income (900), producing an
    # EBIT of 1000 that never existed in any single real fiscal year. EBIT must instead use the
    # most recent period where BOTH rows actually have data together (here, 2025: 900 + 90 =
    # 990), matching figures_by_year()'s existing period-aligned pairing for this same figure.
    cols = [pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")]
    income = pd.DataFrame(
        {cols[0]: [None, 100.0], cols[1]: [900.0, 90.0]},
        index=["Pretax Income", "Interest Expense"],
    )
    fake_yf = type("_FakeYF", (), {"Ticker": staticmethod(lambda _sym: _FakeTicker(income))})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    figs = YFinanceFigureSource().figures("X")

    assert figs["ebit"] == 990.0
    # Interest Expense standalone must stay the freshest available value (2026's 100), unaffected
    # by which period the EBIT pairing had to fall back to.
    assert figs["interest_expense"] == 100.0


def test_ebit_pairing_falls_back_to_a_synonym_row_when_the_first_is_entirely_empty(monkeypatch):
    # WHY (regression, found by adversarial review): the first EBIT fix locked onto the FIRST
    # candidate label present in the statement even if that row is entirely NaN, never trying
    # the next synonym -- unlike _latest(), which tries every candidate row and only gives up if
    # ALL of them are empty. A statement whose "Pretax Income" row is entirely empty but whose
    # "Income Before Tax" synonym row has real, period-aligned data must still produce an EBIT,
    # not silently fall through to the cruder Operating Income proxy.
    cols = [pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")]
    income = pd.DataFrame(
        {cols[0]: [None, None, 100.0], cols[1]: [None, 900.0, 90.0]},
        index=["Pretax Income", "Income Before Tax", "Interest Expense"],
    )
    fake_yf = type("_FakeYF", (), {"Ticker": staticmethod(lambda _sym: _FakeTicker(income))})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    figs = YFinanceFigureSource().figures("X")

    assert figs["ebit"] == 990.0
