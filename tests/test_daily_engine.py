from src.analysis.daily_engine import (
    candidate_from_report,
    picks_to_rows,
    refresh_today_if_stale,
    research_and_rank,
)
from src.analysis.sizing import Stance
from src.analysis.suggestions import RankedPick
from src.data.figure_sources import FRAMEWORK_FIGURES, FigureSource
from src.data.sheets_backend import InMemoryGateway
from src.research.report import (
    Confidence,
    Leaning,
    QualityTier,
    Report,
    ValuationTier,
    Verdict,
)


def test_refresh_returns_cached_when_today_is_fresh():
    gw = InMemoryGateway({"Today": [
        {"date": "2026-07-08", "symbol": "BLS", "stance": "favorable", "score": "6", "reason": "x"}]})

    def _must_not_run(*a, **k):
        raise AssertionError("should not research when Today is already fresh")

    rows, refreshed = refresh_today_if_stale(gw, ["BLS"], {}, 100.0, 0.25,
                                             today="2026-07-08", researcher=_must_not_run)
    assert refreshed is False
    assert rows[0]["symbol"] == "BLS"


def test_refresh_researches_writes_and_pushes_when_stale():
    gw = InMemoryGateway()               # empty -> stale
    pushed = []

    def fake_researcher(symbols, value_by, total, cap):
        return [RankedPick("BLS", Stance.FAVORABLE, 6.0, "cheap vs history")]

    rows, refreshed = refresh_today_if_stale(
        gw, ["BLS"], {}, 100.0, 0.25, ntfy_topic="mytopic", today="2026-07-08",
        researcher=fake_researcher, pusher=lambda t, p: pushed.append((t, [x.symbol for x in p])))
    assert refreshed is True
    assert rows[0] == {"date": "2026-07-08", "symbol": "BLS", "stance": "favorable",
                       "score": "6", "reason": "cheap vs history"}
    assert gw.read("Today")[0]["symbol"] == "BLS"          # persisted
    assert pushed == [("mytopic", ["BLS"])]                # pushed


def test_refresh_force_recomputes_even_when_fresh():
    gw = InMemoryGateway({"Today": [
        {"date": "2026-07-08", "symbol": "OLD", "stance": "neutral", "score": "1", "reason": "x"}]})

    def fake_researcher(symbols, value_by, total, cap):
        return [RankedPick("NEW", Stance.FAVORABLE, 6.0, "fresh")]

    rows, refreshed = refresh_today_if_stale(gw, ["NEW"], {}, 100.0, 0.25, today="2026-07-08",
                                             force=True, researcher=fake_researcher,
                                             pusher=lambda t, p: None)
    assert refreshed is True                                   # force overrides the freshness check
    assert rows[0]["symbol"] == "NEW"


def test_candidate_from_report_maps_signals():
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM)
    rep = Report(company="BLS", verdict=v, trend_improving=True,
                 insights=("Price: cheaper than usual.",
                           "Track record: sales have been growing 26% a year."))
    c = candidate_from_report("BLS", rep, held_value=0.0, total_value=100.0, cap_pct=0.25)
    assert c.stance == Stance.FAVORABLE
    assert c.quality_strong and c.valuation_cheap
    assert c.has_room is True                              # holds 0, cap 25 of 100 -> room
    assert c.trend_improving is True                       # from report.trend_improving (structured)
    assert c.reason.startswith("Price:")
    assert 0.0 < c.strength <= 1.0                          # conviction populated from the verdict


def test_candidate_trend_reads_structured_flag_not_prose():
    # WHY (regression): the scoring flag must come from report.trend_improving, so a name whose
    # prose happens to say "growing" but whose structured signal is False is NOT counted.
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM)
    rep = Report(company="X", verdict=v, trend_improving=False,
                 insights=("Track record: sales have been growing 26% a year.",))
    c = candidate_from_report("X", rep, held_value=0.0, total_value=100.0, cap_pct=0.25)
    assert c.trend_improving is False


def test_picks_to_rows_floors_the_conviction_fraction():
    # WHY: the ordering fraction must not inflate the shown integer (6.9 -> "6", never "7").
    rows = picks_to_rows([RankedPick("BLS", Stance.FAVORABLE, 6.9, "x")], "2026-07-09")
    assert rows[0]["score"] == "6"


class _FakeSource(FigureSource):
    """A symbol with a real name resolves cleanly; anything else returns nothing from either
    source at all -- mirrors the live-verified PAGE-shaped case (a likely wrong/nonexistent
    ticker: e.g. Page Industries trades as PAGEIND, not PAGE)."""

    def __init__(self, source_id: str, known: dict):
        self.source_id = source_id
        self._known = known

    def figures(self, symbol: str) -> dict[str, float | None]:
        data = self._known.get(symbol, {})
        return {name: data.get(name) for name in FRAMEWORK_FIGURES}


def test_research_and_rank_logs_symbols_with_no_data_from_any_source():
    # WHY (operator visibility, real money): this daily engine is Parv's own background job with
    # no interactive UI -- a typo'd watchlist symbol (the PAGE-shaped case) already can't produce
    # a wrong PICK (INSUFFICIENT_DATA is excluded by rank_picks), but without a diagnostic it
    # would be silently dropped forever with no signal in the log Parv actually reviews.
    known = {"GOOD": {"current_pe": 15, "median_pe": 20, "net_profit": 100 * 1e7,
                      "operating_cash_flow": 90 * 1e7, "total_debt": 20 * 1e7,
                      "equity": 100 * 1e7, "ebit": 30 * 1e7, "interest_expense": 3 * 1e7}}

    def sources_factory():
        return [_FakeSource("a", known), _FakeSource("b", known)]

    skipped: list[str] = []
    research_and_rank(["GOOD", "WRONG"], {}, 100.0, 0.25,
                      sources_factory=sources_factory, throttle_seconds=0, skipped=skipped)
    assert len(skipped) == 1
    assert "WRONG" in skipped[0]
    assert "ticker" in skipped[0].lower() or "no data" in skipped[0].lower()


def test_research_and_rank_skipped_is_optional_and_backward_compatible():
    # Existing callers that don't pass `skipped` see no change in behavior.
    picks = research_and_rank(["WRONG"], {}, 100.0, 0.25,
                              sources_factory=lambda: [_FakeSource("a", {})], throttle_seconds=0)
    assert picks == []
