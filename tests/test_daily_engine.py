from src.analysis.daily_engine import candidate_from_report, refresh_today_if_stale
from src.analysis.sizing import Stance
from src.analysis.suggestions import RankedPick
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


def test_candidate_from_report_maps_signals():
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM)
    rep = Report(company="BLS", verdict=v,
                 insights=("Price: cheaper than usual.",
                           "Track record: sales have been growing 26% a year."))
    c = candidate_from_report("BLS", rep, held_value=0.0, total_value=100.0, cap_pct=0.25)
    assert c.stance == Stance.FAVORABLE
    assert c.quality_strong and c.valuation_cheap
    assert c.has_room is True                              # holds 0, cap 25 of 100 -> room
    assert c.trend_improving is True                       # 'growing' in insights
    assert c.reason.startswith("Price:")
