from src.analysis.daily_engine import (
    _ntfy_body,
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


def test_candidate_reason_is_the_first_sentence_not_a_paragraph():
    # WHY (real money, shortlist readability): the daily "Today" shortlist a parent scans shows one
    # SYM (stance) — reason bullet per name. reason = report.insights[0], which can be MULTI-sentence
    # -- the Price insight now appends a de-rating-trap caveat for a cheap read (right for the detailed
    # Research view) that turned the bullet into a ~76-word paragraph. Keep the compact list scannable:
    # the reason is the core FIRST sentence; the full insight (caveats and all) still shows in Research.
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM)
    long_insight = ("Price: you pay about ₹15 for every ₹1 of yearly profit (P/E 15) — cheaper than "
                    "usual versus its own history (about 50% of its normal). A below-usual multiple "
                    "is a cue to research WHY, not a bargain on its own.")
    rep = Report(company="X", verdict=v, insights=(long_insight,))
    c = candidate_from_report("X", rep, held_value=0.0, total_value=100.0, cap_pct=0.25)
    assert "cheaper than usual" in c.reason               # the core 'why' is kept
    assert c.reason.endswith("of its normal).")           # ...just the first sentence
    assert "research WHY" not in c.reason                 # the trailing caveat is dropped for the bullet


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


def test_ntfy_body_frames_as_research_not_a_buy_call():
    # WHY (real money, honesty; hard "never a buy/sell call" invariant): the daily push is a
    # NOTIFICATION of the suggestions feature, and a parent acts on a phone glance. The app frames
    # suggestions in-app as "evidence leans favorable ... not a buy or sell call" (STANCE_CAVEAT), but
    # the notification said "Today's long-term PICKS" with no caveat -- "picks" reads as a buy tip.
    # The notification must carry the same non-advice framing as the app, and not call them "picks".
    picks = [RankedPick("RELIANCE", Stance.FAVORABLE, 5.0, "cheap"),
             RankedPick("TCS", Stance.NEUTRAL, 4.0, "fair")]
    body = _ntfy_body(picks)
    assert "RELIANCE - favorable" in body and "TCS - neutral" in body   # the names + stance still shown
    assert "not buy/sell advice" in body.lower()                        # carries the non-advice framing
    assert "picks" not in body.lower()                                  # not framed as recommended 'picks'


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


class _RaisingSource(FigureSource):
    """A source whose figures() call raises for one specific symbol -- mirrors a real network
    blip (timeout, rate limit, malformed page) on a single watchlist name, distinct from the
    no_data_found case (which is a clean, empty result, not an error)."""

    def __init__(self, source_id: str, raise_for: str):
        self.source_id = source_id
        self._raise_for = raise_for

    def figures(self, symbol: str) -> dict[str, float | None]:
        if symbol == self._raise_for:
            raise RuntimeError("network timeout")
        return {name: None for name in FRAMEWORK_FIGURES}


def test_research_and_rank_logs_symbols_that_raise_during_fetch():
    # WHY (operator visibility): a network blip on ONE symbol must not silently vanish from the
    # daily log -- it should show up in `skipped` with the actual exception message, so Parv can
    # tell "transient network issue, retry tomorrow" apart from "check your watchlist spelling"
    # (the no_data_found case) without digging through a stack trace he'd never see (this job
    # runs headless via launchd).
    def sources_factory():
        return [_RaisingSource("a", raise_for="BROKEN"), _RaisingSource("b", raise_for="BROKEN")]

    skipped: list[str] = []
    picks = research_and_rank(["BROKEN"], {}, 100.0, 0.25,
                              sources_factory=sources_factory, throttle_seconds=0, skipped=skipped)
    assert picks == []
    assert len(skipped) == 1
    assert "BROKEN" in skipped[0]
    assert "fetch failed" in skipped[0]
    assert "network timeout" in skipped[0]


def test_daily_notification_icon_is_not_a_buy_or_market_direction_signal():
    # WHY (real money, the hard "never a buy/sell call" invariant): the daily push is "acted on at a
    # glance" (see _ntfy_body), so the notification ICON is part of the framing, not decoration. A
    # market-direction / hype emoji -- an up- or down-trend chart, a rocket, a moneybag -- reads as a
    # bullish buy tip and contradicts the body's own explicit "to research, NOT buy/sell advice". The
    # icon must stay neutral / research-oriented.
    from src.analysis.daily_engine import NTFY_TAG
    hype_or_directional = {
        "chart_with_upwards_trend", "chart_with_downwards_trend", "rocket", "moneybag",
        "money_with_wings", "fire", "rotating_light", "boom", "100",
    }
    assert NTFY_TAG not in hype_or_directional
