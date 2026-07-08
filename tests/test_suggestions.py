from src.analysis.sizing import Stance
from src.analysis.suggestions import Candidate, rank_picks, score_candidate


def _c(sym, stance, strong=False, cheap=False, room=True, trend=False):
    return Candidate(sym, stance, quality_strong=strong, valuation_cheap=cheap,
                     has_room=room, trend_improving=trend)


def test_score_sums_signals_and_excludes_ineligible():
    best = _c("A", Stance.FAVORABLE, strong=True, cheap=True, room=True, trend=True)
    assert score_candidate(best) == 6.0                       # 2 + strong + cheap + room + trend
    assert score_candidate(_c("B", Stance.NEUTRAL, room=False)) == 1.0   # base only
    assert score_candidate(_c("C", Stance.UNFAVORABLE)) == 0.0
    assert score_candidate(_c("D", Stance.INSUFFICIENT_DATA)) == 0.0


def test_rank_excludes_unfavorable_insufficient_and_no_room():
    cands = [
        _c("FAV", Stance.FAVORABLE, strong=True),
        _c("UNFAV", Stance.UNFAVORABLE, strong=True),
        _c("INSUF", Stance.INSUFFICIENT_DATA),
        _c("NOROOM", Stance.FAVORABLE, strong=True, room=False),
    ]
    syms = [p.symbol for p in rank_picks(cands)]
    assert syms == ["FAV"]                                    # others excluded


def test_rank_orders_best_first_then_symbol():
    cands = [                                                 # all have room (default) -> +1 each
        _c("MID", Stance.NEUTRAL, strong=True),               # 1 + strong + room = 3
        _c("TOP", Stance.FAVORABLE, strong=True, cheap=True), # 2 + strong + cheap + room = 5
        _c("LOW", Stance.NEUTRAL),                            # 1 + room = 2
        _c("TIE", Stance.NEUTRAL, strong=True),               # 3 (ties MID -> alpha order)
    ]
    ranked = rank_picks(cands)
    assert [p.symbol for p in ranked] == ["TOP", "MID", "TIE", "LOW"]
    assert ranked[0].score == 5.0
