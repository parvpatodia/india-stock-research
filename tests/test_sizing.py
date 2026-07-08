from src.analysis.sizing import (
    AllocationCandidate,
    Stance,
    position_sizing,
    stance_from_verdict,
    suggest_allocation,
)
from src.research.report import (
    Confidence,
    Leaning,
    QualityTier,
    ValuationTier,
    Verdict,
)


def _verdict(leaning: Leaning, confidence: Confidence) -> Verdict:
    return Verdict(valuation=ValuationTier.FAIR, quality=QualityTier.STRONG,
                   leaning=leaning, confidence=confidence)


# --- stance_from_verdict ---

def test_stance_maps_leaning_when_confident():
    assert stance_from_verdict(_verdict(Leaning.CONSTRUCTIVE, Confidence.MEDIUM)) == Stance.FAVORABLE
    assert stance_from_verdict(_verdict(Leaning.CAUTIOUS, Confidence.HIGH)) == Stance.UNFAVORABLE
    assert stance_from_verdict(_verdict(Leaning.NEUTRAL, Confidence.MEDIUM)) == Stance.NEUTRAL


def test_stance_insufficient_when_unknown_or_low_confidence():
    # the safety catch: unknown leaning, or low confidence, must not read as a confident lean.
    assert stance_from_verdict(_verdict(Leaning.UNKNOWN, Confidence.MEDIUM)) == Stance.INSUFFICIENT_DATA
    assert stance_from_verdict(_verdict(Leaning.CONSTRUCTIVE, Confidence.LOW)) == Stance.INSUFFICIENT_DATA
    assert stance_from_verdict(None) == Stance.INSUFFICIENT_DATA


# --- position_sizing (transparent cap math) ---

def test_position_sizing_room_and_over_cap():
    s = position_sizing(current_value=0.0, portfolio_value=100.0, cap_pct=0.25)
    assert s.cap_value == 25.0 and s.room == 25.0 and s.headroom == 25.0 and s.over_cap is False

    over = position_sizing(current_value=40.0, portfolio_value=100.0, cap_pct=0.25)
    assert over.room == -15.0 and over.over_cap is True and over.headroom == 0.0


# --- suggest_allocation ---

def test_allocation_favorable_first_then_neutral_within_caps():
    # portfolio 100 + amount 30 => base 130, cap 25% => 32.5 ceiling per name.
    cands = [
        AllocationCandidate("N", Stance.NEUTRAL, current_value=0.0),     # room 32.5
        AllocationCandidate("F", Stance.FAVORABLE, current_value=20.0),  # room 12.5
    ]
    plan = suggest_allocation(30.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert [(a.symbol, a.amount) for a in plan.allocations] == [("F", 12.5), ("N", 17.5)]
    assert plan.uninvested == 0.0
    assert plan.invested == 30.0


def test_allocation_excludes_unfavorable_and_insufficient():
    cands = [
        AllocationCandidate("A", Stance.FAVORABLE, 0.0),
        AllocationCandidate("B", Stance.UNFAVORABLE, 0.0),
        AllocationCandidate("C", Stance.INSUFFICIENT_DATA, 0.0),
    ]
    plan = suggest_allocation(30.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert [a.symbol for a in plan.allocations] == ["A"]   # only the favorable name


def test_allocation_reports_uninvested_when_caps_bind():
    # one favorable name, base 200, cap 25% => 50 ceiling; 100 asked -> only 50 placeable.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.invested == 50.0
    assert plan.uninvested == 50.0
    assert any("per-stock cap" in n for n in plan.notes)


def test_allocation_over_cap_name_gets_nothing():
    cands = [AllocationCandidate("A", Stance.FAVORABLE, current_value=60.0)]  # already > cap
    plan = suggest_allocation(30.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.allocations == ()
    assert plan.uninvested == 30.0


def test_allocation_no_eligible_names_suggests_nothing():
    cands = [AllocationCandidate("A", Stance.UNFAVORABLE, 0.0)]
    plan = suggest_allocation(50.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.allocations == ()
    assert plan.uninvested == 50.0
    assert any("approved" in n.lower() for n in plan.notes)


def test_allocation_zero_amount_is_a_noop():
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(0.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.allocations == () and plan.uninvested == 0.0


def test_allocation_spreads_across_two_favorable_names():
    cands = [
        AllocationCandidate("A", Stance.FAVORABLE, 0.0),
        AllocationCandidate("B", Stance.FAVORABLE, 0.0),
    ]
    # base 200, cap 25% => 50 each; 100 asked -> 50 + 50, fully placed.
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert sorted(a.amount for a in plan.allocations) == [50.0, 50.0]
    assert plan.uninvested == 0.0
