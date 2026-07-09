from src.analysis.sizing import (
    AllocationCandidate,
    Stance,
    long_term_guidance,
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
    # One favorable name, book 100, add 100, cap 25%. Capped against the REALIZED book:
    # base = 100 + placed, placed = 0.25*base  ->  base = 100/0.75 = 133.33, placed = 33.33.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert abs(plan.invested - 100.0 / 3) < 1.0        # ~33.33, not 50
    assert abs(plan.uninvested - 200.0 / 3) < 1.0      # ~66.67
    assert any("per-stock cap" in n for n in plan.notes)


def test_allocation_never_exceeds_cap_of_the_realized_book():
    # WHY (regression): the old code capped against pv+amount, so when caps bound the placed
    # positions exceeded cap% of the smaller realized book. Each name must stay <= cap% of the
    # book it actually ends up in (current holdings + what was placed).
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0),
             AllocationCandidate("B", Stance.FAVORABLE, 0.0)]
    for pv, amount in [(0.0, 100.0), (100.0, 100.0), (50.0, 500.0)]:
        plan = suggest_allocation(amount, cands, portfolio_value=pv, cap_pct=0.25)
        realized = pv + plan.invested
        for a in plan.allocations:
            held = next(c.current_value for c in cands if c.symbol == a.symbol)
            assert held + a.amount <= 0.25 * realized + 1.0   # within cap of the realized book


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


def _v(valuation, quality, leaning, conf=Confidence.MEDIUM):
    return Verdict(valuation=valuation, quality=quality, leaning=leaning, confidence=conf)


def test_guidance_insufficient_data_says_do_not_act():
    g = long_term_guidance(Stance.INSUFFICIENT_DATA,
                           position_sizing(0, 100, 0.25), None, held=False)
    assert "Not enough verified data" in g.headline
    assert any("Don't act" in p for p in g.points)


def test_guidance_favorable_with_room_suggests_accumulate_gradually():
    v = _v(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE)
    g = long_term_guidance(Stance.FAVORABLE, position_sizing(0, 100, 0.25), v, held=False)
    assert "accumulate" in g.headline.lower()
    assert any("room" in p for p in g.points) and any("gradually" in p for p in g.points)


def test_guidance_favorable_over_cap_says_hold_dont_add():
    v = _v(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE)
    g = long_term_guidance(Stance.FAVORABLE, position_sizing(40, 100, 0.25), v, held=True)
    assert "cap" in g.headline.lower()
    assert any("concentrates risk" in p for p in g.points)


def test_guidance_unfavorable_strong_business_hold_dont_add_and_trim():
    v = _v(ValuationTier.EXPENSIVE, QualityTier.STRONG, Leaning.CAUTIOUS)
    g = long_term_guidance(Stance.UNFAVORABLE, position_sizing(40, 100, 0.25), v, held=True)
    assert "don't add" in g.headline.lower()
    assert any("trimming toward" in p for p in g.points)          # over-cap trim suggestion
    assert any("Revisit if" in p for p in g.points)


def test_guidance_weak_quality_reviews_thesis():
    v = _v(ValuationTier.CHEAP, QualityTier.WEAK, Leaning.CAUTIOUS)
    g = long_term_guidance(Stance.UNFAVORABLE, position_sizing(0, 100, 0.25), v, held=True)
    assert "thesis" in g.headline.lower()
    assert any("weak fundamentals matter more" in p for p in g.points)


def test_allocation_spreads_across_two_favorable_names():
    cands = [
        AllocationCandidate("A", Stance.FAVORABLE, 0.0),
        AllocationCandidate("B", Stance.FAVORABLE, 0.0),
    ]
    # base 200, cap 25% => 50 each; 100 asked -> 50 + 50, fully placed.
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert sorted(a.amount for a in plan.allocations) == [50.0, 50.0]
    assert plan.uninvested == 0.0
