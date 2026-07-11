from src.analysis.sizing import (
    AllocationCandidate,
    Stance,
    long_term_guidance,
    position_sizing,
    stance_from_verdict,
    suggest_allocation,
    verdict_strength,
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


# --- verdict_strength (fine-grained ranking refinement) ---

def _full(valuation, quality, confidence):
    return Verdict(valuation=valuation, quality=quality,
                   leaning=Leaning.CONSTRUCTIVE, confidence=confidence)


def test_verdict_strength_is_bounded_0_to_1():
    top = _full(ValuationTier.CHEAP, QualityTier.STRONG, Confidence.HIGH)
    bottom = _full(ValuationTier.EXPENSIVE, QualityTier.WEAK, Confidence.LOW)
    assert verdict_strength(top) == 1.0
    assert 0.0 <= verdict_strength(bottom) < 0.3        # low-confidence floor, not zero
    assert verdict_strength(None) == 0.0


def test_verdict_strength_rewards_deeper_margin_of_safety():
    # WHY: two names both CHEAP on the tier, but one trades far below its history. The deeper
    # discount (bigger margin of safety) must rank higher — the tier alone can't see that.
    deep = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                   Confidence.HIGH, valuation_ratio=0.40)      # ~60% below its own median
    shallow = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                      Confidence.HIGH, valuation_ratio=0.78)   # barely cheap
    assert verdict_strength(deep) > verdict_strength(shallow)


def test_verdict_strength_falls_back_to_tier_without_a_ratio():
    # No ratio (median unavailable) -> tier degree, unchanged behavior.
    cheap = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.HIGH)
    fair = Verdict(ValuationTier.FAIR, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.HIGH)
    assert verdict_strength(cheap) > verdict_strength(fair)


def test_verdict_strength_rewards_cheaper_stronger_more_confident():
    cheap = _full(ValuationTier.CHEAP, QualityTier.STRONG, Confidence.HIGH)
    fair = _full(ValuationTier.FAIR, QualityTier.STRONG, Confidence.HIGH)
    assert verdict_strength(cheap) > verdict_strength(fair)          # margin of safety counts
    strong = _full(ValuationTier.CHEAP, QualityTier.STRONG, Confidence.HIGH)
    mixed = _full(ValuationTier.CHEAP, QualityTier.MIXED, Confidence.HIGH)
    assert verdict_strength(strong) > verdict_strength(mixed)        # business quality counts
    hi = _full(ValuationTier.CHEAP, QualityTier.STRONG, Confidence.HIGH)
    med = _full(ValuationTier.CHEAP, QualityTier.STRONG, Confidence.MEDIUM)
    assert verdict_strength(hi) > verdict_strength(med)              # how sure we are counts


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
    # One favorable name, book 100, add 100, cap 25%. Capped against the TOTAL money considered:
    # base = 100 + 100 = 200 (fixed), cap 25% => room 50, so 50 is placed and 50 stays uninvested.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.invested == 50.0
    assert plan.uninvested == 50.0
    assert any("per-stock cap" in n for n in plan.notes)
    # WHY (honesty, found by adversarial review): "resulting book" is stale wording left over
    # from the old self-referential design -- the cap here is measured against the TOTAL money
    # considered (this ask + existing holdings), not the smaller book that actually ends up
    # invested. Must not claim a basis the code no longer uses.
    assert not any("resulting book" in n for n in plan.notes)


def test_allocation_discloses_that_realized_concentration_runs_above_cap_pct_when_cash_remains():
    # WHY (real money, honesty; found by adversarial review): the cap is measured against total
    # money considered (pv + amount), not the smaller book that actually ends up invested. So
    # when cash is left uninvested, each placed name is a BIGGER share of what's ACTUALLY invested
    # than cap_pct -- e.g. here A gets 50 against a book of only 50 actually deployed (100%, not
    # 25%). position_sizing()/over_cap, used elsewhere in this app, would immediately flag this
    # exact position as over its cap. The plan must disclose this instead of silently implying
    # the placed amount is safely under cap_pct of the investor's real resulting stock portfolio.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(100.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.uninvested > 0
    joined = " ".join(plan.notes)
    assert "actually invested" in joined.lower() or "actually deployed" in joined.lower()


def test_allocation_never_exceeds_cap_of_total_money_considered():
    # WHY: the cap is measured against the FULL money under consideration (existing holdings +
    # the whole lump sum being invested), not just whatever ends up deployed. Money that stays
    # uninvested is still the investor's money -- it hasn't left their control, they can still
    # deploy it into another approved name later, so it still counts toward "how concentrated
    # would this position be against everything I have to invest".
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0),
             AllocationCandidate("B", Stance.FAVORABLE, 0.0)]
    for pv, amount in [(0.0, 100.0), (100.0, 100.0), (50.0, 500.0)]:
        plan = suggest_allocation(amount, cands, portfolio_value=pv, cap_pct=0.25)
        total_money = pv + amount
        for a in plan.allocations:
            held = next(c.current_value for c in cands if c.symbol == a.symbol)
            assert held + a.amount <= 0.25 * total_money + 1.0  # within cap of total money considered


def test_allocation_new_investor_with_few_names_still_gets_a_useful_answer():
    # WHY (real money, regression): capping against the REALIZED (deployed-only) book instead of
    # the total money under consideration creates a self-referential trap for exactly the most
    # common real scenario -- a new investor (portfolio_value=0) with only a handful of approved
    # names relative to the cap. With 4 names and a 20% cap, 4*20%=80%<100%, so no allocation
    # using ONLY these 4 names can ever keep each <=20% of a book made ENTIRELY of these 4 names
    # (their shares must average 25% of each other). Iterating "cap against what actually got
    # placed" spirals toward placing almost nothing (live-verified: ~13 rupees of a 100,000 ask)
    # instead of the obviously sound answer -- an even 20,000 into each name, 20,000 left as cash.
    cands = [AllocationCandidate(f"S{i}", Stance.FAVORABLE, 0.0) for i in range(4)]
    plan = suggest_allocation(100_000.0, cands, portfolio_value=0.0, cap_pct=0.20)
    assert plan.invested >= 79_000.0   # not ~13 rupees
    amounts = sorted(a.amount for a in plan.allocations)
    assert all(abs(x - 20_000.0) < 1.0 for x in amounts)


def test_allocation_over_cap_name_gets_nothing():
    cands = [AllocationCandidate("A", Stance.FAVORABLE, current_value=60.0)]  # already > cap
    plan = suggest_allocation(30.0, cands, portfolio_value=100.0, cap_pct=0.25)
    assert plan.allocations == ()
    assert plan.uninvested == 30.0


def test_allocation_over_cap_name_excluded_others_absorb_its_share():
    # WHY: a MIXED batch (some already over cap, some with room) in the SAME call -- confirms the
    # over-cap name doesn't "poison" the batch or leave its share stranded as uninvested; it's
    # cleanly excluded (zero room) and the full amount flows to the remaining eligible names.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, current_value=40.0),   # over 25% cap
             AllocationCandidate("B", Stance.FAVORABLE, current_value=0.0),
             AllocationCandidate("C", Stance.FAVORABLE, current_value=0.0)]
    plan = suggest_allocation(60.0, cands, portfolio_value=100.0, cap_pct=0.25)
    by = {a.symbol: a.amount for a in plan.allocations}
    assert "A" not in by                                    # over-cap name gets nothing
    assert abs(by["B"] - 30.0) < 0.5 and abs(by["C"] - 30.0) < 0.5   # its share split evenly
    assert plan.uninvested == 0.0


def test_allocation_explains_a_name_skipped_because_it_is_already_over_cap():
    # WHY (real money, clarity): a FAVORABLE, approved name the user already holds AT/OVER the
    # per-stock cap correctly gets nothing (adding more would over-concentrate) -- but it then
    # vanishes from the suggested spread with NO reason, so a user wonders why their approved
    # favorite got zero. Name it and say the cap comes first. A name that simply wasn't reached
    # (money ran out while still under cap) must NOT be labeled over-cap.
    cands = [AllocationCandidate("AAA", Stance.FAVORABLE, current_value=500000),   # already over cap
             AllocationCandidate("BBB", Stance.FAVORABLE, current_value=0)]
    plan = suggest_allocation(300000, cands, portfolio_value=700000, cap_pct=0.25)  # cap_value 250k
    assert not any(a.symbol == "AAA" for a in plan.allocations)     # correctly gets nothing
    joined = " ".join(plan.notes)
    assert "AAA" in joined and "cap" in joined.lower()               # and it's explained
    assert "BBB" not in joined                                       # BBB got the money; not mislabeled


def test_allocation_near_cap_absorber_caps_out_rest_reported_uninvested():
    # A near-cap name (small real room) plus an over-cap name, asked for far more than the
    # available room: A is filled up to (exactly) its cap of the TOTAL money considered (pv +
    # amount = 160, cap 25% => room 40, A already holds 20 => 20 more), never forced beyond it,
    # and the genuine remainder is honestly reported as uninvested, not silently dropped or
    # force-fit into a name past its cap.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, current_value=20.0),   # room to 40
             AllocationCandidate("B", Stance.FAVORABLE, current_value=40.0)]   # already over cap
    plan = suggest_allocation(60.0, cands, portfolio_value=100.0, cap_pct=0.25)
    by = {a.symbol: a.amount for a in plan.allocations}
    assert "B" not in by
    total_money = 100.0 + 60.0
    assert 20.0 + by["A"] <= 0.25 * total_money + 1.0   # never exceeds cap of total money considered
    assert plan.uninvested > 30.0                            # the genuine remainder, not silently lost
    assert abs(plan.invested + plan.uninvested - 60.0) < 1e-6  # accounts for the full requested amount


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


def test_guidance_unfavorable_does_not_call_unverified_quality_solid():
    # WHY (real money, over-confidence): expensive valuation can co-occur with UNKNOWN quality
    # (e.g. price data cross-verifies but no quality signal does) -- assemble_verdict's leaning
    # logic reaches CAUTIOUS/UNFAVORABLE via EXPENSIVE valuation alone, regardless of quality.
    # The guidance must not claim "solid business" when quality was never actually verified.
    v = _v(ValuationTier.EXPENSIVE, QualityTier.UNKNOWN, Leaning.CAUTIOUS)
    g = long_term_guidance(Stance.UNFAVORABLE, position_sizing(0, 100, 0.25), v, held=False)
    joined = " ".join(g.points)
    assert "solid business" not in joined.lower()
    assert "quality" in joined.lower() and ("unverified" in joined.lower()
                                            or "not verified" in joined.lower()
                                            or "couldn't be verified" in joined.lower())


def test_guidance_unfavorable_mixed_quality_is_hedged_not_solid():
    # A MIXED quality (some concern, not weak enough to be WEAK) is also not "solid" -- must be
    # described honestly, distinct from both the WEAK case and the genuinely STRONG case.
    v = _v(ValuationTier.EXPENSIVE, QualityTier.MIXED, Leaning.CAUTIOUS)
    g = long_term_guidance(Stance.UNFAVORABLE, position_sizing(0, 100, 0.25), v, held=False)
    joined = " ".join(g.points)
    assert "solid business" not in joined.lower()


def test_guidance_unfavorable_strong_quality_still_says_solid():
    # Regression: the genuinely-earned "solid business" phrasing must still appear when quality
    # really is STRONG (this is the existing, correct case; must not be lost by the fix above).
    v = _v(ValuationTier.EXPENSIVE, QualityTier.STRONG, Leaning.CAUTIOUS)
    g = long_term_guidance(Stance.UNFAVORABLE, position_sizing(0, 100, 0.25), v, held=False)
    assert any("solid business" in p.lower() for p in g.points)


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


def test_allocation_diversifies_when_the_cap_is_loose():
    # WHY (diversification): with a large book the per-stock cap is loose, so a greedy fill would
    # pour the whole lump into ONE name. "Spread it across your approved names" must actually
    # spread it, not concentrate — poor, unsafe advice for a non-expert investing real money.
    cands = [AllocationCandidate("A", Stance.FAVORABLE, 0.0),
             AllocationCandidate("B", Stance.FAVORABLE, 0.0),
             AllocationCandidate("C", Stance.FAVORABLE, 0.0)]
    plan = suggest_allocation(90.0, cands, portfolio_value=1000.0, cap_pct=0.25)
    amounts = sorted(a.amount for a in plan.allocations)
    assert len(plan.allocations) == 3                      # all three funded, not one
    assert all(abs(x - 30.0) < 1.0 for x in amounts)       # ~even split (cap of ~₹272 is loose)
    assert plan.uninvested == 0.0


def test_allocation_even_spread_still_prefers_favorable_over_neutral():
    # Favorable names are filled evenly first; neutral only absorbs the remainder.
    cands = [AllocationCandidate("F1", Stance.FAVORABLE, 0.0),
             AllocationCandidate("F2", Stance.FAVORABLE, 0.0),
             AllocationCandidate("N1", Stance.NEUTRAL, 0.0)]
    plan = suggest_allocation(60.0, cands, portfolio_value=1000.0, cap_pct=0.25)
    by = {a.symbol: a.amount for a in plan.allocations}
    assert abs(by["F1"] - 30.0) < 1.0 and abs(by["F2"] - 30.0) < 1.0   # favorable split evenly
    assert "N1" not in by                                              # neutral untouched (fav absorbed all)
