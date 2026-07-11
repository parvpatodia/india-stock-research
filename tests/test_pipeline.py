import pytest

from src.pipeline import build_company_report
from src.research.report import Leaning, QualityTier, ReviewStatus, ValuationTier
from src.research.verification import SourcedValue
from src.samples import SAMPLE_COMPANIES


def _two_source(figs: dict) -> dict:
    """Wrap each scalar as two agreeing sources so every figure cross-verifies."""
    return {name: [SourcedValue(v, "a"), SourcedValue(v, "b")] for name, v in figs.items()}


def test_weak_roce_pulls_quality_below_strong():
    # WHY (CA-level rigor, real money): the verdict now factors return on capital employed. A
    # business with strong cash conversion (OCF ~92% of profit), a healthy balance sheet (D/E 0.01,
    # interest cover 8x) and no promoter pledge -- clean on the OLD three quality lenses -- but a
    # WEAK ROCE of ~8% (below the ~11-12% cost of capital) is NOT strong quality: it earns poorly on
    # the capital it deploys. It must read MIXED, not STRONG. (Same figures without the ROCE lens
    # would read STRONG: two positives, zero concerns.)
    r = build_company_report("X", _two_source({
        "operating_cash_flow": 92, "net_profit": 100,       # OCF/NP 0.92 -> strong (positive)
        "total_debt": 5, "equity": 1000,                    # D/E 0.005 -> healthy (positive)
        "ebit": 80, "interest_expense": 10,                 # cover 8x; ROCE 80/1005 = 8.0% -> weak
        "promoter_pledge_pct": 0,                           # none
    }))
    assert r.verdict.quality == QualityTier.MIXED


def test_strong_roce_keeps_a_clean_company_strong():
    # Guard against over-tightening: the SAME clean company with a STRONG ROCE (>=15%) stays STRONG.
    r = build_company_report("Y", _two_source({
        "operating_cash_flow": 92, "net_profit": 100,       # strong
        "total_debt": 5, "equity": 1000,                    # healthy
        "ebit": 200, "interest_expense": 10,                # ROCE 200/1005 = 19.9% -> strong
        "promoter_pledge_pct": 0,
    }))
    assert r.verdict.quality == QualityTier.STRONG


def test_constructive_company_is_draft_not_trusted():
    r = build_company_report("Acme", SAMPLE_COMPANIES["Acme Industries (SAMPLE)"])
    assert r.status == ReviewStatus.DRAFT
    assert r.is_trusted is False           # nothing trusted before expert sign-off
    assert r.verdict.valuation == ValuationTier.CHEAP
    assert r.verdict.quality == QualityTier.STRONG
    assert r.verdict.leaning == Leaning.CONSTRUCTIVE
    assert r.conflicts == ()
    assert r.verdict.caveat                 # verdict always caveated


def test_cautious_company_flags_weak_quality():
    r = build_company_report("Risky", SAMPLE_COMPANIES["Risky Corp (SAMPLE)"])
    assert r.verdict.valuation == ValuationTier.EXPENSIVE
    assert r.verdict.quality == QualityTier.WEAK
    assert r.verdict.leaning == Leaning.CAUTIOUS


def test_conflict_is_flagged_and_blocks_approval():
    r = build_company_report("Risky", SAMPLE_COMPANIES["Risky Corp (SAMPLE)"])
    assert len(r.conflicts) >= 1            # revenue sources disagree
    with pytest.raises(ValueError):
        r.approve(reviewer="expert")        # cannot silently approve over a conflict
    ok = r.approve(reviewer="expert", note="revenue is a units mismatch, reconciled by hand",
                   acknowledge_conflicts=True)
    assert ok.is_trusted


def test_all_sample_figures_are_present_in_report():
    r = build_company_report("Acme", SAMPLE_COMPANIES["Acme Industries (SAMPLE)"])
    names = {f.name for f in r.figures}
    assert "current_pe" in names and "net_profit" in names
