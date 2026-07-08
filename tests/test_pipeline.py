import pytest

from src.pipeline import build_company_report
from src.research.report import Leaning, QualityTier, ReviewStatus, ValuationTier
from src.samples import SAMPLE_COMPANIES


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
