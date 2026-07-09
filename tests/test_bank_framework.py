from src.analysis.bank_framework import assemble_bank_verdict, return_on_assets
from src.analysis.framework import valuation_vs_history
from src.pipeline import build_company_report
from src.research.report import QualityTier
from src.research.verification import SourcedValue


def test_roa_tiers():
    assert return_on_assets(120, 10000).verdict == "strong"   # ROA 1.2%
    assert return_on_assets(30, 10000).verdict == "weak"      # 0.3%
    assert return_on_assets(30, 10000).concern is True
    assert return_on_assets(70, 10000).verdict == "mixed"     # 0.7%
    assert return_on_assets(None, 10000).known is False
    assert return_on_assets(120, 0).known is False


def test_bank_verdict_quality_from_roa_and_carries_caveat():
    v = assemble_bank_verdict(valuation_vs_history(None, None), return_on_assets(120, 10000))
    assert v.quality == QualityTier.STRONG
    assert any("GNPA" in r for r in v.reasons)     # the "check the filing" caveat is present


def test_build_report_bank_uses_roa_not_leverage():
    figs = {
        "net_profit": [SourcedValue(120, "a"), SourcedValue(120, "b")],
        "total_assets": [SourcedValue(10000, "a"), SourcedValue(10000, "b")],
    }
    r = build_company_report("SBIN", figs, is_bank=True)
    assert r.verdict.quality == QualityTier.STRONG          # ROA 1.2% -> strong
    assert any("GNPA" in x for x in r.verdict.reasons)


def test_non_bank_still_uses_industrial_framework():
    figs = {
        "total_debt": [SourcedValue(20, "a"), SourcedValue(20, "b")],
        "equity": [SourcedValue(100, "a"), SourcedValue(100, "b")],
        # WHY: two corroborating industrial quality signals (leverage + earnings quality) are
        # needed for STRONG now; one alone reads MIXED (see framework over-confidence fix).
        "operating_cash_flow": [SourcedValue(90, "a"), SourcedValue(90, "b")],
        "net_profit": [SourcedValue(100, "a"), SourcedValue(100, "b")],
    }
    r = build_company_report("X", figs, is_bank=False)   # D/E 0.2 healthy + OCF 90% -> strong
    assert r.verdict.quality == QualityTier.STRONG
    assert not any("GNPA" in x for x in r.verdict.reasons)
