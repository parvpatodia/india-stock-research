from src.analysis.bank_framework import _industry_category, assemble_bank_verdict, return_on_assets
from src.analysis.framework import REAL_ESTATE_LEVERAGE_CAVEAT, valuation_vs_history
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
    assert any("GNPA" in c for c in v.sector_caveats)   # "check the filing" caveat is present
    # WHY: the caveat is NOT itself a cross-verified figure, so it must never blend into
    # `reasons` -- the app renders that list under a "Why (each from cross-verified figures)"
    # header, which must stay literally true.
    assert not any("GNPA" in r for r in v.reasons)


def test_build_report_bank_uses_roa_not_leverage():
    figs = {
        "net_profit": [SourcedValue(120, "a"), SourcedValue(120, "b")],
        "total_assets": [SourcedValue(10000, "a"), SourcedValue(10000, "b")],
    }
    r = build_company_report("SBIN", figs, is_bank=True)
    assert r.verdict.quality == QualityTier.STRONG          # ROA 1.2% -> strong
    assert any("GNPA" in x for x in r.verdict.sector_caveats)
    assert not any("GNPA" in x for x in r.verdict.reasons)


def test_industry_category_detects_banks():
    assert _industry_category("Banks - Regional") == "bank"
    assert _industry_category("Banks - Diversified") == "bank"


def test_industry_category_detects_nbfc_lenders():
    # WHY (sector-aware analysis): NBFCs (Bajaj Finance, Cholamandalam, housing-finance cos) borrow
    # to lend, just like a bank, so debt/equity is a feature of the business model, not a risk
    # signal. yfinance tags these industries 'Credit Services' / 'Financial - Mortgages'.
    assert _industry_category("Credit Services") == "nbfc"
    assert _industry_category("Financial - Credit Services") == "nbfc"
    assert _industry_category("Financial - Mortgages") == "nbfc"


def test_industry_category_other_financials_stay_industrial():
    # Insurance, asset management, capital markets/exchanges do NOT run a borrow-to-lend model;
    # they must stay on the industrial D/E lens, not be swept into the ROA-only framework.
    assert _industry_category("Insurance - Life") == "other"
    assert _industry_category("Asset Management") == "other"
    assert _industry_category("Capital Markets") == "other"
    assert _industry_category("") == "other"


def test_industry_category_detects_real_estate():
    # WHY (sector-aware analysis, live-verified): DLF, Godrej Properties, Oberoi Realty, Sobha,
    # Lodha all tag "Real Estate - Development"; Prestige and Phoenix Mills tag "Real Estate -
    # Diversified". Real-estate developers still run a genuine D/E lens (unlike banks/NBFCs, this
    # is not a borrow-to-lend model) but commonly carry higher leverage funded against project
    # collections/RERA-escrow, so the generic industrial bands can misread a normally-financed
    # developer as stretched. Live-verified D/E across 8 real names: DLF 0.01, Oberoi 0.16, Sobha
    # 0.22, Lodha 0.42, Phoenix 0.48 (all already read "healthy"/"moderate" fine); Godrej
    # Properties 0.83 and Brigade 0.93 sit close to (but just under) the generic 1.00 "stretched"
    # line, reading "moderate"; Prestige 1.09 is the one that actually crosses it, reading "high,
    # worth watching"/stretched despite being a large, established developer, not a distressed
    # one -- exactly the mislabeling this caveat exists to flag.
    assert _industry_category("Real Estate - Development") == "real_estate"
    assert _industry_category("Real Estate - Diversified") == "real_estate"


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
    assert not any(x == REAL_ESTATE_LEVERAGE_CAVEAT for x in r.verdict.reasons)
    assert r.verdict.sector_caveats == ()


def test_build_report_real_estate_carries_leverage_caveat():
    # WHY (sector-aware analysis): a real-estate developer at D/E 1.09 (Prestige's live-verified
    # figure) reads "stretched"/"high, worth watching" under the generic industrial band, but
    # that leverage is commonly normal for a developer funded against project collections/RERA-
    # escrow. Rather than silently loosen the band with an invented number, disclose the sector
    # context (same honesty-first pattern as the bank/NBFC caveat) so the reader checks sector
    # peers and collections momentum instead of reading this as a generic solvency red flag.
    figs = {
        "total_debt": [SourcedValue(109, "a"), SourcedValue(109, "b")],
        "equity": [SourcedValue(100, "a"), SourcedValue(100, "b")],
        "operating_cash_flow": [SourcedValue(90, "a"), SourcedValue(90, "b")],
        "net_profit": [SourcedValue(100, "a"), SourcedValue(100, "b")],
    }
    r = build_company_report("PRESTIGE", figs, is_bank=False, is_real_estate=True)
    assert any(x == REAL_ESTATE_LEVERAGE_CAVEAT for x in r.verdict.sector_caveats)
    assert not any(x == REAL_ESTATE_LEVERAGE_CAVEAT for x in r.verdict.reasons)


def test_build_report_real_estate_with_comfortable_debt_has_no_caveat_clutter():
    # WHY: a real-estate name with low/comfortable debt (e.g. DLF, live D/E 0.01) has nothing to
    # caveat -- attaching REAL_ESTATE_LEVERAGE_CAVEAT unconditionally to every real-estate report
    # regardless of its actual leverage would be clutter dressed up as diligence, not honesty.
    figs = {
        "total_debt": [SourcedValue(1, "a"), SourcedValue(1, "b")],
        "equity": [SourcedValue(100, "a"), SourcedValue(100, "b")],
        "operating_cash_flow": [SourcedValue(90, "a"), SourcedValue(90, "b")],
        "net_profit": [SourcedValue(100, "a"), SourcedValue(100, "b")],
    }
    r = build_company_report("DLF", figs, is_bank=False, is_real_estate=True)
    assert not any(x == REAL_ESTATE_LEVERAGE_CAVEAT for x in r.verdict.reasons)
    assert r.verdict.sector_caveats == ()
