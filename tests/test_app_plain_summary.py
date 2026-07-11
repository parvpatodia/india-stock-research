"""plain_summary lives in app.py; import the module directly (env-safe pattern, matching
test_app_ask_tab.py) and check its sector-aware wording -- a bank must never be summarized as
having a 'balance sheet' verdict the app cannot actually assess for a lender."""
import os

from src.analysis.sizing import Stance
from src.research.report import (
    Confidence,
    Leaning,
    QualityTier,
    ValuationTier,
    Verdict,
)


def _import_app_with_clean_env():
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        import app
        return app
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_plain_summary_industrial_uses_balance_sheet_language():
    app = _import_app_with_clean_env()
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM)
    s = app.plain_summary(v, Stance.FAVORABLE)
    assert "balance sheet" in s


def test_plain_summary_unknown_valuation_reads_grammatically():
    # WHY (real money, UI honesty): a real industrial whose historical median P/E can't be computed
    # (thin/short price history, or loss years break the EPS series) but whose balance sheet cross-
    # verifies STRONG reads NEUTRAL with valuation UNKNOWN -- a reachable production state. The
    # one-liner is built as "It {val}, with {qual}."; every other val phrase is a verb clause
    # ("looks cheap versus its own history"), but the UNKNOWN phrase was a bare noun ("valuation
    # could not be verified"), producing the broken "It valuation could not be verified, with a
    # strong balance sheet." Broken English in the headline summary of a money tool erodes trust;
    # the sentence must compose grammatically while still conveying the valuation is unconfirmed.
    app = _import_app_with_clean_env()
    v = Verdict(ValuationTier.UNKNOWN, QualityTier.STRONG, Leaning.NEUTRAL, Confidence.MEDIUM)
    s = app.plain_summary(v, Stance.NEUTRAL)
    assert "It valuation could not be verified" not in s   # the broken, ungrammatical phrasing
    assert s.startswith("It ")
    assert "valuation" in s.lower()                        # still tells the reader valuation is unconfirmed
    assert "strong balance sheet" in s                     # quality still described


def test_plain_summary_bank_does_not_claim_a_balance_sheet_verdict():
    # WHY (real money, sector-aware honesty): a bank's quality tier comes from ROA (profitability),
    # and the app CANNOT assess a lender's actual balance-sheet quality (asset quality/GNPA, capital
    # adequacy) from the free feeds -- so the plain one-liner must never tell a parent a bank has a
    # "strong balance sheet". It reflects the lender's profitability instead.
    app = _import_app_with_clean_env()
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE, Confidence.MEDIUM,
                is_bank=True)
    s = app.plain_summary(v, Stance.FAVORABLE)
    assert "balance sheet" not in s
    assert "profitability" in s.lower()


def test_data_vintage_note_names_latest_annual_fy_and_points_at_recent_quarters():
    # WHY (real money, honesty): the quality verdict is built entirely on the latest ANNUAL figures,
    # which never include recent quarters -- a vintage only implicit in FY tags inside the collapsed
    # evidence panel. A non-expert reading "Evidence leans favorable" as the headline could act on a
    # view up to ~15 months stale. Surface it, naming the LATEST cross-verified fiscal year.
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    app = _import_app_with_clean_env()
    figs = (
        VerifiedFigure("net_profit", VerificationStatus.VERIFIED, 100.0,
                       (SourcedValue(100.0, "yfinance", locator="FY2025"),
                        SourcedValue(100.0, "screener", locator="FY2025")), "agree"),
        VerifiedFigure("total_debt", VerificationStatus.VERIFIED, 50.0,
                       (SourcedValue(50.0, "yfinance", locator="FY2024"),
                        SourcedValue(50.0, "screener", locator="FY2024")), "agree"),
    )
    note = app.data_vintage_note(figs)
    assert note is not None
    assert "FY2025" in note and "FY2024" not in note   # the LATEST annual vintage, not an older leg
    assert "quarter" in note.lower()                    # points at the actionable step


def test_data_vintage_note_none_without_a_cross_verified_annual_figure():
    # A valuation-only/point-figure report has no annual vintage to caveat; a single-source (not
    # trustworthy) annual figure doesn't count either -- only what actually drove the verdict does.
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    app = _import_app_with_clean_env()
    figs = (
        VerifiedFigure("current_pe", VerificationStatus.VERIFIED, 18.0,
                       (SourcedValue(18.0, "yfinance"), SourcedValue(18.0, "screener")), "agree"),
        VerifiedFigure("net_profit", VerificationStatus.SINGLE_SOURCE, 100.0,
                       (SourcedValue(100.0, "yfinance", locator="FY2025"),), "single"),
    )
    assert app.data_vintage_note(figs) is None
