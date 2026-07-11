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
