"""ask_no_figures_tip lives in app.py, so it's exercised by importing the module directly (same
env-safe pattern as test_app_cache_ttls.py / test_pdf_report.py).
"""
import os


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


def test_never_researched_this_session_says_research_it_first():
    app = _import_app_with_clean_env()
    tip = app.ask_no_figures_tip("RELIANCE", already_researched_this_session=False)
    assert "research it in the 'Research a Stock' tab first" in tip


def test_already_researched_but_nothing_cross_verified_does_not_claim_unresearched():
    # WHY (real money, workflow honesty; regression): verified_figures_document returns None
    # whether the stock was NEVER researched this session, or WAS researched but every figure
    # came back single-source or in genuine CONFLICT -- vf_doc is None can't tell these apart on
    # its own. Telling a user who already researched the stock to "research it first" is a false
    # claim about what they just did, and re-researching cannot resolve a genuine cross-source
    # disagreement between yfinance and Screener -- point them at the evidence panel instead.
    app = _import_app_with_clean_env()
    tip = app.ask_no_figures_tip("RELIANCE", already_researched_this_session=True)
    assert "research it in the 'Research a Stock' tab first" not in tip
    assert "already researched" in tip
    assert "evidence panel" in tip
    # WHY (found by adversarial review): verified_figures_document also returns None when a
    # figure was found by NEITHER source at all (UNVERIFIABLE), not just single-source/conflict --
    # the message must not imply a figure necessarily exists somewhere, just unreconciled.
    assert "unavailable" in tip or "not found" in tip


def test_conflict_values_line_shows_each_sources_actual_value_in_its_unit():
    # WHY (real money, review workflow + honesty): a CONFLICT figure is otherwise shown only as
    # "withheld", hiding WHAT the sources disagreed on. The expert must acknowledge a conflict
    # before approving, and the disagreeing numbers are what let them tell a benign definitional
    # gap (e.g. to-owners vs consolidated net profit) from a real parse/scale error.
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    app = _import_app_with_clean_env()
    fig = VerifiedFigure(
        "net_profit", VerificationStatus.CONFLICT, None,
        (SourcedValue(807750000000.0, "yfinance"), SourcedValue(957540000000.0, "screener")),
        "independent sources disagree beyond tolerance")
    line = app.conflict_values_line(fig)
    assert "yfinance" in line and "screener" in line
    assert "80,775 crore" in line and "95,754 crore" in line   # crore units, both values visible


def test_conflict_values_line_formats_ratios_and_percents_correctly():
    from src.research.verification import SourcedValue, VerificationStatus, VerifiedFigure
    app = _import_app_with_clean_env()
    pe = VerifiedFigure("current_pe", VerificationStatus.CONFLICT, None,
                        (SourcedValue(22.7, "yfinance"), SourcedValue(2.27, "screener")), "")
    line = app.conflict_values_line(pe)
    assert "22.7x" in line and "2.3x" in line          # ratio unit, not rupees -- catches a 10x gap
