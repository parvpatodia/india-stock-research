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


def test_ask_source_caption_shows_publisher_date_for_news_and_dedups():
    # WHY (real money, Ask-tab freshness): a news-backed claim's "Source:" line must let the reader
    # judge how recent it is. For a dated news item, surface the publisher + article date; the app's
    # own figure/filing documents keep just their name (their internal locator is redundant noise).
    app = _import_app_with_clean_env()
    from src.research.claims import Citation
    from src.sources.registry import CredibilityTier, Source, SourceRegistry
    reg = SourceRegistry([
        Source("news_google", "Google News", CredibilityTier.ANALYST),
        Source("verified_figures", "This app's cross-verified figures", CredibilityTier.PRIMARY),
    ])
    news = Citation("news_google", CredibilityTier.ANALYST, "Reuters, 2026-05-15 chunk 0")
    fig = Citation("verified_figures", CredibilityTier.PRIMARY, "RELIANCE verified figures chunk 0")
    assert app.ask_source_caption([news], reg) == "Google News — Reuters, 2026-05-15"
    assert app.ask_source_caption([fig], reg) == "This app's cross-verified figures"
    # two chunks of the same source de-duplicate to one label:
    news2 = Citation("news_google", CredibilityTier.ANALYST, "Reuters, 2026-05-15 chunk 1")
    assert app.ask_source_caption([news, news2], reg) == "Google News — Reuters, 2026-05-15"
    # an UNDATED news item keeps its publisher AND is flagged "undated" (freshness unknown), so a
    # parent doesn't assume it's recent -- it must NOT be reduced to the bare feed name like a
    # figure doc. This completes the same freshness-transparency the dated case above provides.
    undated = Citation("news_google", CredibilityTier.ANALYST, "Reuters, undated chunk 0")
    assert app.ask_source_caption([undated], reg) == "Google News — Reuters, undated"
    assert app.ask_source_caption([], reg) == "no source"
