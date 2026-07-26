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


# --- W7 trust UI (SPEC v4 §4): claim-type badges, freshness banners, show-the-computation ---

def _trust_registry():
    from src.sources.registry import CredibilityTier, Source, SourceRegistry
    return SourceRegistry([
        Source("ar", "Annual Report", CredibilityTier.PRIMARY),
        Source("news_google", "Google News", CredibilityTier.ANALYST),
    ])


def test_claim_badge_green_only_for_a_verified_primary_backed_fact():
    # WHY (real money, the one green-tick invariant): the GREEN badge is the app's strongest trust
    # signal and must appear ONLY for a FACT backed solely by primary sources -- never for opinion,
    # an estimate, or an unverified/misquoted figure. Mirrors the Ask tab's own message rendering.
    from src.research.claims import FACT, Citation
    from src.sources.registry import CredibilityTier
    app = _import_app_with_clean_env()
    reg = _trust_registry()
    fact = app_claim(FACT, [Citation("ar", CredibilityTier.PRIMARY, "FY2024 chunk 0",
                                     quote="Net profit was Rs 121 crore.")])
    label, color = app.claim_badge(fact, reg)
    assert color == "green" and "Verified fact" in label


def test_claim_badge_opinion_estimate_and_the_two_unverified_flavors():
    from src.research.claims import ESTIMATE, OPINION, UNVERIFIED, Citation
    from src.sources.registry import CredibilityTier
    app = _import_app_with_clean_env()
    reg = _trust_registry()
    opinion = app_claim(OPINION, [Citation("news_google", CredibilityTier.ANALYST,
                                           "Reuters, 2026-05-15 chunk 0")])
    assert app.claim_badge(opinion, reg)[1] == "blue"
    estimate = app_claim(ESTIMATE, [Citation("ar", CredibilityTier.PRIMARY, "FY2024 chunk 0")])
    assert app.claim_badge(estimate, reg)[1] == "grey"
    # UNVERIFIED resting only on a PRIMARY source = a misquoted/absent figure -> hard RED warning
    unv_primary = app_claim(UNVERIFIED, [Citation("ar", CredibilityTier.PRIMARY, "FY2024 chunk 0")])
    assert app.claim_badge(unv_primary, reg)[1] == "red"
    # UNVERIFIED resting on news/analyst text = reported context -> softer ORANGE, never red or green
    unv_news = app_claim(UNVERIFIED, [Citation("news_google", CredibilityTier.ANALYST,
                                              "Reuters, undated chunk 0")])
    assert app.claim_badge(unv_news, reg)[1] == "orange"


def test_claim_freshness_flags_stale_fresh_undated_and_skips_undated_figure_docs():
    # WHY (real money, "never present a stale figure as current"): a news date lives in the citation
    # locator ("Publisher, YYYY-MM-DD"); describe_freshness turns it into a visible stale/fresh/
    # undated verdict. A figure/filing locator carries no content date and must yield NO freshness
    # line (no false "fresh" on an undated figure).
    from src.research.claims import FACT, OPINION, Citation
    from src.sources.registry import CredibilityTier
    app = _import_app_with_clean_env()
    today = "2026-07-25"
    stale = app_claim(OPINION, [Citation("news_google", CredibilityTier.ANALYST,
                                         "Reuters, 2020-01-01 chunk 0")])
    (line, is_stale, is_unknown), = app.claim_freshness_lines(stale, today)
    assert is_stale and not is_unknown and "2020-01-01" in line
    fresh = app_claim(OPINION, [Citation("news_google", CredibilityTier.ANALYST,
                                         "Reuters, 2026-07-20 chunk 0")])
    assert app.claim_freshness_lines(fresh, today)[0][1] is False       # 5 days < 30 -> not stale
    undated = app_claim(OPINION, [Citation("news_google", CredibilityTier.ANALYST,
                                           "Reuters, undated chunk 0")])
    assert app.claim_freshness_lines(undated, today)[0][2] is True      # freshness unknown
    figure_doc = app_claim(FACT, [Citation("ar", CredibilityTier.PRIMARY,
                                           "RELIANCE verified figures chunk 0")])
    assert app.claim_freshness_lines(figure_doc, today) == []           # no date -> no line


def test_format_computed_figure_shows_value_inputs_and_formula():
    # WHY (SPEC v4 §2 compute-don't-generate, transparency): the show-the-computation panel must
    # prove the number was derived by the system, so it renders the finished value in its unit, the
    # exact source inputs, and the formula.
    from src.research.computed_figures import ComputedFigure
    app = _import_app_with_clean_env()
    fig = ComputedFigure(label="year-over-year growth", value=10.0, unit="percent",
                         inputs=(110.0, 121.0), formula="(current - previous) / |previous| * 100")
    s = app.format_computed_figure(fig)
    assert "10.00%" in s and "110.00" in s and "121.00" in s
    assert "growth" in s and "previous" in s


def test_trust_badges_render_as_streamlit_elements_through_the_real_runtime():
    # WHY: assert the trust badge the Ask tab emits (st.badge(label, color) -> a ':color-badge[..]'
    # element) actually renders through Streamlit's OWN runtime for the colors app.claim_badge
    # chooses -- an invalid color/label would surface here, before it reaches the parents' live page.
    # Connects the badge LOGIC (claim_badge picks color) to the RENDER (st.badge produces an element).
    from streamlit.testing.v1 import AppTest

    from src.research.claims import FACT, OPINION, UNVERIFIED, Citation
    from src.sources.registry import CredibilityTier
    app = _import_app_with_clean_env()
    reg = _trust_registry()
    claims = [
        app_claim(FACT, [Citation("ar", CredibilityTier.PRIMARY, "FY2024 chunk 0", quote="q")]),
        app_claim(OPINION, [Citation("news_google", CredibilityTier.ANALYST,
                                     "Reuters, 2026-05-15 chunk 0")]),
        app_claim(UNVERIFIED, [Citation("ar", CredibilityTier.PRIMARY, "FY2024 chunk 0")]),
    ]
    badges = [app.claim_badge(c, reg) for c in claims]
    lines = "\n".join(f"st.badge({label!r}, color={color!r})" for label, color in badges)
    at = AppTest.from_string("import streamlit as st\n" + lines).run(timeout=60)
    assert len(at.exception) == 0
    values = [m.value for m in at.markdown]
    assert len(values) == len(badges)
    assert any(v.startswith(":green-badge[") for v in values)   # verified fact -> green rendered
    assert any(v.startswith(":blue-badge[") for v in values)    # opinion -> blue rendered
    assert any(v.startswith(":red-badge[") for v in values)     # unverified-primary -> red rendered


def app_claim(kind, citations):
    """A Claim of `kind` with the given citations (helper local to these trust-UI tests)."""
    from src.research.claims import Claim
    return Claim(text="x", citations=tuple(citations), kind=kind)
