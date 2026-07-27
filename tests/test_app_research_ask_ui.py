"""Research verdict rating strip (app.verdict_rating_html) + Ask claim card (app.claim_card_html).

Pure builders, exercised by importing app directly (same env-safe pattern as the other app-fn
tests). Locks the real-money-visible behaviour: favourability coloring, Confidence staying neutral
(it's data coverage, not good/bad news), HTML escaping, and the claim-card trust accents.
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


def _verdict(val, qual, lean, conf):
    from src.research.report import (Confidence, Leaning, QualityTier, ValuationTier, Verdict)
    return Verdict(ValuationTier(val), QualityTier(qual), Leaning(lean), Confidence(conf),
                   reasons=("x",))


def test_rating_strip_tones_dimensions_by_favourability():
    app = _import_app_with_clean_env()
    html = app.verdict_rating_html(_verdict("cheap", "strong", "constructive", "high"))
    assert 'class="rv g">cheap' in html          # cheap valuation is favourable -> green
    assert 'class="rv g">strong' in html
    assert 'class="rv g">constructive' in html
    for label in ("Valuation", "Quality", "Leaning", "Confidence"):
        assert label in html


def test_rating_strip_flags_unfavourable_dimensions_red_or_amber():
    app = _import_app_with_clean_env()
    html = app.verdict_rating_html(_verdict("expensive", "weak", "cautious", "low"))
    assert 'class="rv r">expensive' in html      # expensive -> red
    assert 'class="rv r">weak' in html
    assert 'class="rv o">cautious' in html       # cautious -> amber (a lean, not a hard loss)


def test_confidence_is_always_neutral_never_toned_like_good_news():
    # WHY (real money, comprehension): high confidence means "well cross-verified", NOT "likely to
    # gain" -- so it must never render green/red, always neutral.
    app = _import_app_with_clean_env()
    for conf in ("low", "medium", "high"):
        html = app.verdict_rating_html(_verdict("fair", "mixed", "neutral", conf))
        assert f'class="rv n">{conf}' in html


def test_rating_strip_none_verdict_is_empty():
    app = _import_app_with_clean_env()
    assert app.verdict_rating_html(None) == ""


def test_claim_card_accents_and_escaping():
    app = _import_app_with_clean_env()
    fact = app.claim_card_html("fact", "✓ Verified fact", "Net profit was Rs 500 crore.", "AR, FY2026")
    assert 'class="ier-claim fact"' in fact
    assert "Net profit was Rs 500 crore." in fact
    assert "Source: AR, FY2026" in fact

    warn = app.claim_card_html("warn", "⚠ Unverified", "profit was <b>900</b>", "")
    assert 'class="ier-claim warn"' in warn
    assert "<b>900</b>" not in warn and "&lt;b&gt;900&lt;/b&gt;" in warn   # escaped
    assert "Source:" not in warn                                          # no source -> omitted


def test_claim_card_info_accent_for_reported_and_opinion():
    app = _import_app_with_clean_env()
    html = app.claim_card_html("info", "Reported / opinion", "Analysts expect growth.", "Mint")
    assert 'class="ier-claim info"' in html
