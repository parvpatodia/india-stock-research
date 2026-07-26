"""W7 visual redesign (increment 2) smoke: the premium look must render cleanly and, above all,
must not break the app or drop a feature. Driven through Streamlit's AppTest (no LLM, no network
mocked beyond what the sample-portfolio path already does in test_app_auth).

These guard the load-bearing invariants of a styling change: the app still renders with no
exception and no error element, all four tabs still exist, and the new visual elements (hero,
badge guide, premium metric tiles) are actually present -- so a future edit that silently blanks
the CSS-backed markup fails here instead of on the parents' phones."""
import os

from streamlit.testing.v1 import AppTest

_APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def _run():
    # WHY (mirrors test_app_auth): app.py's load_dotenv sets LLM_MODEL etc. into the process env;
    # snapshot + restore so running the full app here doesn't leak env into other tests.
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        at = AppTest.from_file(_APP)
        return at.run(timeout=180)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _all_markdown(at) -> str:
    return " ".join(str(getattr(m, "value", "")) for m in at.markdown)


def test_redesign_renders_without_exception_or_error():
    at = _run()
    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
    assert len(at.tabs) >= 4          # every tab still present after the restyle


def test_hero_and_onboarding_guide_render():
    md = _all_markdown(_run())
    assert "India Equity Research" in md          # product hero title
    assert "Research only" in md or "research only" in md  # the load-bearing promise, up front
    # the 30-second guide teaches the trust badges a parent must understand
    assert "Verified fact" in md
    assert "Reported, not verified" in md


def test_premium_metric_tiles_render_not_the_fallback():
    # The custom tiles carry the class `ier-metric`; its presence proves the styled path ran (not
    # the native-st.metric fallback), and that the four summary numbers are shown as tiles.
    md = _all_markdown(_run())
    assert "ier-metric" in md
    assert "Invested" in md and "Market value" in md
    assert "Profit / loss" in md and "Holdings priced" in md
