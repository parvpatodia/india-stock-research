"""The password gate: a ?key= in the URL auto-signs-in (the parents' bookmark), the bare URL is
gated, and no password configured = open (local dev). Driven through Streamlit's AppTest."""
import os

from streamlit.testing.v1 import AppTest

_APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")


def _run(app_password=None, url_key=None):
    # WHY: app.py's load_dotenv sets LLM_MODEL etc. into the process env; snapshot + restore so
    # running the full app here doesn't leak env into other tests (e.g. test_llm's no-model case).
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        at = AppTest.from_file(_APP)
        if app_password is not None:
            at.secrets["app_password"] = app_password
        if url_key is not None:
            at.query_params["key"] = url_key
        return at.run(timeout=120)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_matching_url_key_auto_authenticates():
    at = _run(app_password="letmein", url_key="letmein")
    assert len(at.exception) == 0
    assert len(at.tabs) >= 4          # signed in with no typing


def test_bare_and_wrong_url_are_gated():
    assert len(_run(app_password="letmein").tabs) == 0            # no key -> prompt
    assert len(_run(app_password="letmein", url_key="nope").tabs) == 0  # wrong key -> prompt


def test_no_password_configured_is_open():
    at = _run(app_password=None)
    assert len(at.tabs) >= 4          # local dev, open
