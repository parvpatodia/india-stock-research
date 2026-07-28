"""Site access: the password gate was REMOVED by the owner (2026-07-27) -- the deployed app is
shared by URL only, so it opens straight to the tabs with no prompt, even if an `app_password`
secret is present. These tests lock that in (and guard against a gate being accidentally
reintroduced). Driven through Streamlit's AppTest."""
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


def test_site_is_open_with_no_secret():
    at = _run(app_password=None)
    assert len(at.exception) == 0
    assert len(at.tabs) >= 4          # opens straight to the tabs


def test_site_is_open_even_with_a_password_secret_set():
    # the gate is removed, so a leftover app_password secret must NOT reintroduce a prompt.
    at = _run(app_password="letmein")
    assert len(at.exception) == 0
    assert len(at.tabs) >= 4          # no gate -> tabs render regardless of the secret
