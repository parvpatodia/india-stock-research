"""_secret trims whitespace from STRING secrets (blank -> absent) so a padded URL / token / sheet
key / model doesn't silently fail its fetch / lookup -- the config-string-hygiene class behind the
app_password lockout (a351684) and the LLM_MODEL Ask-tab break (2826d84). Non-strings (the
service-account dict, a bool) pass through unchanged."""
import os


def _app():
    # env-safe import (app.py touches os.environ at import); snapshot + restore so this never leaks
    # into other tests, matching test_pdf_report / test_app_plain_summary.
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        import app
        return app
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_clean_secret_trims_strings_and_passes_non_strings_through():
    _clean_secret = _app()._clean_secret
    assert _clean_secret("https://x.csv ", None) == "https://x.csv"       # trailing space trimmed
    assert _clean_secret("  ntfy-topic\n", None) == "ntfy-topic"
    assert _clean_secret("   ", None) is None                              # whitespace-only -> absent (default)
    assert _clean_secret("", None) is None
    assert _clean_secret("false ", False) == "false"                      # trimmed; parse_demo_enabled handles it
    # non-strings pass through unchanged
    assert _clean_secret(True, False) is True
    assert _clean_secret({"client_email": "x"}, None) == {"client_email": "x"}
    assert _clean_secret(None, None) is None
