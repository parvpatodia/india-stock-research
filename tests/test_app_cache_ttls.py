"""Regression guard for a real, live-verified bug (see app.py's get_screener_source WHY
comment): ScreenerFigureSource memoizes each symbol's fetched HTML internally, forever, on the
instance. get_screener_source() is itself an @st.cache_resource singleton, so ITS ttl controls
how long that internal memoization survives; fetch_promoter_trend()'s own @st.cache_data ttl is
just a facade on top unless the two stay coupled. Live-verified: with the resource ttl missing,
3 calls to the same symbol across simulated hours produced exactly 1 real fetch, silently
defeating fetch_promoter_trend's freshness guarantee. A future edit to either ttl in isolation
would reintroduce that bug with no other test able to catch it.
"""
import os


def _import_app_with_clean_env():
    # WHY: importing app.py runs load_dotenv, which would set LLM_MODEL/LLM_API_BASE from this
    # repo's .env into the process for the rest of the pytest session (module imports are cached
    # after the first one) -- snapshot/restore the same way test_app_auth.py does around each run.
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        import app
        return app
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_screener_source_and_promoter_trend_ttls_stay_coupled():
    app = _import_app_with_clean_env()
    assert app.get_screener_source._info.ttl == app.fetch_promoter_trend._info.ttl


def test_screener_source_and_cash_conversion_trend_ttls_stay_coupled():
    # WHY: fetch_cash_conversion_trend shares the exact same ScreenerFigureSource singleton and
    # internal-memoization shape as fetch_promoter_trend above, so it is exposed to the identical
    # bug if its ttl is ever changed without updating get_screener_source's to match.
    app = _import_app_with_clean_env()
    assert app.get_screener_source._info.ttl == app.fetch_cash_conversion_trend._info.ttl


def test_screener_source_and_other_income_share_ttls_stay_coupled():
    # WHY: fetch_other_income_share shares the same ScreenerFigureSource singleton and internal-
    # memoization shape as the other two Screener-derived cached functions above.
    app = _import_app_with_clean_env()
    assert app.get_screener_source._info.ttl == app.fetch_other_income_share._info.ttl
