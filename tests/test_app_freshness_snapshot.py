"""H7 Research-tab freshness SNAPSHOT banner (the Cloud-visible half of the W1 engine).

The pure helper `freshness_snapshot_line` + the degrade-safe reader `load_freshness_snapshot` live
in app.py, exercised by importing the module directly (same env-safe pattern as
test_app_freshness_banner.py). This is what lets the DEPLOYED app show real freshness even though
Streamlit Cloud can't run the scheduler and NSE/BSE block its IP: a Mac-side run publishes the
snapshot to the Sheets backend and the app reads it here.
"""
import os
from datetime import date, timedelta


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


def _snap(app, **kw):
    from src.freshness.snapshot import SymbolSnapshot
    base = dict(symbol="RELIANCE", checked_at="2026-07-26", window_days=120,
                news_recent=8, announcements_recent=42,
                annual_report_fy=2026, annual_report_as_of="2026-03-31")
    base.update(kw)
    return SymbolSnapshot(**base)


_TODAY = "2026-07-26"


def test_line_summarizes_refresh_counts_and_annual_report():
    app = _import_app_with_clean_env()
    line, stale = app.freshness_snapshot_line(_snap(app), _TODAY)
    assert stale is False
    assert "2026-07-26" in line                 # the real refresh date, never fabricated
    assert "8 news" in line and "42 filings" in line
    assert "120 days" in line
    assert "FY2026" in line and "2026-03-31" in line


def test_line_flags_an_overdue_automated_refresh():
    # WHY (real money, honesty): if the Mac-side scheduler stops, the parents must SEE that the data
    # is going stale rather than trust a silently frozen snapshot as current.
    app = _import_app_with_clean_env()
    stale_date = (date(2026, 7, 26) - timedelta(days=10)).isoformat()
    _, stale = app.freshness_snapshot_line(_snap(app, checked_at=stale_date), _TODAY)
    assert stale is True


def test_none_snapshot_renders_nothing():
    app = _import_app_with_clean_env()
    assert app.freshness_snapshot_line(None, _TODAY) is None


def test_missing_checked_at_renders_nothing_not_a_fake_date():
    app = _import_app_with_clean_env()
    assert app.freshness_snapshot_line(_snap(app, checked_at=""), _TODAY) is None


def test_unparseable_checked_at_renders_nothing():
    app = _import_app_with_clean_env()
    assert app.freshness_snapshot_line(_snap(app, checked_at="not-a-date"), _TODAY) is None


def test_line_drops_missing_pieces_gracefully():
    # no AR resolved + zero counts -> still shows the refresh date, no "FY-1", no empty "0 news"
    app = _import_app_with_clean_env()
    line, _ = app.freshness_snapshot_line(
        _snap(app, news_recent=0, announcements_recent=0, annual_report_fy=-1,
              annual_report_as_of=""), _TODAY)
    assert "2026-07-26" in line
    assert "FY" not in line
    assert "news" not in line and "filings" not in line


def test_load_freshness_snapshot_degrades_to_empty_when_gateway_raises(monkeypatch):
    # WHY (degrade, real money): the freshness banner is best-effort; a broken/unconfigured Sheet
    # backend must yield {} so the app shows nothing, never a crash on the parents' page.
    app = _import_app_with_clean_env()

    class _Boom:
        def read(self, tab):
            raise RuntimeError("sheet unreachable")

    monkeypatch.setattr(app, "get_gateway", lambda: _Boom())
    app.load_freshness_snapshot.clear()  # drop any cached value from a prior test
    assert app.load_freshness_snapshot() == {}


def test_load_freshness_snapshot_parses_rows(monkeypatch):
    app = _import_app_with_clean_env()
    from src.freshness.snapshot import FRESHNESS_TAB, SymbolSnapshot

    rows = [SymbolSnapshot("RELIANCE", "2026-07-26", 120, 8, 42, 2026, "2026-03-31").as_row()]

    class _GW:
        def read(self, tab):
            return rows if tab == FRESHNESS_TAB else []

    monkeypatch.setattr(app, "get_gateway", lambda: _GW())
    app.load_freshness_snapshot.clear()
    snaps = app.load_freshness_snapshot()
    assert "RELIANCE" in snaps
    assert snaps["RELIANCE"].news_recent == 8


def _all_text(at) -> str:
    parts = []
    for attr in ("markdown", "caption", "warning", "info", "success", "text"):
        try:
            for el in getattr(at, attr):
                parts.append(str(getattr(el, "value", "")))
        except Exception:
            pass
    return " ".join(parts)


def test_research_tab_renders_the_snapshot_banner_end_to_end(monkeypatch):
    # WHY: prove the WIRING through Streamlit's own runtime -- a report for a symbol whose freshness
    # snapshot is published renders the 🔄 banner ON CLOUD -- with the Sheets read INJECTED at the
    # shared src-level gateway so the test is offline/deterministic. Guards against a future edit
    # silently dropping the Cloud-visible freshness the parents rely on.
    from datetime import date

    from streamlit.testing.v1 import AppTest

    import src.data.sheets_backend as sb
    from src.freshness.snapshot import FRESHNESS_TAB, SymbolSnapshot
    from src.research.report import (Confidence, Leaning, QualityTier, Report, ValuationTier,
                                     Verdict)
    from src.research.verification import SourcedValue, verify_figure

    today = date.today().isoformat()   # a fresh snapshot (refreshed today) -> no overdue warning
    rows = [SymbolSnapshot("RELIANCE", today, 120, 8, 42, 2026, "2026-03-31").as_row()]
    orig_read = sb.LocalJsonGateway.read

    def fake_read(self, tab):
        return rows if tab == FRESHNESS_TAB else orig_read(self, tab)

    monkeypatch.setattr(sb.LocalJsonGateway, "read", fake_read)

    verdict = Verdict(ValuationTier.FAIR, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                      Confidence.MEDIUM, reasons=("ROCE steady",))
    fig = verify_figure("net_profit", [SourcedValue(79000.0, "yfinance"),
                                       SourcedValue(79010.0, "screener")])
    report = Report(company="RELIANCE", figures=(fig,), verdict=verdict)
    key = "RELIANCE (live/yfinance + screener)"

    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    saved = dict(os.environ)
    try:
        for k in ("LLM_MODEL", "LLM_API_KEY", "LLM_API_BASE", "GROQ_API_KEY"):
            os.environ.pop(k, None)
        # load_freshness_snapshot is @st.cache_data keyed by qualname+code, so an EARLIER AppTest
        # that rendered the Research tab (e.g. the H6 banner test) may have cached an empty {} under
        # the same key. Clear the caches so THIS run reads through our injected gateway, not a stale
        # cross-test {} (production only ever caches a real read, ttl-bounded).
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
        at = AppTest.from_file(app_path)
        at.session_state["reports"] = {key: report}
        at.session_state["active_report"] = key
        at.run(timeout=180)
    finally:
        os.environ.clear()
        os.environ.update(saved)

    assert len(at.exception) == 0, [e.value for e in at.exception]
    text = _all_text(at)
    assert "Data last refreshed" in text
    assert "8 news" in text and "42 filings" in text
    assert "FY2026" in text
