"""H6 Research-tab freshness banner: the vintage + fresh/stale flag of the PRIMARY source (the
latest annual report resolved for a symbol).

The pure helper `annual_report_freshness_line` lives in app.py, so it is exercised by importing the
module directly (same env-safe pattern as test_app_ask_tab.py / test_app_visual.py). A final AppTest
drives the FULL Research-tab render with an INJECTED fake resolver, so the wiring is proven offline
and deterministic -- fetch_ar_ref -> annual_report_freshness_line -> the ⏳ banner -- without ever
touching NSE.
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


def _ref(fiscal_year, as_of):
    from src.data.nse_annual_reports import AnnualReportRef
    return AnnualReportRef(symbol="RELIANCE", url="http://x/report.pdf",
                           fiscal_year=fiscal_year, as_of=as_of)


_TODAY = "2026-07-25"


def test_fresh_annual_report_reads_fresh_with_fy_and_date():
    # WHY (real money, honesty): the most recently completed FY's report (dated its 31-March FY end)
    # must read as fresh and name the fiscal year + the date, so a parent sees WHAT drove the read.
    app = _import_app_with_clean_env()
    line, stale, unknown = app.annual_report_freshness_line(_ref(2026, "2026-03-31"), _TODAY)
    assert stale is False and unknown is False
    assert "FY2026 annual report" in line
    assert "2026-03-31" in line
    assert "fresh" in line


def test_annual_report_over_a_year_old_is_flagged_stale():
    # WHY: a report whose newest available vintage is a full extra year behind must be FLAGGED, so a
    # parent doesn't act on a headline that predates the last year of results.
    app = _import_app_with_clean_env()
    _, stale, unknown = app.annual_report_freshness_line(_ref(2025, "2025-03-31"), _TODAY)
    assert stale is True and unknown is False


def test_threshold_boundary_exactly_400_days_is_not_stale_401_is():
    # WHY: pin the chosen ~400-day annual-report threshold at the boundary (age == threshold is still
    # fresh; only strictly older is stale), so a future tweak to the constant is a deliberate change.
    app = _import_app_with_clean_env()
    today = date(2026, 7, 25)
    at_400 = (today - timedelta(days=400)).isoformat()
    at_401 = (today - timedelta(days=401)).isoformat()
    assert app.annual_report_freshness_line(_ref(2025, at_400), today.isoformat())[1] is False
    assert app.annual_report_freshness_line(_ref(2025, at_401), today.isoformat())[1] is True


def test_unresolved_ref_renders_nothing():
    # WHY (degrade): no resolved AR (NSE blocked, e.g. Streamlit Cloud) -> None, so the caller shows
    # nothing, never a fabricated date.
    app = _import_app_with_clean_env()
    assert app.annual_report_freshness_line(None, _TODAY) is None


def test_resolved_but_undated_ref_renders_nothing_not_a_fake_date():
    # WHY (real money, honesty): a ref with no usable as-of date must render NOTHING rather than an
    # awkward "date unknown" with no fiscal year -- never invent a date for the primary source.
    app = _import_app_with_clean_env()
    assert app.annual_report_freshness_line(_ref(2026, ""), _TODAY) is None


def test_unknown_fiscal_year_drops_the_fy_prefix():
    # WHY: an unparseable fiscal year (never seen from NSE, but guarded) must not print "FY-1 annual
    # report" -- it degrades to a bare "annual report".
    app = _import_app_with_clean_env()
    line, _, _ = app.annual_report_freshness_line(_ref(-1, "2026-03-31"), _TODAY)
    assert "annual report" in line
    assert "FY" not in line.split(":")[0]


def _all_text(at) -> str:
    parts = []
    for attr in ("markdown", "caption", "warning", "info", "success", "text"):
        try:
            for el in getattr(at, attr):
                parts.append(str(getattr(el, "value", "")))
        except Exception:
            pass
    return " ".join(parts)


def test_research_tab_renders_the_freshness_banner_with_an_injected_resolver(monkeypatch):
    # WHY: prove the WIRING end to end through Streamlit's own runtime -- a report for a symbol with
    # a resolvable AR renders the ⏳ banner -- with the NSE resolver INJECTED so the test is offline
    # and deterministic (never a live NSE call). Guards against a future edit silently dropping the
    # banner on the parents' live page.
    from streamlit.testing.v1 import AppTest

    import src.data.nse_annual_reports as nse
    from src.research.report import (Confidence, Leaning, QualityTier, Report, ValuationTier,
                                     Verdict)
    from src.research.verification import SourcedValue, verify_figure

    fy = date.today().year   # a FRESH latest report, dated this FY's 31-March end
    fresh_ref = _ref(fy, f"{fy}-03-31")
    monkeypatch.setattr(nse.NseAnnualReportResolver, "latest_report",
                        lambda self, symbol: fresh_ref)

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
        at = AppTest.from_file(app_path)
        at.session_state["reports"] = {key: report}
        at.session_state["active_report"] = key
        at.run(timeout=180)
    finally:
        os.environ.clear()
        os.environ.update(saved)

    assert len(at.exception) == 0, [e.value for e in at.exception]
    text = _all_text(at)
    assert "Analysis based on the" in text
    assert "annual report" in text
    assert f"{fy}-03-31" in text
