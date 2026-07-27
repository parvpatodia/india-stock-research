"""Evidence figures table (app.figures_table_html) + Invest allocation table (app.allocation_table_html).

Pure builders, exercised by importing app directly (same env-safe pattern as the other app-fn
tests). Locks the trust-signal coloring (verified/single-source/conflict), share-bar scaling, and
HTML escaping.
"""
import os
from types import SimpleNamespace


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


def _row(figure, status, value="₹500 crore", period="FY2026", sources="yfinance, screener"):
    return {"Figure": figure, "Status": status, "Value": value, "Period": period,
            "Sources": sources}


def test_figures_table_color_codes_verification_status():
    app = _import_app_with_clean_env()
    html = app.figures_table_html([
        _row("Net profit", "verified"),
        _row("Revenue", "single_source"),
        _row("Debt", "conflict"),
    ])
    assert 'class="ier-chip g">Verified' in html       # cross-verified -> green
    assert 'class="ier-chip o">1 source' in html       # single source -> amber
    assert 'class="ier-chip r">Conflict' in html        # sources disagree -> red


def test_figures_table_escapes_and_handles_unknown_status_and_empty():
    app = _import_app_with_clean_env()
    html = app.figures_table_html([_row("P&L <b>x", "weird", value="<i>25</i>%")])
    assert "<b>x" not in html and "P&amp;L &lt;b&gt;x" in html
    assert "<i>25</i>%" not in html
    assert 'class="ier-chip n">weird' in html          # unknown status -> neutral chip
    assert "<tbody></tbody>" in app.figures_table_html([])   # empty rows -> empty body, no crash


def test_allocation_table_renders_amounts_and_scales_the_share_bar():
    app = _import_app_with_clean_env()
    html = app.allocation_table_html([
        SimpleNamespace(symbol="RELIANCE", amount=300000.0),
        SimpleNamespace(symbol="INFY", amount=100000.0),
    ])
    assert "RELIANCE" in html and "INFY" in html
    assert "₹3,00,000" in html and "₹1,00,000" in html   # Indian-grouped amounts
    assert "width:100%" in html                          # largest allocation fills the bar
    assert html.index("RELIANCE") < html.index("INFY")   # input order preserved


def test_allocation_table_escapes_symbol_and_handles_empty():
    app = _import_app_with_clean_env()
    html = app.allocation_table_html([SimpleNamespace(symbol="A<b>X", amount=1000.0)])
    assert "<b>X" not in html and "A&lt;b&gt;X" in html
    assert "<tbody></tbody>" in app.allocation_table_html([])   # empty -> empty body, no crash
