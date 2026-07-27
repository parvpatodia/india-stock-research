"""The premium holdings-table markup (app.holdings_table_html) — the color-coded portfolio book.

Pure builder, so it's exercised by importing app directly (same env-safe pattern as
test_app_freshness_banner.py). Locks the real-money-visible behaviour: gain/loss coloring, a
never-fabricated P&L for a zero-cost lot, integer quantities, precise per-share prices, HTML
escaping, and the mobile hide-sm columns.
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


def _pos(symbol="RELIANCE", sector="Energy", quantity=48, avg_cost=1270.0,
         current_price=1435.70, market_value=68913.6, pnl_pct=13.05, weight=0.5):
    return SimpleNamespace(symbol=symbol, sector=sector, quantity=quantity, avg_cost=avg_cost,
                           current_price=current_price, market_value=market_value,
                           pnl_pct=pnl_pct, weight=weight)


def test_gain_and_loss_are_color_coded():
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([_pos(symbol="WIN", pnl_pct=45.8, weight=0.6),
                                    _pos(symbol="LOSE", pnl_pct=-41.6, market_value=1000.0,
                                         weight=0.1)])
    assert 'class="pl gain">+45.80%' in html
    assert 'class="pl loss">-41.60%' in html


def test_zero_cost_lot_shows_em_dash_never_a_fake_zero():
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([_pos(pnl_pct=None)])
    assert "—" in html
    assert "0.00%" not in html


def test_quantity_is_integer_and_price_keeps_paise():
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([_pos(quantity=18100, avg_cost=15.62, current_price=23.05)])
    assert ">18,100<" in html          # integer, not 18,100.00
    assert "₹15.62" in html            # paise kept on a low-priced stock
    assert "₹23.05" in html


def test_symbol_and_sector_are_escaped():
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([_pos(symbol="A<b>X", sector="M&M <i>")])
    assert "<b>X" not in html and "A&lt;b&gt;X" in html
    assert "M&amp;M" in html


def test_sorted_by_market_value_desc_and_weight_bar_scales_to_max():
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([
        _pos(symbol="SMALL", market_value=1000.0, weight=0.1),
        _pos(symbol="BIG", market_value=9000.0, weight=0.9),
    ])
    assert html.index("BIG") < html.index("SMALL")     # bigger position first
    assert 'width:100%' in html                        # the max-weight holding fills the bar


def test_secondary_columns_are_hidden_on_mobile():
    # Qty / Avg cost / Price / Weight carry hide-sm so only Holding/Value/P&L show at 375px.
    app = _import_app_with_clean_env()
    html = app.holdings_table_html([_pos()])
    assert html.count("hide-sm") >= 8                  # 4 header + 4 body cells
