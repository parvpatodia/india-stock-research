from src.data.figure_sources import FRAMEWORK_FIGURES
from src.data.screener_source import ScreenerFigureSource, parse_screener_figures

FIXTURE = """
<div><ul><li>Stock P/E <span>20.5</span></li></ul></div>
<table><thead><tr><th></th><th>Mar 2023</th><th>Mar 2024</th></tr></thead><tbody>
<tr><td>Sales +</td><td>900000</td><td>950000</td></tr>
<tr><td>Operating Profit</td><td>150000</td><td>160000</td></tr>
<tr><td>Interest</td><td>5000</td><td>5100</td></tr>
<tr><td>Profit before tax</td><td>100000</td><td>105000</td></tr>
<tr><td>Net Profit +</td><td>79000</td><td>80000</td></tr>
</tbody></table>
<table><thead><tr><th></th><th>Mar 2023</th><th>Mar 2024</th></tr></thead><tbody>
<tr><td>Equity Capital</td><td>6000</td><td>6500</td></tr>
<tr><td>Reserves</td><td>190000</td><td>200000</td></tr>
<tr><td>Borrowings +</td><td>30000</td><td>32000</td></tr>
</tbody></table>
<table><thead><tr><th></th><th>Mar 2023</th><th>Mar 2024</th></tr></thead><tbody>
<tr><td>Cash from Operating Activity +</td><td>85000</td><td>88000</td></tr>
</tbody></table>
"""

CR = 1e7


def test_parse_screener_figures():
    figs = parse_screener_figures(FIXTURE)
    assert figs["current_pe"] == 20.5
    assert figs["net_profit"] == 80000 * CR            # latest year, crore -> absolute
    assert figs["operating_cash_flow"] == 88000 * CR
    assert figs["total_debt"] == 32000 * CR
    assert figs["equity"] == (6500 + 200000) * CR       # equity capital + reserves
    assert figs["interest_expense"] == 5100 * CR
    assert figs["ebit"] == (105000 + 5100) * CR         # PBT + interest


def test_parse_empty_html_all_none():
    figs = parse_screener_figures("<html><body>no tables here</body></html>")
    assert all(figs[name] is None for name in FRAMEWORK_FIGURES)


def test_source_with_injected_fetcher():
    src = ScreenerFigureSource(fetcher=lambda symbol: FIXTURE)
    figs = src.figures("ANYTHING")
    assert figs["net_profit"] == 80000 * CR


def test_screener_fetches_page_once_per_symbol():
    # WHY (regression): the batch called figures()+figures_by_year() repeatedly, hitting Screener
    # ~3x per stock and tripping its rate limit. The page must be memoized per symbol.
    calls = []

    def counting_fetcher(symbol):
        calls.append(symbol)
        return FIXTURE

    src = ScreenerFigureSource(fetcher=counting_fetcher)
    src.figures("BLS")
    src.figures_by_year("BLS")
    src.figures("BLS")
    assert calls == ["BLS"]                       # one fetch despite three calls
    src.figures("TCS")
    assert calls == ["BLS", "TCS"]                # a different symbol fetches once more


def test_source_fetch_failure_returns_all_none():
    src = ScreenerFigureSource(fetcher=lambda symbol: None)
    assert all(v is None for v in src.figures("X").values())
