from src.data.figure_sources import FRAMEWORK_FIGURES
from src.data.screener_source import (
    ScreenerFigureSource,
    parse_promoter_holding_series,
    parse_screener_figures,
    promoter_holding_trend_point,
)

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


# A quarterly results table (multi-month columns straddling a calendar year) placed BEFORE the
# annual P&L, sharing its row labels — exactly how Screener lays the page out.
QUARTERLY_THEN_ANNUAL = """
<div><ul><li>Stock P/E <span>20.5</span></li></ul></div>
<table><thead><tr><th></th><th>Dec 2023</th><th>Mar 2024</th><th>Jun 2024</th></tr></thead><tbody>
<tr><td>Sales +</td><td>230000</td><td>240000</td><td>250000</td></tr>
<tr><td>Operating Profit</td><td>40000</td><td>41000</td><td>42000</td></tr>
<tr><td>Net Profit +</td><td>18000</td><td>19000</td><td>20000</td></tr>
</tbody></table>
<table><thead><tr><th></th><th>Mar 2023</th><th>Mar 2024</th></tr></thead><tbody>
<tr><td>Sales +</td><td>900000</td><td>950000</td></tr>
<tr><td>Operating Profit</td><td>150000</td><td>160000</td></tr>
<tr><td>Net Profit +</td><td>79000</td><td>80000</td></tr>
</tbody></table>
"""


def test_quarterly_table_not_misread_as_annual():
    # WHY (resilience): Screener's quarterly table precedes the annual P&L and shares its labels.
    # A quarter straddling a calendar year must NOT be picked as annual, or quarterly figures get
    # cross-checked against yfinance's annual -> silent conflict -> good annual data withheld.
    figs = parse_screener_figures(QUARTERLY_THEN_ANNUAL)
    assert figs["net_profit"] == 80000 * CR       # ANNUAL value, not 20000 (a single quarter)
    assert figs["revenue"] == 950000 * CR          # annual sales, not a quarter


BARE_YEAR_ANNUAL = """
<div><ul><li>Stock P/E <span>18.0</span></li></ul></div>
<table><thead><tr><th></th><th>2023</th><th>2024</th></tr></thead><tbody>
<tr><td>Sales +</td><td>500000</td><td>550000</td></tr>
<tr><td>Operating Profit</td><td>90000</td><td>95000</td></tr>
<tr><td>Net Profit +</td><td>60000</td><td>65000</td></tr>
</tbody></table>
"""


def test_bare_year_headers_are_recognized_as_annual():
    # WHY (resilience): if Screener ever renders year-only headers (no 'Mon'), the month-consistency
    # check must still treat them as annual, or the entire second source is silently lost and every
    # figure drops to single-source. Quarterly tables always carry months, so this stays safe.
    figs = parse_screener_figures(BARE_YEAR_ANNUAL)
    assert figs["net_profit"] == 65000 * CR
    assert figs["revenue"] == 550000 * CR


# Real structure verified live against screener.in/company/RELIANCE/consolidated/: a quarterly
# "Shareholding Pattern" table with Promoters/FIIs/DIIs/Government/Public rows (each label carries
# a Screener "+" suffix like the P&L/balance rows), percentages as "50.39%" strings, columns
# oldest -> newest. This table sits alongside, not instead of, the P&L/balance/cash tables above.
SHAREHOLDING_FIXTURE = FIXTURE + """
<table><thead><tr><th></th><th>Jun 2023</th><th>Sep 2023</th><th>Dec 2023</th></tr></thead><tbody>
<tr><td>Promoters +</td><td>50.39%</td><td>50.27%</td><td>50.30%</td></tr>
<tr><td>FIIs +</td><td>22.55%</td><td>22.60%</td><td>22.13%</td></tr>
<tr><td>DIIs +</td><td>16.13%</td><td>15.99%</td><td>16.59%</td></tr>
<tr><td>Government +</td><td>0.17%</td><td>0.17%</td><td>0.18%</td></tr>
<tr><td>Public +</td><td>10.76%</td><td>10.98%</td><td>10.80%</td></tr>
<tr><td>No. of Shareholders</td><td>3506867</td><td>3698648</td><td>3613814</td></tr>
</tbody></table>
"""


def test_parse_promoter_holding_series_extracts_percentages_in_order():
    series = parse_promoter_holding_series(SHAREHOLDING_FIXTURE)
    assert series == {"Jun 2023": 50.39, "Sep 2023": 50.27, "Dec 2023": 50.30}


def test_parse_promoter_holding_does_not_confuse_the_pnl_borrowings_rows():
    # WHY: the P&L/balance tables in the same page must not be mistaken for the shareholding
    # table (neither has both a 'promoters' and a 'public' row).
    assert parse_promoter_holding_series(FIXTURE) == {}


def test_promoter_holding_trend_point_increasing():
    point = promoter_holding_trend_point({"Mar 2017": 46.32, "Mar 2026": 50.00})
    assert point is not None
    assert "increased" in point
    assert "46.3%" in point and "50.0%" in point
    assert "not cross-verified" in point.lower() or "screener only" in point.lower()


def test_promoter_holding_trend_point_decreasing():
    point = promoter_holding_trend_point({"Mar 2020": 55.0, "Mar 2024": 48.0})
    assert point is not None and "decreased" in point


def test_promoter_holding_trend_point_decrease_wording_does_not_presume_alarm():
    # WHY (real money, honesty): live-verified against HDFC Bank's real Screener data, promoter
    # holding steps from 25.59% to EXACTLY 0.00% at Mar 2024 -- not a parsing bug, this matches
    # the actual HDFC Ltd-HDFC Bank merger (Jul 2023), after which HDFC Bank has no designated
    # promoter. Alarmist wording ("worth watching") on a decrease would mislabel a benign,
    # well-known structural event as a red flag. The wording must stay neutral and name a
    # merger/reclassification as a real possibility, not presuppose concern.
    point = promoter_holding_trend_point({"Mar 2023": 25.59, "Mar 2024": 0.0})
    assert point is not None
    assert "worth watching" not in point.lower()
    assert "merger" in point.lower() or "reclassification" in point.lower()


def test_promoter_holding_trend_point_roughly_steady_below_threshold():
    point = promoter_holding_trend_point({"Mar 2020": 50.0, "Mar 2024": 50.2})
    assert point is not None and "steady" in point.lower()


def test_promoter_holding_trend_point_none_when_insufficient_data():
    assert promoter_holding_trend_point({}) is None
    assert promoter_holding_trend_point({"Mar 2024": 50.0}) is None   # need >=2 points


def test_screener_source_exposes_promoter_holding_trend():
    src = ScreenerFigureSource(fetcher=lambda symbol: SHAREHOLDING_FIXTURE)
    point = src.promoter_holding_trend("ANYTHING")
    assert point is not None and ("increased" in point or "decreased" in point or "steady" in point)


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
