"""The daily-suggestions CLI is the PRODUCTION daily refresh (it runs on the owner's Mac, a
residential IP that can reach Screener; the Streamlit app only DISPLAYS the result). So its symbol
handling matters for the real shortlist, not just a dev convenience."""


def test_watchlist_symbols_are_normalized_like_holdings():
    # WHY (real money, daily shortlist completeness): read_holdings already runs holdings through
    # normalize_symbol, but the optional 'Watchlist' tab's symbols were only .strip().upper() -- so an
    # entry typed with an exchange prefix ('NSE:RELIANCE'/'BSE:500325') or NSE series tag ('INFY-EQ')
    # failed the yfinance lookup, came back single-source, and silently dropped out of the daily
    # shortlist, unlike an identically typed holding. Normalize consistently (strip .NS/.BO/NSE:/BSE:/-EQ).
    from scripts.daily_suggestions import _watchlist_symbols

    class Gateway:
        def read(self, tab):
            assert tab == "Watchlist"
            return [{"Symbol": "NSE:RELIANCE"}, {"symbol": "INFY-EQ"}, {"Symbol": "TCS.NS"},
                    {"Symbol": ""}, {"Symbol": "  hdfcbank  "}, {"Symbol": "BSE:500325"}]

    assert _watchlist_symbols(Gateway()) == ["RELIANCE", "INFY", "TCS", "HDFCBANK", "500325"]


def test_watchlist_missing_or_unreadable_tab_is_not_fatal():
    from scripts.daily_suggestions import _watchlist_symbols

    class Missing:
        def read(self, tab):
            raise RuntimeError("no Watchlist tab")

    assert _watchlist_symbols(Missing()) == []      # missing/unreadable tab -> [], never a crash
