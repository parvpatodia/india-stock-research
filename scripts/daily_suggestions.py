"""CLI entry for the daily engine: this is the PRODUCTION daily refresh.

Forces a refresh: research holdings (+ optional 'Watchlist' tab), write the 'Today' tab, push
ntfy. WHY this (the owner's Mac), not the app: Screener/Cloudflare rate-limits a datacenter IP, so
the Streamlit Cloud app can NOT cross-verify against Screener -- it only DISPLAYS the 'Today' tab
the Mac computed (see app.py's shortlist section). This CLI runs on the owner's Mac (a residential
IP that CAN reach Screener) on a daily schedule, so it gets FULL cross-verification. Shared logic
lives in src/analysis/daily_engine.py.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.analysis.daily_engine import (  # noqa: E402
    TODAY_HEADER,
    picks_to_rows,
    push_ntfy,
    research_and_rank,
)
from src.portfolio.loader import normalize_symbol  # noqa: E402


def _watchlist_symbols(gateway) -> list[str]:
    """Symbols from the optional 'Watchlist' tab, normalized the SAME way holdings are.

    WHY (real money, daily-shortlist completeness): read_holdings already runs holdings through
    normalize_symbol, but the Watchlist tab's symbols were only upper-cased -- so an entry typed with
    an exchange prefix ('NSE:RELIANCE'/'BSE:500325') or an NSE series tag ('INFY-EQ') failed the
    yfinance lookup, came back single-source, and silently dropped out of the shortlist, unlike an
    identically typed holding. normalize_symbol strips .NS/.BO/NSE:/BSE:/-EQ. A missing/unreadable
    tab -> [] (the Watchlist is optional; never fatal to the daily run).
    """
    try:
        rows = gateway.read("Watchlist")
    except Exception:
        return []
    out: list[str] = []
    for r in rows:
        sym = normalize_symbol(r.get("Symbol") or r.get("symbol") or "")
        if sym and sym != "NAN":
            out.append(sym)
    return out


def main() -> None:
    # WHY load .env HERE, not at import: launchd doesn't inherit the shell env, so the secrets live in
    # the gitignored .env. Loading it at IMPORT time would mutate os.environ for anything that merely
    # imports this module (e.g. a unit test of _watchlist_symbols), polluting other tests' env -- which
    # is exactly what happened. .env belongs to the script RUN, not the import.
    load_dotenv(_ROOT / ".env")
    cap = float(os.environ.get("POSITION_CAP", "0.25"))

    from src.data.sheets_backend import AppsScriptGateway, read_holdings
    from src.data.yfinance_provider import YFinanceProvider
    from src.portfolio.analysis import analyze_portfolio

    gateway = AppsScriptGateway(os.environ["APPS_SCRIPT_URL"], os.environ["APPS_SCRIPT_TOKEN"])
    holdings = read_holdings(gateway)
    # holdings are already normalized by read_holdings; normalize the optional Watchlist the same way
    # so an exchange-prefixed/-EQ entry there isn't silently dropped (see _watchlist_symbols).
    symbols = [h.symbol for h in holdings] + _watchlist_symbols(gateway)

    prices = YFinanceProvider().current_prices([h.symbol for h in holdings])
    analysis = analyze_portfolio(holdings, prices)
    value_by = {p.symbol: p.market_value for p in analysis.positions}

    skipped: list[str] = []
    picks = research_and_rank(symbols, value_by, analysis.total_value, cap, skipped=skipped)
    rows = picks_to_rows(picks, datetime.date.today().isoformat())
    gateway.write("Today", TODAY_HEADER, rows)
    push_ntfy(os.environ.get("NTFY_TOPIC", ""), picks)
    print(f"wrote {len(rows)} picks")
    if skipped:
        print(f"skipped {len(skipped)} symbol(s):")
        for reason in skipped:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
