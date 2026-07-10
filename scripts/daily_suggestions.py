"""CLI entry for the daily engine (manual GitHub dispatch / local runs).

Forces a refresh: research holdings (+ optional 'Watchlist' tab), write the 'Today' tab, push
ntfy. NOTE: in production the APP drives the daily refresh, because Streamlit Cloud can reach
Screener but a scheduler's datacenter IP is blocked (so a cron run comes back single-source/thin).
Shared logic lives in src/analysis/daily_engine.py.
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

# WHY: launchd does not inherit the shell env, so load the gitignored .env for the secrets.
load_dotenv(_ROOT / ".env")
CAP = float(os.environ.get("POSITION_CAP", "0.25"))


def main() -> None:
    from src.data.sheets_backend import AppsScriptGateway, read_holdings
    from src.data.yfinance_provider import YFinanceProvider
    from src.portfolio.analysis import analyze_portfolio

    gateway = AppsScriptGateway(os.environ["APPS_SCRIPT_URL"], os.environ["APPS_SCRIPT_TOKEN"])
    holdings = read_holdings(gateway)
    symbols = [h.symbol for h in holdings]
    try:
        symbols += [str(r.get("Symbol") or r.get("symbol") or "").strip().upper()
                    for r in gateway.read("Watchlist")]
        symbols = [s for s in symbols if s]
    except Exception:
        pass

    prices = YFinanceProvider().current_prices([h.symbol for h in holdings])
    analysis = analyze_portfolio(holdings, prices)
    value_by = {p.symbol: p.market_value for p in analysis.positions}

    skipped: list[str] = []
    picks = research_and_rank(symbols, value_by, analysis.total_value, CAP, skipped=skipped)
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
