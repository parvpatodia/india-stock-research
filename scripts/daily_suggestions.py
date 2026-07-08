"""Daily long-term suggestions engine (runs on a schedule; see .github/workflows/daily.yml).

Reads holdings (+ an optional 'Watchlist' tab) from the Sheet, researches each with the
cross-verified pipeline, ranks the long-term-fit candidates, writes a 'Today' tab back to the
Sheet, and pushes the top picks to ntfy for a free phone notification.

Env: APPS_SCRIPT_URL, APPS_SCRIPT_TOKEN (Sheet bridge); LLM_MODEL/LLM_API_KEY (optional, enables
the annual-report tiebreaker); NTFY_TOPIC (optional push); POSITION_CAP (default 0.25).
"""
from __future__ import annotations

import datetime
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.sizing import Stance, position_sizing, stance_from_verdict  # noqa: E402
from src.analysis.suggestions import Candidate, rank_picks  # noqa: E402
from src.data.figure_sources import YFinanceFigureSource  # noqa: E402
from src.data.screener_source import ScreenerFigureSource  # noqa: E402
from src.pipeline import build_report_for_symbol  # noqa: E402
from src.research.report import QualityTier, ValuationTier  # noqa: E402

CAP = float(os.environ.get("POSITION_CAP", "0.25"))
TODAY_HEADER = ["date", "symbol", "stance", "score", "reason"]


def _sources():
    return [YFinanceFigureSource(), ScreenerFigureSource()]


def candidate_from_report(symbol: str, report, held_value: float,
                          total_value: float) -> Candidate:
    """Build a ranking candidate from a researched report + the holder's position."""
    v = report.verdict
    sizing = position_sizing(held_value, total_value or 1.0, CAP)
    trend_improving = any(("growing" in i or "improving" in i) for i in report.insights)
    return Candidate(
        symbol=symbol,
        stance=stance_from_verdict(v),
        quality_strong=(v is not None and v.quality == QualityTier.STRONG),
        valuation_cheap=(v is not None and v.valuation == ValuationTier.CHEAP),
        has_room=sizing.headroom > 0,
        trend_improving=trend_improving,
        reason=(report.insights[0] if report.insights else ""),
    )


def research_and_rank(symbols: list[str], value_by_symbol: dict[str, float],
                      total_value: float, sources_factory=_sources):
    """Research each symbol and rank long-term-fit candidates. Network-heavy; one failure per
    symbol is skipped, not fatal. Returns list[RankedPick]."""
    candidates = []
    for symbol in dict.fromkeys(symbols):   # dedup, preserve order
        try:
            report = build_report_for_symbol(symbol, sources_factory())
        except Exception:
            continue
        candidates.append(candidate_from_report(
            symbol, report, value_by_symbol.get(symbol, 0.0), total_value))
    return rank_picks(candidates)


def push_ntfy(topic: str, picks) -> None:
    if not topic or not picks:
        return
    body = "Today's long-term picks (within your cap):\n" + "\n".join(
        f"{i + 1}. {p.symbol} — {p.stance.value}" for i, p in enumerate(picks[:5]))
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
            headers={"Title": "India Equity Research", "Tags": "chart_with_upwards_trend"})
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass


def main() -> None:
    from src.data.sheets_backend import AppsScriptGateway, read_holdings
    from src.portfolio.analysis import analyze_portfolio
    from src.data.yfinance_provider import YFinanceProvider

    gateway = AppsScriptGateway(os.environ["APPS_SCRIPT_URL"], os.environ["APPS_SCRIPT_TOKEN"])
    holdings = read_holdings(gateway)
    symbols = [h.symbol for h in holdings]
    try:
        watch = [str(r.get("Symbol") or r.get("symbol") or "").strip().upper()
                 for r in gateway.read("Watchlist")]
        symbols += [s for s in watch if s]
    except Exception:
        pass

    prices = YFinanceProvider().current_prices([h.symbol for h in holdings])
    analysis = analyze_portfolio(holdings, prices)
    value_by = {p.symbol: p.market_value for p in analysis.positions}

    picks = research_and_rank(symbols, value_by, analysis.total_value)

    today = datetime.date.today().isoformat()
    rows = [{"date": today, "symbol": p.symbol, "stance": p.stance.value,
             "score": f"{p.score:.0f}", "reason": p.reason} for p in picks]
    gateway.write("Today", TODAY_HEADER, rows)
    push_ntfy(os.environ.get("NTFY_TOPIC", ""), picks)
    print(f"wrote {len(rows)} picks for {today}")


if __name__ == "__main__":
    main()
