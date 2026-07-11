"""Daily long-term suggestions engine (shared by the app and the CLI script).

Researches holdings (+ watchlist) with the cross-verified pipeline, ranks long-term-fit names,
and refreshes a 'Today' tab in the Sheet at most once per day. Runs from wherever it's called: in
production it's driven by the app (Streamlit Cloud can reach Screener; a scheduler's datacenter IP
can't), so the refresh happens on the first visit each day, not overnight.
"""
from __future__ import annotations

import datetime
import urllib.request

from .sizing import position_sizing, stance_from_verdict, verdict_strength
from .suggestions import Candidate, rank_picks

TODAY_HEADER = ["date", "symbol", "stance", "score", "reason"]


def candidate_from_report(symbol: str, report, held_value: float, total_value: float,
                          cap_pct: float) -> Candidate:
    from ..research.report import QualityTier, ValuationTier
    v = report.verdict
    sizing = position_sizing(held_value, total_value or 1.0, cap_pct)
    return Candidate(
        symbol=symbol,
        stance=stance_from_verdict(v),
        quality_strong=(v is not None and v.quality == QualityTier.STRONG),
        valuation_cheap=(v is not None and v.valuation == ValuationTier.CHEAP),
        has_room=sizing.headroom > 0,
        trend_improving=report.trend_improving,   # structured signal, not parsed from prose
        strength=verdict_strength(v),   # orders names within the same flag band by conviction
        reason=(report.insights[0] if report.insights else ""),
    )


def _default_sources():
    from ..data.figure_sources import YFinanceFigureSource
    from ..data.screener_source import ScreenerFigureSource
    return [YFinanceFigureSource(), ScreenerFigureSource()]


def research_and_rank(symbols: list[str], value_by_symbol: dict[str, float], total_value: float,
                      cap_pct: float, sources_factory=_default_sources,
                      throttle_seconds: float = 1.5,
                      skipped: list[str] | None = None):
    """Research each symbol, build candidates, rank. One failure per symbol is skipped.

    WHY (throttle + shared sources): Screener throttles a rapid burst of requests from a datacenter
    IP. Build the sources ONCE and reuse them so the Screener source memoizes each page per symbol
    (was ~3 fetches/symbol), and pace the loop so the batch isn't a burst. throttle_seconds=0 in
    tests to avoid sleeping.

    WHY (skipped, optional): this engine is also driven headless by scripts/daily_suggestions.py
    (a launchd background job with no interactive UI). A mistyped watchlist/holding symbol already
    can't produce a wrong PICK (INSUFFICIENT_DATA is excluded by rank_picks), but without a
    diagnostic it's silently dropped forever with no signal in the log the operator reviews. When
    given a list, this appends a human-readable reason per skipped symbol; None (default) preserves
    prior behavior for existing callers.
    """
    import time

    from ..pipeline import build_report_for_symbol
    sources = sources_factory()
    candidates = []
    for i, symbol in enumerate(dict.fromkeys(symbols)):
        if i and throttle_seconds:
            time.sleep(throttle_seconds)
        try:
            report = build_report_for_symbol(symbol, sources)
        except Exception as exc:
            if skipped is not None:
                skipped.append(f"{symbol}: fetch failed ({exc})")
            continue
        if report.no_data_found:
            if skipped is not None:
                skipped.append(f"{symbol}: no data from any source (check the exact ticker symbol)")
            continue
        candidates.append(candidate_from_report(
            symbol, report, value_by_symbol.get(symbol, 0.0), total_value, cap_pct))
    return rank_picks(candidates)


def _ntfy_body(picks) -> str:
    """The push-notification text for the daily suggestions. WHY (real money, honesty; the hard
    "never a buy/sell call" invariant): this pushes to a parent's phone, acted on at a glance, so it
    must carry the SAME non-advice framing the app applies everywhere else (STANCE_CAVEAT: "not a buy
    or sell call"). Calling them "picks" reads as a buy tip; frame them as a research shortlist the
    evidence leans toward, explicitly not advice. Per-name stance is still shown."""
    lines = "\n".join(f"{i + 1}. {p.symbol} - {p.stance.value}" for i, p in enumerate(picks[:5]))
    return ("Names your research surfaced today, within your per-stock cap -- to research, "
            "NOT buy/sell advice:\n" + lines)


def push_ntfy(topic: str, picks) -> None:
    if not topic or not picks:
        return
    body = _ntfy_body(picks)
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
            headers={"Title": "India Equity Research", "Tags": "chart_with_upwards_trend"})
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass


def picks_to_rows(picks, today: str) -> list[dict]:
    # WHY: floor, don't round. The score carries a sub-1 conviction fraction used only to order
    # ties; rounding a 6.9 to "7" would show a count above the real max of 6 flags. int() = floor
    # for the non-negative score, so the displayed integer stays the true signal count.
    return [{"date": today, "symbol": p.symbol, "stance": p.stance.value,
             "score": f"{int(p.score)}", "reason": p.reason} for p in picks]


def refresh_today_if_stale(gateway, symbols: list[str], value_by_symbol: dict[str, float],
                           total_value: float, cap_pct: float, ntfy_topic: str = "",
                           force: bool = False, today: str | None = None,
                           researcher=research_and_rank,
                           pusher=push_ntfy) -> tuple[list[dict], bool]:
    """Return (today_rows, refreshed). If the Sheet's Today tab is already dated today (and not
    force), return it unchanged (fast). Otherwise research, write the Today tab, push ntfy, and
    return the new rows. force=True always recomputes (the manual 'Refresh' button).
    researcher/pusher are injectable so this is tested without network."""
    today = today or datetime.date.today().isoformat()
    existing = gateway.read("Today")
    if not force and existing and str(existing[0].get("date")) == today:
        return existing, False
    picks = researcher(symbols, value_by_symbol, total_value, cap_pct)
    rows = picks_to_rows(picks, today)
    gateway.write("Today", TODAY_HEADER, rows)
    if ntfy_topic:
        pusher(ntfy_topic, picks)
    return rows, True
