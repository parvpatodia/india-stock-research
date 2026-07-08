"""Deeper analysis metrics + plain-language explanations.

The ratios a chartered accountant actually looks at (ROE, ROCE, ROA, margins, asset turnover),
computed ONLY from cross-verified figures, plus a plain_points() renderer that turns the numbers
into short, everyday-language sentences a non-expert can read. Deterministic: no LLM, so there is
nothing to fabricate. Thresholds are documented heuristics an expert can tune; a missing input
(figure not cross-verified) makes a metric known=False and it simply isn't shown.
"""
from __future__ import annotations

from .framework import MetricResult

# Documented heuristic thresholds (expert-tunable).
_ROE_GOOD, _ROE_WEAK = 15.0, 8.0          # % return on shareholders' equity
_ROCE_GOOD, _ROCE_WEAK = 15.0, 10.0       # % return on capital employed
_ROA_GOOD, _ROA_WEAK = 6.0, 2.0           # % return on assets (industrials; banks differ)
_NETMARGIN_GOOD, _NETMARGIN_WEAK = 12.0, 3.0
_OPMARGIN_GOOD, _OPMARGIN_WEAK = 15.0, 5.0


def _unknown(name: str, why: str) -> MetricResult:
    return MetricResult(name, known=False, verdict="unknown", detail=why)


def _rate(value: float, good: float, weak: float) -> tuple[str, bool]:
    if value >= good:
        return "strong", False
    if value < weak:
        return "weak", True
    return "moderate", False


def return_on_equity(net_profit: float | None, equity: float | None) -> MetricResult:
    name = "Return on equity (ROE)"
    if net_profit is None or equity is None or equity <= 0:
        return _unknown(name, "net profit or (positive) shareholders' equity unavailable.")
    roe = net_profit / equity * 100
    v, concern = _rate(roe, _ROE_GOOD, _ROE_WEAK)
    return MetricResult(name, True, v,
                        f"For every ₹100 of owners' money in the business, it earns about "
                        f"₹{roe:.0f} a year (ROE {roe:.0f}%) — {v}.", concern)


def return_on_capital(ebit: float | None, equity: float | None,
                      total_debt: float | None) -> MetricResult:
    name = "Return on capital employed (ROCE)"
    if ebit is None or equity is None or total_debt is None or (equity + total_debt) <= 0:
        return _unknown(name, "operating profit, equity, or debt unavailable.")
    roce = ebit / (equity + total_debt) * 100
    v, concern = _rate(roce, _ROCE_GOOD, _ROCE_WEAK)
    return MetricResult(name, True, v,
                        f"It earns about ₹{roce:.0f} a year for every ₹100 of capital it uses "
                        f"(ROCE {roce:.0f}%) — {v}.", concern)


def return_on_assets(net_profit: float | None, total_assets: float | None) -> MetricResult:
    name = "Return on assets (ROA)"
    if net_profit is None or total_assets is None or total_assets <= 0:
        return _unknown(name, "net profit or (positive) total assets unavailable.")
    roa = net_profit / total_assets * 100
    v, concern = _rate(roa, _ROA_GOOD, _ROA_WEAK)
    return MetricResult(name, True, v,
                        f"For every ₹100 of everything it owns, it earns about ₹{roa:.1f} "
                        f"(ROA {roa:.1f}%) — {v}.", concern)


def net_margin(net_profit: float | None, revenue: float | None) -> MetricResult:
    name = "Net profit margin"
    if net_profit is None or revenue is None or revenue <= 0:
        return _unknown(name, "net profit or (positive) revenue unavailable.")
    m = net_profit / revenue * 100
    v, concern = _rate(m, _NETMARGIN_GOOD, _NETMARGIN_WEAK)
    return MetricResult(name, True, v,
                        f"It keeps about ₹{m:.0f} of final profit from every ₹100 of sales "
                        f"(net margin {m:.0f}%) — {v}.", concern)


def operating_margin(ebit: float | None, revenue: float | None) -> MetricResult:
    name = "Operating margin"
    if ebit is None or revenue is None or revenue <= 0:
        return _unknown(name, "operating profit or (positive) revenue unavailable.")
    m = ebit / revenue * 100
    v, concern = _rate(m, _OPMARGIN_GOOD, _OPMARGIN_WEAK)
    return MetricResult(name, True, v,
                        f"From every ₹100 of sales, ₹{m:.0f} is left as operating profit "
                        f"(operating margin {m:.0f}%) — {v}.", concern)


def asset_turnover(revenue: float | None, total_assets: float | None) -> MetricResult:
    name = "Asset turnover"
    if revenue is None or total_assets is None or total_assets <= 0:
        return _unknown(name, "revenue or (positive) total assets unavailable.")
    t = revenue / total_assets
    v = "high" if t >= 1.0 else "low" if t < 0.4 else "moderate"
    return MetricResult(name, True, v,
                        f"It generates ₹{t:.2f} of sales a year for every ₹1 of assets "
                        f"(asset turnover {t:.2f}x) — {v}.", concern=False)


def compute_deep_metrics(v: dict, is_bank: bool = False) -> list[MetricResult]:
    """Compute the ratio suite from a dict of cross-verified values (missing -> None). For banks,
    margins and asset turnover are skipped (bank P&L is interest income, not sales)."""
    metrics = [
        return_on_equity(v.get("net_profit"), v.get("equity")),
        return_on_assets(v.get("net_profit"), v.get("total_assets")),
    ]
    if not is_bank:
        metrics += [
            return_on_capital(v.get("ebit"), v.get("equity"), v.get("total_debt")),
            net_margin(v.get("net_profit"), v.get("revenue")),
            operating_margin(v.get("ebit"), v.get("revenue")),
            asset_turnover(v.get("revenue"), v.get("total_assets")),
        ]
    return metrics


def plain_points(v: dict, deep: list[MetricResult]) -> list[str]:
    """5-6 short, everyday-language reasons with the real numbers, for a non-expert reader.
    Covers price, cash quality, and debt from the core figures, then the ratio suite. Only
    includes points whose inputs cross-verified."""
    points: list[str] = []

    cpe, mpe = v.get("current_pe"), v.get("median_pe")
    if cpe and mpe and cpe > 0 and mpe > 0:
        ratio = cpe / mpe
        if ratio < 0.8:
            tag = f"cheaper than its usual price (about {ratio:.0%} of normal)"
        elif ratio > 1.2:
            tag = f"pricier than usual ({ratio:.1f}x its normal price)"
        else:
            tag = "around its usual price"
        points.append(f"Price: you pay about ₹{cpe:.0f} for every ₹1 of yearly profit "
                      f"(P/E {cpe:.0f}); historically it traded near ₹{mpe:.0f} — {tag}.")

    np_, ocf = v.get("net_profit"), v.get("operating_cash_flow")
    if np_ and ocf and np_ > 0:
        r = ocf / np_
        word = ("well backed by real cash" if r >= 0.8
                else "only partly backed by cash (watch this)" if r < 0.5
                else "reasonably backed by cash")
        points.append(f"Cash quality: for every ₹1 of reported profit it actually collected "
                      f"₹{r:.2f} of cash — profits are {word}.")

    debt, eq = v.get("total_debt"), v.get("equity")
    if debt is not None and eq and eq > 0:
        de = debt / eq
        ebit, interest = v.get("ebit"), v.get("interest_expense")
        cover = (ebit / interest) if (ebit and interest and interest > 0) else None
        word = "low, comfortable" if de < 0.5 else "high, worth watching" if de > 1 else "moderate"
        s = f"Debt: it owes ₹{de:.2f} for every ₹1 of owners' money (D/E {de:.2f}) — {word}"
        if cover is not None:
            s += f", and operating profit covers its interest bill about {cover:.0f}x over"
        points.append(s + ".")

    for m in deep:
        if m.known:
            points.append(m.detail)
    return points
