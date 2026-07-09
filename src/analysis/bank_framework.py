"""Bank-specific analysis.

Banks are leveraged by design, so debt/equity, interest coverage, and EBIT (the industrial
lenses) are meaningless for them. The reliably-free bank metric is Return on Assets (net profit
/ total assets), both cross-verifiable. Asset quality (GNPA/NNPA), CASA mix, and capital
adequacy (CRAR) are not in the free structured feeds, so the verdict names them explicitly as
"check the filing" rather than guessing. Valuation (P/E vs own median) still applies.
"""
from __future__ import annotations

from dataclasses import replace

from ..research.report import Verdict
from .framework import MetricResult, assemble_verdict

# ROA thresholds for Indian banks (a well-run bank earns ~1%+; a strained one is below 0.5%).
_ROA_STRONG = 1.0
_ROA_WEAK = 0.5

_BANK_CAVEAT = ("Bank: asset quality (GNPA/NNPA), CASA mix, and capital adequacy (CRAR) are NOT "
                "in the free structured feeds. Check the annual report / investor presentation for "
                "those before deciding, they can outweigh ROA.")


def return_on_assets(net_profit: float | None,
                     total_assets: float | None) -> MetricResult:
    name = "Return on assets (ROA)"
    if net_profit is None or total_assets is None or total_assets <= 0:
        return MetricResult(name, False, "unknown", "net profit or total assets unavailable.")
    roa = net_profit / total_assets * 100.0
    if roa >= _ROA_STRONG:
        verdict, concern = "strong", False
    elif roa < _ROA_WEAK:
        verdict, concern = "weak", True
    else:
        verdict, concern = "mixed", False
    return MetricResult(name, True, verdict, f"ROA {roa:.2f}% ({verdict} for a bank).", concern)


def is_bank(symbol: str) -> bool:
    """Detect a bank from its yfinance industry (e.g. 'Banks - Regional')."""
    import yfinance as yf

    from ..data.figure_sources import _safe
    from ..data.yfinance_provider import to_yahoo_symbol
    ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
    if ticker is None:
        return False
    info = _safe(lambda: ticker.info) or {}
    return "bank" in str(info.get("industry") or "").lower()


def assemble_bank_verdict(valuation: MetricResult, roa: MetricResult) -> Verdict:
    """Bank verdict from ROA + valuation, always carrying the 'check the filing' caveat."""
    # min_signals_for_strong=1: ROA is a bank's single designated quality lens (the industrial
    # leverage/coverage metrics do not apply), so one verified strong ROA is the intended STRONG.
    verdict = assemble_verdict(valuation, [roa], min_signals_for_strong=1)
    return replace(verdict, reasons=verdict.reasons + (_BANK_CAVEAT,))
