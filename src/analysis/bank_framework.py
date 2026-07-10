"""Bank and NBFC (lending-business) analysis.

Banks and NBFCs (non-bank lenders: Bajaj Finance, Cholamandalam, housing-finance companies) are
leveraged BY DESIGN, they borrow to lend, so debt/equity, interest coverage, and EBIT (the
industrial lenses) would wrongly flag their normal funding structure as a solvency concern. The
reliably-free lender metric is Return on Assets (net profit / total assets), cross-verifiable
from both yfinance and Screener. Asset quality (GNPA/NNPA) and capital adequacy (CRAR) are not
in the free structured feeds, so the verdict names them explicitly as "check the filing" rather
than guessing; it also discloses that the ROA bands here are calibrated for banks and may not
fit every NBFC segment (e.g. gold-loan/microfinance lenders typically run higher ROA). Valuation
(P/E vs own median) still applies to both.
"""
from __future__ import annotations

from ..research.report import Verdict
from .framework import MetricResult, assemble_verdict

# ROA thresholds for Indian banks (a well-run bank earns ~1%+; a strained one is below 0.5%).
_ROA_STRONG = 1.0
_ROA_WEAK = 0.5

_BANK_CAVEAT = ("Bank/NBFC: asset quality (GNPA/NNPA) and capital adequacy (CRAR) are NOT in the "
                "free structured feeds; a bank's CASA mix and an NBFC's borrowing cost/spread also "
                "matter. The ROA bands used here are calibrated for banks and may read a strong "
                "NBFC (e.g. gold-loan or microfinance) as only 'mixed' — check the annual report / "
                "investor presentation and segment benchmarks before deciding.")


def return_on_assets(net_profit: float | None,
                     total_assets: float | None) -> MetricResult:
    name = "Return on assets (ROA)"
    # WHY: ROA is a lender's core quality dimension (critical); unknown -> confidence can't be HIGH.
    if net_profit is None or total_assets is None or total_assets <= 0:
        return MetricResult(name, False, "unknown", "net profit or total assets unavailable.",
                            critical=True)
    roa = net_profit / total_assets * 100.0
    if roa >= _ROA_STRONG:
        verdict, concern = "strong", False
    elif roa < _ROA_WEAK:
        verdict, concern = "weak", True
    else:
        verdict, concern = "mixed", False
    return MetricResult(name, True, verdict, f"ROA {roa:.2f}% ({verdict} for a lender).", concern,
                        critical=True)


def _industry_category(industry: str) -> str:
    """Classify a yfinance industry string into 'bank', 'nbfc', 'real_estate', or 'other'. Pure
    and unit-tested (the network fetch that supplies `industry` is not, matching is_bank's
    existing pattern). Deliberately narrow: 'Financial Services' broadly also covers insurance,
    asset management, and capital markets/exchanges, businesses that do NOT run a borrow-to-lend
    model and belong on the industrial D/E framework, not swept into the ROA-only lens.
    'real_estate' still uses the industrial D/E lens (unlike bank/nbfc, this is not a
    borrow-to-lend business) but gets an added leverage caveat -- see
    framework.REAL_ESTATE_LEVERAGE_CAVEAT."""
    ind = industry.lower()
    if "bank" in ind:
        return "bank"
    if "credit services" in ind or "mortgage" in ind:
        return "nbfc"
    if "real estate" in ind:
        return "real_estate"
    return "other"


def _yfinance_industry(symbol: str) -> str:
    import yfinance as yf

    from ..data.figure_sources import _safe
    from ..data.yfinance_provider import to_yahoo_symbol
    ticker = _safe(lambda: yf.Ticker(to_yahoo_symbol(symbol)))
    if ticker is None:
        return ""
    info = _safe(lambda: ticker.info) or {}
    return str(info.get("industry") or "")


def is_bank(symbol: str) -> bool:
    """Detect a bank from its yfinance industry (e.g. 'Banks - Regional')."""
    return _industry_category(_yfinance_industry(symbol)) == "bank"


def is_nbfc(symbol: str) -> bool:
    """Detect a lending-business NBFC from its yfinance industry (e.g. 'Credit Services',
    'Financial - Mortgages'). Routes to the same ROA-based framework as a bank: see module
    docstring for why the industrial D/E lens is wrong for a borrow-to-lend business."""
    return _industry_category(_yfinance_industry(symbol)) == "nbfc"


def is_real_estate(symbol: str) -> bool:
    """Detect a real-estate developer from its yfinance industry (e.g. 'Real Estate -
    Development', 'Real Estate - Diversified'; live-verified across DLF, Godrej Properties,
    Oberoi Realty, Prestige, Brigade, Sobha, Lodha, Phoenix Mills). Unlike is_bank/is_nbfc, this
    does NOT change the analysis framework (real estate still uses the industrial D/E lens); it
    only adds a leverage caveat -- see framework.REAL_ESTATE_LEVERAGE_CAVEAT for why."""
    return _industry_category(_yfinance_industry(symbol)) == "real_estate"


def assemble_bank_verdict(valuation: MetricResult, roa: MetricResult) -> Verdict:
    """Bank/NBFC verdict from ROA + valuation, always carrying the 'check the filing' caveat."""
    # min_signals_for_strong=1: ROA is a lender's single designated quality lens (the industrial
    # leverage/coverage metrics do not apply), so one verified strong ROA is the intended STRONG.
    return assemble_verdict(valuation, [roa], min_signals_for_strong=1,
                            sector_caveats=(_BANK_CAVEAT,))
