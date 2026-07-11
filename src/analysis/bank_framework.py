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

from dataclasses import replace

from ..research.report import Confidence, Verdict
from .framework import MetricResult, _avg_denominator, assemble_verdict

# ROA thresholds for Indian banks (a well-run bank earns ~1%+; a strained one is below 0.5%).
_ROA_STRONG = 1.0
_ROA_WEAK = 0.5

_BANK_CAVEAT = ("Bank/NBFC: asset quality (GNPA/NNPA) and capital adequacy (CRAR) are NOT in the "
                "free structured feeds; a bank's CASA mix and an NBFC's borrowing cost/spread also "
                "matter. The ROA bands used here are calibrated for banks and may read a strong "
                "NBFC (e.g. gold-loan or microfinance) as only 'mixed' — check the annual report / "
                "investor presentation and segment benchmarks before deciding.")


def return_on_assets(net_profit: float | None, total_assets: float | None,
                     prior_total_assets: float | None = None) -> MetricResult:
    name = "Return on assets (ROA)"
    # WHY: ROA is a lender's core quality dimension (critical); unknown -> confidence can't be HIGH.
    if net_profit is None or total_assets is None or total_assets <= 0:
        return MetricResult(name, False, "unknown", "net profit or total assets unavailable.",
                            critical=True)
    # Average (opening+closing) assets when a cross-verified prior year exists, the SAME denominator
    # rule the displayed ROA insight uses (deep_metrics), so the verdict's ROA and the shown ROA
    # never land in different bands for a bank that grew its balance sheet. Falls back to closing.
    denom, averaged = _avg_denominator(total_assets, prior_total_assets)
    roa = net_profit / denom * 100.0
    basis = ", on average assets" if averaged else ""
    # WHY (real money, severity; mirrors framework.earnings_quality's negative-OCF red flag): a
    # NEGATIVE ROA means the bank posted a NET LOSS, not merely a sub-par positive return. ROA is a
    # bank's SINGLE quality lens, so without singling out a loss it lands in the same "weak" band as a
    # thin-but-profitable bank and, being one concern, reaches only MIXED quality -> a NEUTRAL stance
    # that is ELIGIBLE for the daily suggestions/allocation. A loss-making lender must never be
    # suggestible: mark it severe so a lone ROA loss drags quality to WEAK -> CAUTIOUS -> UNFAVORABLE
    # (see assemble_verdict's severe_concern path), the same protection non-banks get for negative OCF.
    if roa < 0:
        return MetricResult(name, True, "loss",
                            f"ROA {roa:.2f}%{basis} — a NET LOSS: the bank lost money on its asset "
                            "base this year, a serious concern for a lender.", concern=True,
                            critical=True, severe=True)
    if roa >= _ROA_STRONG:
        verdict, concern = "strong", False
    elif roa < _ROA_WEAK:
        verdict, concern = "weak", True
    else:
        verdict, concern = "mixed", False
    # positive only for an affirmatively strong ROA: a "mixed" (0.5-1.0%) ROA is concern-free but
    # NOT a strength, so as a bank's single quality lens it must not by itself reach a STRONG verdict.
    return MetricResult(name, True, verdict, f"ROA {roa:.2f}% ({verdict} for a lender){basis}.",
                        concern, critical=True, positive=(verdict == "strong"))


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
    # WHY (live-verified 2026-07-10): Aditya Birla Capital tags "Financial Conglomerates", not
    # "Credit Services"/"Mortgage Finance", yet its balance sheet is a textbook borrow-to-lend
    # NBFC profile (real D/E ~5.2x) -- left unclassified, it fell to "other" and its normal-for-
    # an-NBFC leverage would be flagged "stretched" on the industrial D/E lens, a false solvency
    # alarm. Checked other real Financial Services names for the same tag first (insurers, asset
    # managers, capital-markets firms all carry their OWN distinct yfinance industry tags, not
    # this one), so this does not sweep in a business that genuinely isn't a lender.
    #
    # WHY require BOTH words (regression, HIGH severity; found by adversarial review): a bare
    # "conglomerate" substring also matches "Conglomerates" -- yfinance's real, distinct
    # Industrials-sector tag for genuine non-lending holding companies (live-verified: Godrej
    # Industries D/E ~4.6x, JSW Holdings, Thermax). That wrongly hid a real, meaningful leverage
    # signal behind a meaningless ROA-only read -- the opposite failure mode from the one this
    # NBFC mapping exists to close. Matching the full "financial conglomerate" phrase excludes it.
    if "credit services" in ind or "mortgage" in ind or "financial conglomerate" in ind:
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


def sector_category(symbol: str) -> str:
    """Classify `symbol`'s business model from ONE yfinance industry fetch: 'bank', 'nbfc',
    'real_estate', or 'other'.

    WHY single-fetch (rate-limit risk): this used to be three separate functions
    (is_bank/is_nbfc/is_real_estate), each independently re-fetching the same industry string --
    classifying one symbol for the pipeline's routing meant up to THREE yfinance .info calls for
    a single piece of information. Consolidated to one fetch, reused for every classification.

    - 'bank' / 'nbfc' (e.g. 'Banks - Regional', 'Credit Services', 'Financial - Mortgages'):
      borrow-to-lend businesses, routed to the ROA-based framework (assemble_bank_verdict) --
      the industrial D/E lens would wrongly flag their normal funding structure as a concern.
    - 'real_estate' (e.g. 'Real Estate - Development'/'Real Estate - Diversified'; live-verified
      across DLF, Godrej Properties, Oberoi Realty, Prestige, Brigade, Sobha, Lodha, Phoenix
      Mills): still uses the industrial D/E lens (this is not a borrow-to-lend model) but gets
      an added leverage caveat -- see framework.REAL_ESTATE_LEVERAGE_CAVEAT.
    - 'other': everything else, including insurance/asset-management/capital-markets financials
      that do NOT run a borrow-to-lend model and must stay on the industrial framework.
    """
    return _industry_category(_yfinance_industry(symbol))


def assemble_bank_verdict(valuation: MetricResult, roa: MetricResult) -> Verdict:
    """Bank/NBFC verdict from ROA + valuation, always carrying the 'check the filing' caveat."""
    # min_signals_for_strong=1: ROA is a lender's single designated quality lens (the industrial
    # leverage/coverage metrics do not apply), so one verified strong ROA is the intended STRONG.
    verdict = assemble_verdict(valuation, [roa], min_signals_for_strong=1,
                               sector_caveats=(_BANK_CAVEAT,))
    # WHY (real money, over-confidence; bank-specific safety-net gap): ROA is the lender's SINGLE
    # critical quality lens, and a bank verdict carries only two metrics (ROA + valuation). When ROA
    # can't be cross-verified but valuation can, the generic confidence math reads known_frac 1/2 =
    # MEDIUM -> a NEUTRAL stance that is ELIGIBLE for the daily suggestions/allocation, leaning on
    # PRICE alone with nothing verified about the bank's quality. An equivalently thin INDUSTRIAL
    # (valuation known, all four quality signals unknown) correctly reads INSUFFICIENT_DATA -- its
    # extra metrics dilute known_frac below 0.5 -- so the bank slips that same net only for having
    # fewer metrics. Force LOW when ROA is unknown so a no-quality-verified bank reads INSUFFICIENT_
    # DATA like the industrial, never a suggestible NEUTRAL on valuation alone. (ROA unknown AND
    # valuation unknown already reads LOW; this only newly catches the valuation-known sub-case.)
    if not roa.known:
        return replace(verdict, confidence=Confidence.LOW, is_bank=True)
    # WHY (real money, honesty): with BOTH metrics known the generic math reads HIGH (2/2), which
    # would imply a comprehensive check when a bank's single most important risk -- asset quality
    # (GNPA/NNPA) and capital adequacy (CRAR) -- is STRUCTURALLY absent from the free feeds (see
    # _BANK_CAVEAT). Cap at MEDIUM so the structured confidence signal matches what the caveat
    # already says in words, and so a bank never out-ranks a fully cross-verified industrial on
    # conviction it structurally cannot earn. No bank scenario warrants HIGH from these feeds.
    # is_bank=True lets the display layer use lender-appropriate phrasing (ROA-based profitability,
    # never "balance sheet", which this framework cannot assess for a lender).
    capped = Confidence.MEDIUM if verdict.confidence == Confidence.HIGH else verdict.confidence
    return replace(verdict, confidence=capped, is_bank=True)
