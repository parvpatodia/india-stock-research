"""Seasoned-investor analysis metrics, computed ONLY from cross-verified figures.

Each analyzer takes plain numbers and returns a MetricResult. If a required input is missing
(because the underlying figure was not cross-verified, so the caller passed None), the metric
is `known=False` and contributes nothing to the verdict rather than being guessed. Thresholds
are documented heuristics a human expert can tune; they are not gospel, and the verdict they
produce is always caveated and expert-gated (see report.py).

Pull inputs with value_if_trustworthy(figure): it yields a figure's value only when that
figure is VERIFIED, so an unverified number can never drive a metric.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..constants import PROMOTER_PLEDGE_HIGH_PCT
from ..research.report import (
    Confidence,
    Leaning,
    QualityTier,
    ValuationTier,
    Verdict,
)
from ..research.verification import VerifiedFigure

# --- documented heuristic thresholds (expert-tunable) ---
_PE_CHEAP = 0.80        # current P/E below 80% of its own history median reads cheap
_PE_EXPENSIVE = 1.20    # above 120% reads expensive
_OCF_STRONG = 0.80      # operating cash flow >= 80% of net profit = earnings backed by cash
_OCF_WEAK = 0.50        # below 50% = a quality-of-earnings concern
_DE_HEALTHY = 0.50      # debt/equity below this is comfortable
_DE_STRETCHED = 1.00    # above this is stretched
_COVERAGE_MIN = 3.0     # interest coverage (EBIT/interest) below this is a concern
_PLEDGE_HIGH = PROMOTER_PLEDGE_HIGH_PCT  # promoter pledge above this % is a serious red flag
_MIN_SIGNALS_FOR_STRONG = 2  # "strong" needs >=2 verified quality dimensions, not one lucky one

# WHY (sector-aware analysis, live-verified 2026-07-09): real-estate developers commonly carry
# higher debt funded against project collections and RERA-escrow accounts, so the generic
# industrial D/E bands above can misread a normally-financed developer as stretched. Live D/E
# across 8 real names: DLF 0.01, Oberoi Realty 0.16, Sobha 0.22, Lodha 0.42, Phoenix Mills 0.48
# (all already read fine under the generic bands, well below _DE_STRETCHED); Godrej Properties
# 0.83 and Brigade 0.93 sit close to (but still just under) the generic 1.00 "stretched" line,
# reading "moderate"; Prestige 1.09 is the one that actually crosses it, reading "high, worth
# watching"/stretched despite being a large, established developer, not a distressed one. Rather
# than invent a replacement sector-specific threshold with no authoritative basis (the same trap
# this app avoids everywhere else), disclose the sector context as an explicit caveat -- the same
# honesty-first pattern already used for the bank/NBFC framework -- so the reader compares against
# sector peers and checks collections/pre-sales momentum and RERA compliance, rather than reading
# this as a generic solvency concern.
REAL_ESTATE_LEVERAGE_CAVEAT = (
    "Real estate/construction: developers commonly run higher debt funded against project "
    "collections and RERA-escrow accounts, so the industrial D/E benchmark above may read a "
    "normally-financed developer as more leveraged than it really is. Compare against sector "
    "peers and check collections/pre-sales momentum and RERA compliance in the annual report "
    "before treating this as a solvency concern.")


@dataclass(frozen=True)
class MetricResult:
    name: str
    known: bool
    verdict: str        # short human label, e.g. "cheap" / "strong" / "concern" / "unknown"
    detail: str         # plain-English reason including the numbers used
    concern: bool = False   # True = a negative signal, used to aggregate quality
    critical: bool = False  # a solvency/core dimension (e.g. leverage); if unknown, cap confidence
    magnitude: float | None = None  # optional continuous value (e.g. P/E-vs-median ratio) for ranking
    positive: bool = False  # True = an AFFIRMATIVELY strong dimension (not merely concern-free); a
    #                         # STRONG quality verdict requires at least one of these, so a lone
    #                         # middling-but-not-bad signal (e.g. a bank's "mixed" ROA) can't read STRONG


def value_if_trustworthy(figure: VerifiedFigure | None) -> float | None:
    # WHY: enforces "only cross-verified numbers drive analysis" at the boundary.
    if figure is None or not figure.is_trustworthy:
        return None
    return figure.value


def _unknown(name: str, why: str, critical: bool = False) -> MetricResult:
    return MetricResult(name, known=False, verdict="unknown", detail=why, critical=critical)


def _avg_denominator(closing: float, prior: float | None) -> tuple[float, bool]:
    """The denominator for a return ratio: the AVERAGE of opening (prior) and closing balance when
    a positive prior is available, else the closing value alone. WHY (CA-level rigor): profit is
    earned OVER the year, so it should be measured against the capital deployed OVER the year --
    averaging opening+closing. Using the closing snapshot understates the ratio for a company that
    grew its equity/assets during the year (retained earnings, a capital raise, a merger). Falls
    back to the closing value when no cross-verified prior year exists, so nothing is fabricated.

    Lives here (the shared base module), not in deep_metrics, so the bank verdict's ROA
    (bank_framework) and the displayed ratio suite (deep_metrics) apply ONE identical denominator
    rule -- otherwise a bank's verdict ROA (closing) and its shown ROA (average) can land in
    different bands and contradict each other for a bank that grew its balance sheet."""
    if prior is not None and prior > 0:
        return (closing + prior) / 2.0, True
    return closing, False


def valuation_vs_history(current_pe: float | None,
                         median_pe: float | None) -> MetricResult:
    name = "Valuation (P/E vs own history)"
    if current_pe is None or median_pe is None or current_pe <= 0 or median_pe <= 0:
        return _unknown(name, "P/E or its historical median is unavailable or non-positive "
                              "(e.g. loss-making); cannot judge valuation.")
    ratio = current_pe / median_pe
    if ratio < _PE_CHEAP:
        v, concern = "cheap", False
    elif ratio > _PE_EXPENSIVE:
        v, concern = "expensive", True
    else:
        v, concern = "fair", False
    return MetricResult(name, True, v,
                        f"P/E {current_pe:.1f} vs its own median {median_pe:.1f} "
                        f"({ratio:.0%} of history) reads {v}.", concern, magnitude=ratio)


def earnings_quality(operating_cash_flow: float | None,
                     net_profit: float | None) -> MetricResult:
    name = "Earnings quality (cash flow vs profit)"
    if operating_cash_flow is None or net_profit is None or net_profit <= 0:
        return _unknown(name, "operating cash flow or (positive) net profit unavailable.")
    ratio = operating_cash_flow / net_profit
    # WHY (CA-level rigor, real money, live-verified against real filings: BHEL FY2024, SAIL
    # FY2023, VAKRANGEE FY2025/FY2023): a company can report a genuine profit while operating
    # cash flow is actually NEGATIVE -- cash left the business even as it claimed to make money.
    # Folding this into the same "weak" band as a merely-thin-but-still-positive cash conversion
    # (e.g. ratio 10%) understates a much more severe, textbook earnings-quality red flag (working
    # capital stress, or aggressive revenue recognition); it must read as distinctly more serious.
    if ratio < 0:
        return MetricResult(
            name, True, "red_flag",
            f"operating cash flow was NEGATIVE ({ratio:.0%} of net profit) despite a reported "
            "profit -- a serious quality-of-earnings red flag: the business consumed cash from "
            "operations while claiming to be profitable. Check receivables, working capital, "
            "and revenue recognition before trusting the profit figure.", concern=True)
    if ratio >= _OCF_STRONG:
        v, concern = "strong", False
    elif ratio < _OCF_WEAK:
        v, concern = "weak", True
    else:
        v, concern = "mixed", False
    return MetricResult(name, True, v,
                        f"operating cash flow is {ratio:.0%} of net profit; profits are "
                        f"{'well' if concern is False and v == 'strong' else 'only partly'} "
                        f"backed by cash.", concern, positive=(v == "strong"))


def leverage_health(total_debt: float | None, equity: float | None,
                    ebit: float | None, interest: float | None) -> MetricResult:
    name = "Leverage and interest cover"
    # WHY: leverage is critical (solvency). Marked so even when unknown, so a verdict can never be
    # HIGH-confidence with the company's debt unverified.
    if total_debt is None or equity is None or equity <= 0:
        return _unknown(name, "debt or (positive) equity unavailable.", critical=True)
    de = total_debt / equity
    has_interest = interest is not None and interest > 0
    # Interest coverage (EBIT/interest) is only meaningful with a POSITIVE operating profit; a
    # negative EBIT can't cover interest from operations at all, so don't display a nonsensical
    # negative "cover" (e.g. "-2.0x") -- flag the operating loss in words instead. Verdicts are
    # UNCHANGED: a negative-EBIT leveraged name still reads stretched, exactly as the old
    # negative-coverage-below-3 path did, and the ebit<=0 quirk (coverage ignored) is preserved.
    coverage = ebit / interest if (has_interest and ebit is not None and ebit > 0) else None
    operating_loss_vs_interest = has_interest and ebit is not None and ebit < 0
    weak_cover = operating_loss_vs_interest or (coverage is not None and coverage < _COVERAGE_MIN)
    stretched = de > _DE_STRETCHED or weak_cover
    healthy = de < _DE_HEALTHY and not weak_cover
    v = "stretched" if stretched else ("healthy" if healthy else "moderate")
    if coverage is not None:
        detail = f"debt/equity {de:.2f}, interest cover {coverage:.1f}x reads {v}."
    elif operating_loss_vs_interest:
        detail = (f"debt/equity {de:.2f} reads {v}; operating profit is negative, so it isn't "
                  "covering its interest bill from operations.")
    else:
        detail = f"debt/equity {de:.2f} reads {v}."
    return MetricResult(name, True, v, detail, concern=stretched, critical=True, positive=healthy)


def promoter_pledge(pledge_pct: float | None) -> MetricResult:
    # WHY (known gap, live-verified 2026-07-09): promoter_pledge_pct is declared in
    # FRAMEWORK_FIGURES but is NOT currently populated by any real source — neither
    # YFinanceFigureSource nor parse_screener_figures sets it, and the annual-report LLM
    # extractor (annual_report_source.py _TARGETS) does not target it either. Checked Screener's
    # free page live across 9 stocks (incl. names with historically high pledge: SUZLON, ZEEL,
    # RPOWER, SADBHAV, JPASSOCIAT, GVKPIL, YESBANK) and found zero "pledge"/"encumbrance"
    # mentions in the fetched HTML, so this metric always reads UNAVAILABLE in production today.
    # A genuinely high, dangerous pledge would not be flagged. Documented rather than silently
    # "someday works", so this is not re-discovered as a mystery in a future session; see the
    # annual report's Corporate Governance Report (SEBI LODR requires pledge disclosure there)
    # for a real value until this is wired to an actual source.
    name = "Promoter pledge"
    if pledge_pct is None:
        return _unknown(name, "promoter pledge percentage unavailable.")
    if pledge_pct <= 0:
        return MetricResult(name, True, "none", "no promoter pledging.", concern=False)
    if pledge_pct > _PLEDGE_HIGH:
        return MetricResult(name, True, "high",
                            f"{pledge_pct:.0f}% of promoter holding is pledged; a serious "
                            "red flag.", concern=True)
    return MetricResult(name, True, "watch",
                        f"{pledge_pct:.0f}% of promoter holding is pledged; watch it.",
                        concern=False)


def assemble_verdict(valuation: MetricResult,
                     quality_signals: list[MetricResult],
                     min_signals_for_strong: int = _MIN_SIGNALS_FOR_STRONG,
                     sector_caveats: tuple[str, ...] = ()) -> Verdict:
    """Turn the metrics into a caveated Verdict. Confidence reflects how much was actually
    known; a thinly-evidenced verdict is low confidence, not a confident guess.

    min_signals_for_strong: how many verified, concern-free quality dimensions are needed to read
    STRONG. Default 2 for the industrial framework (don't call a balance sheet strong on one lucky
    metric with debt unverified); banks pass 1 because ROA is their single designated quality lens.

    sector_caveats: sector-specific context (e.g. REAL_ESTATE_LEVERAGE_CAVEAT) -- never changes
    any tier or score, only adds disclosure. Kept OUT of `reasons` (see Verdict.sector_caveats):
    a caveat is not itself a cross-verified figure, so it must never blend into the list the app
    renders under a "Why (each from cross-verified figures)" header.
    """
    valuation_tier = {
        "cheap": ValuationTier.CHEAP, "fair": ValuationTier.FAIR,
        "expensive": ValuationTier.EXPENSIVE,
    }.get(valuation.verdict, ValuationTier.UNKNOWN)

    known_quality = [m for m in quality_signals if m.known]
    concerns = [m for m in known_quality if m.concern]
    # A critical quality dimension (leverage/debt for industrials, ROA for banks) left unverified
    # blocks STRONG: you cannot call a balance sheet strong without checking its solvency, even if
    # other, softer signals look clean. This closes the "two non-solvency signals => STRONG" hole.
    critical_quality_unknown = any(m.critical and not m.known for m in quality_signals)
    # WHY (real money): STRONG needs at least one AFFIRMATIVELY strong dimension, not merely the
    # absence of concerns. Without this, a lone concern-free-but-middling signal reads STRONG --
    # most visibly for banks (min_signals=1), where a "mixed for a lender" ROA (0.5-1.0%, e.g. a
    # PSU bank at 0.7%) has concern=False and so passed straight to STRONG, contradicting the
    # metric's own "mixed" label and overstating a whole swath of mid-ROA banks into FAVORABLE.
    # earnings_quality "strong" / leverage_health "healthy" / a bank's "strong" ROA set positive.
    has_affirmative_strength = any(m.positive for m in known_quality)
    if not known_quality:
        quality_tier = QualityTier.UNKNOWN
    elif len(concerns) >= 2:
        quality_tier = QualityTier.WEAK
    elif len(concerns) == 1:
        quality_tier = QualityTier.MIXED
    elif (len(known_quality) >= min_signals_for_strong and not critical_quality_unknown
          and has_affirmative_strength):
        quality_tier = QualityTier.STRONG
    else:
        # WHY (real money): zero concerns but either fewer than min_signals_for_strong verified
        # dimensions, the critical one (debt) unverified, or NOTHING affirmatively strong (only
        # concern-free-but-middling signals), is not enough to call a balance sheet STRONG.
        # Requiring corroboration incl. solvency AND a genuine strength keeps a cheap P/E + soft
        # signals from reading FAVORABLE on thin data. Banks opt into 1 (ROA is their single lens).
        quality_tier = QualityTier.MIXED

    if valuation_tier == ValuationTier.UNKNOWN and quality_tier == QualityTier.UNKNOWN:
        leaning = Leaning.UNKNOWN
    elif valuation_tier == ValuationTier.EXPENSIVE or quality_tier == QualityTier.WEAK:
        leaning = Leaning.CAUTIOUS
    elif (valuation_tier in (ValuationTier.CHEAP, ValuationTier.FAIR)
          and quality_tier == QualityTier.STRONG):
        leaning = Leaning.CONSTRUCTIVE
    else:
        leaning = Leaning.NEUTRAL

    all_metrics = [valuation, *quality_signals]
    known_frac = sum(1 for m in all_metrics if m.known) / max(len(all_metrics), 1)
    confidence = (Confidence.HIGH if known_frac >= 0.75
                  else Confidence.MEDIUM if known_frac >= 0.5 else Confidence.LOW)
    # WHY (real money): a critical dimension (leverage/debt) left unverified caps confidence at
    # MEDIUM. Otherwise a name with everything else known would read HIGH-confidence while its
    # solvency is unchecked, the exact over-confidence this app must avoid.
    if confidence == Confidence.HIGH and any(m.critical and not m.known for m in all_metrics):
        confidence = Confidence.MEDIUM

    reasons = tuple(m.detail for m in all_metrics if m.known)
    return Verdict(valuation=valuation_tier, quality=quality_tier, leaning=leaning,
                   confidence=confidence, reasons=reasons, sector_caveats=sector_caveats,
                   valuation_ratio=valuation.magnitude)
