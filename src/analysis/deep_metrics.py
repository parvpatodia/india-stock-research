"""Deeper analysis metrics + plain-language explanations.

The ratios a chartered accountant actually looks at (ROE, ROCE, ROA, margins, asset turnover),
computed ONLY from cross-verified figures, plus a plain_points() renderer that turns the numbers
into short, everyday-language sentences a non-expert can read. Deterministic: no LLM, so there is
nothing to fabricate. Thresholds are documented heuristics an expert can tune; a missing input
(figure not cross-verified) makes a metric known=False and it simply isn't shown.
"""
from __future__ import annotations

from .framework import (
    REAL_ESTATE_LEVERAGE_CAVEAT,
    MetricResult,
    _avg_denominator,
    earnings_quality,
    leverage_health,
)

# Maps leverage_health's own tier (D/E AND interest coverage) onto this module's plain-language
# wording, so the always-visible summary and the Verdict's tier/concern flag can never diverge --
# see plain_points' debt point for the regression this closes.
_LEVERAGE_WORD = {"healthy": "low, comfortable", "stretched": "high, worth watching",
                  "moderate": "moderate"}

# Maps earnings_quality's own tier onto this module's plain-language wording, so the always-
# visible summary and the Verdict's tier/concern flag can never diverge -- see plain_points'
# cash-quality point for the regression this closes (the same class of bug as _LEVERAGE_WORD).
_OCF_WORD = {"strong": "well backed by real cash", "mixed": "reasonably backed by cash",
            "weak": "only partly backed by cash (watch this)",
            "red_flag": "NOT backed by cash -- a red flag (it consumed cash while reporting a "
                        "profit)"}

# Documented heuristic thresholds (expert-tunable).
_ROE_GOOD, _ROE_WEAK = 15.0, 8.0          # % return on shareholders' equity
_ROCE_GOOD, _ROCE_WEAK = 15.0, 10.0       # % return on capital employed
_ROA_GOOD, _ROA_WEAK = 6.0, 2.0           # % return on assets (industrials; banks differ)
_NETMARGIN_GOOD, _NETMARGIN_WEAK = 12.0, 3.0
_OPMARGIN_GOOD, _OPMARGIN_WEAK = 15.0, 5.0
# Above this a dividend yield is unusual for an Indian equity (the market yields ~1-1.5%; even
# high-payout PSUs/utilities sit ~5-8%). A yield this high is worth a sustainability check: it
# often reflects a fallen price as much as a generous dividend, and can flag an expected cut.
_DIVIDEND_YIELD_UNUSUAL = 6.0


def _unknown(name: str, why: str) -> MetricResult:
    return MetricResult(name, known=False, verdict="unknown", detail=why)


def _rate(value: float, good: float, weak: float) -> tuple[str, bool]:
    if value >= good:
        return "strong", False
    if value < weak:
        return "weak", True
    return "moderate", False


def return_on_equity(net_profit: float | None, equity: float | None,
                     prior_equity: float | None = None) -> MetricResult:
    name = "Return on equity (ROE)"
    if net_profit is None or equity is None or equity <= 0:
        return _unknown(name, "net profit or (positive) shareholders' equity unavailable.")
    denom, averaged = _avg_denominator(equity, prior_equity)
    roe = net_profit / denom * 100
    v, concern = _rate(roe, _ROE_GOOD, _ROE_WEAK)
    basis = ", on average equity" if averaged else ""
    # WHY (real money, clarity): a loss-making company has a NEGATIVE return; "earns about ₹-50 a
    # year" is a confusing double negative for a non-expert -- say it LOSES money instead.
    verb = f"loses about ₹{abs(roe):.0f}" if roe < 0 else f"earns about ₹{roe:.0f}"
    return MetricResult(name, True, v,
                        f"For every ₹100 of owners' money in the business, it {verb} a year "
                        f"(ROE {roe:.0f}%{basis}) — {v}.", concern)


def return_on_capital(ebit: float | None, equity: float | None,
                      total_debt: float | None, prior_equity: float | None = None,
                      prior_total_debt: float | None = None) -> MetricResult:
    name = "Return on capital employed (ROCE)"
    if ebit is None or equity is None or total_debt is None or (equity + total_debt) <= 0:
        return _unknown(name, "operating profit, equity, or debt unavailable.")
    # average capital employed only when BOTH prior components are available (else point capital)
    prior_capital = (prior_equity + prior_total_debt
                     if prior_equity is not None and prior_total_debt is not None else None)
    denom, averaged = _avg_denominator(equity + total_debt, prior_capital)
    roce = ebit / denom * 100
    v, concern = _rate(roce, _ROCE_GOOD, _ROCE_WEAK)
    basis = ", on average capital" if averaged else ""
    # A negative ROCE (an operating loss) must say it LOSES money, not "earns about ₹-20" -- the
    # same double-negative fix applied to ROE/ROA/margins (this metric was missed there).
    verb = f"loses about ₹{abs(roce):.0f}" if roce < 0 else f"earns about ₹{roce:.0f}"
    return MetricResult(name, True, v,
                        f"It {verb} a year for every ₹100 of capital it uses "
                        f"(ROCE {roce:.0f}%{basis}) — {v}.", concern, positive=(v == "strong"))


def return_on_assets(net_profit: float | None, total_assets: float | None,
                     good: float = _ROA_GOOD, weak: float = _ROA_WEAK,
                     prior_total_assets: float | None = None) -> MetricResult:
    name = "Return on assets (ROA)"
    if net_profit is None or total_assets is None or total_assets <= 0:
        return _unknown(name, "net profit or (positive) total assets unavailable.")
    denom, averaged = _avg_denominator(total_assets, prior_total_assets)
    roa = net_profit / denom * 100
    v, concern = _rate(roa, good, weak)
    # WHY: banks run on ~1% ROA by nature, so the caller passes bank bands; otherwise a healthy
    # bank would read "weak" here and contradict the (correct) bank verdict.
    basis = ", on average assets" if averaged else ""
    # A loss-making company earns a NEGATIVE return; say it LOSES money rather than "earns ₹-6.0".
    verb = f"loses about ₹{abs(roa):.1f}" if roa < 0 else f"earns about ₹{roa:.1f}"
    return MetricResult(name, True, v,
                        f"For every ₹100 of everything it owns, it {verb} "
                        f"(ROA {roa:.1f}%{basis}) — {v}.", concern)


def net_margin(net_profit: float | None, revenue: float | None) -> MetricResult:
    name = "Net profit margin"
    if net_profit is None or revenue is None or revenue <= 0:
        return _unknown(name, "net profit or (positive) revenue unavailable.")
    m = net_profit / revenue * 100
    # WHY (quality of earnings, honesty): net profit LARGER than sales (margin > 100%) can't come
    # from the core sales business; it's driven by other income (investment/interest) or one-off
    # gains. "net margin 200% -- strong" misrepresents that, so flag it plainly (concern-free: it
    # can be structural for a holding company, but the reader is told to check repeatability).
    if m > 100:
        return MetricResult(name, True, "unusual",
                            f"Net profit was LARGER than total sales this year (net margin "
                            f"{m:.0f}%), so this profit isn't coming from the core sales business "
                            "-- it's driven by other income (investment/interest) or one-off gains; "
                            "check how repeatable it is.", concern=False)
    v, concern = _rate(m, _NETMARGIN_GOOD, _NETMARGIN_WEAK)
    # A loss-making company keeps a NEGATIVE margin; "keeps about ₹-50 of profit" is a confusing
    # double negative for a non-expert -- say it LOSES money per ₹100 of sales instead.
    if m < 0:
        detail = (f"It loses about ₹{abs(m):.0f} for every ₹100 of sales "
                  f"(net margin {m:.0f}%) — {v}.")
    else:
        detail = (f"It keeps about ₹{m:.0f} of final profit from every ₹100 of sales "
                  f"(net margin {m:.0f}%) — {v}.")
    return MetricResult(name, True, v, detail, concern)


def operating_margin(ebit: float | None, revenue: float | None) -> MetricResult:
    name = "Operating margin"
    if ebit is None or revenue is None or revenue <= 0:
        return _unknown(name, "operating profit or (positive) revenue unavailable.")
    m = ebit / revenue * 100
    # As with net margin: an operating measure LARGER than sales (> 100%) is inflated by
    # non-operating income (this EBIT = PBT + interest, so it carries other income) or one-off
    # items, not the core sales business. "operating margin 150% -- strong" would misrepresent it.
    if m > 100:
        return MetricResult(name, True, "unusual",
                            f"Operating profit exceeded total sales this year (margin {m:.0f}%), so "
                            "it's inflated by non-operating income (investment/interest) or one-off "
                            "items rather than the core sales business; check how repeatable it is.",
                            concern=False)
    v, concern = _rate(m, _OPMARGIN_GOOD, _OPMARGIN_WEAK)
    # A company with an operating LOSS has a negative operating margin; "₹-30 is left as operating
    # profit" reads as a confusing double negative -- say it loses money at the operating level.
    if m < 0:
        detail = (f"It loses about ₹{abs(m):.0f} at the operating level for every ₹100 of sales "
                  f"(operating margin {m:.0f}%) — {v}.")
    else:
        detail = (f"From every ₹100 of sales, ₹{m:.0f} is left as operating profit "
                  f"(operating margin {m:.0f}%) — {v}.")
    return MetricResult(name, True, v, detail, concern)


def asset_turnover(revenue: float | None, total_assets: float | None,
                   prior_total_assets: float | None = None) -> MetricResult:
    name = "Asset turnover"
    if revenue is None or total_assets is None or total_assets <= 0:
        return _unknown(name, "revenue or (positive) total assets unavailable.")
    # WHY average assets (CA-level rigor, consistency with ROA): sales are generated OVER the year,
    # so the flow-over-stock ratio uses the average of opening and closing assets when a
    # cross-verified prior year is available; falls back to the closing value otherwise.
    denom, averaged = _avg_denominator(total_assets, prior_total_assets)
    t = revenue / denom
    v = "high" if t >= 1.0 else "low" if t < 0.4 else "moderate"
    basis = ", on average assets" if averaged else ""
    return MetricResult(name, True, v,
                        f"It generates ₹{t:.2f} of sales a year for every ₹1 of assets "
                        f"(asset turnover {t:.2f}x{basis}) — {v}.", concern=False)


def compute_deep_metrics(v: dict, is_bank: bool = False) -> list[MetricResult]:
    """Compute the ratio suite from a dict of cross-verified values (missing -> None). For banks,
    margins and asset turnover are skipped (bank P&L is interest income, not sales)."""
    from .bank_framework import _ROA_STRONG, _ROA_WEAK as _BANK_ROA_WEAK
    # prior-year (opening) balances, if the caller supplied cross-verified ones, so the return
    # ratios use the CA-standard average denominator (see _avg_denominator); absent -> point value.
    p_eq, p_debt, p_assets = (v.get("prior_equity"), v.get("prior_total_debt"),
                              v.get("prior_total_assets"))
    roa = (return_on_assets(v.get("net_profit"), v.get("total_assets"),
                            good=_ROA_STRONG, weak=_BANK_ROA_WEAK, prior_total_assets=p_assets)
           if is_bank else
           return_on_assets(v.get("net_profit"), v.get("total_assets"), prior_total_assets=p_assets))
    metrics = [return_on_equity(v.get("net_profit"), v.get("equity"), prior_equity=p_eq), roa]
    if not is_bank:
        metrics += [
            return_on_capital(v.get("ebit"), v.get("equity"), v.get("total_debt"),
                              prior_equity=p_eq, prior_total_debt=p_debt),
            net_margin(v.get("net_profit"), v.get("revenue")),
            operating_margin(v.get("ebit"), v.get("revenue")),
            asset_turnover(v.get("revenue"), v.get("total_assets"), prior_total_assets=p_assets),
        ]
    return metrics


def plain_points(v: dict, deep: list[MetricResult], is_real_estate: bool = False,
                 is_bank: bool = False) -> list[str]:
    """5-6 short, everyday-language reasons with the real numbers, for a non-expert reader.
    Covers price, cash quality, and debt from the core figures, then the ratio suite. Only
    includes points whose inputs cross-verified.

    is_bank: WHY (real money, sector-aware; regression exposed once bank balance sheets started
    parsing) -- the industrial D/E "Debt:" line and the OCF-vs-profit "Cash quality:" line do NOT
    apply to a lender. A bank/NBFC is leveraged by design (the verdict routes it to the ROA lens,
    not D/E; see bank_framework), and its operating cash flow is dominated by lending/deposit
    flows, so a healthy growing bank can show negative OCF that would FALSE-flag as a cash red
    flag. compute_deep_metrics and the verdict already skip these lenses for banks; this
    always-visible summary must too, or it contradicts them right where a parent reads it.

    is_real_estate: WHY (real money, UI honesty) -- the real-estate leverage caveat
    (framework.REAL_ESTATE_LEVERAGE_CAVEAT) previously only reached Verdict.reasons, shown inside
    the collapsed "See the evidence" expander. This ALWAYS-VISIBLE summary (report.insights) kept
    saying "high, worth watching" for a real developer at D/E > 1 (live-verified: Prestige 1.09)
    with zero sector context, so a reader could see the un-caveated alarm and never open the
    expander that explains it is sector-normal. Attached only when the debt point actually reads
    "high, worth watching" -- a "moderate" read is not itself presented as a concern, so adding
    sector commentary there would be clutter, not honesty.
    """
    points: list[str] = []

    cpe, mpe = v.get("current_pe"), v.get("median_pe")
    if cpe and mpe and cpe > 0 and mpe > 0:
        ratio = cpe / mpe
        # WHY (real money, clarity): the P/E is a MULTIPLE, not the share price. Framing the median
        # P/E as "traded near ₹24" or "its usual price" reads the valuation LEVEL as a rupee share
        # price -- confusing next to the real (e.g. ₹1,400) price the parent sees for the same
        # holding elsewhere. Keep the intuitive "₹ paid per ₹1 of profit" explanation, but tag the
        # comparison as the valuation level versus the company's OWN history, never a "price".
        if ratio < 0.8:
            tag = f"cheaper than usual versus its own history (about {ratio:.0%} of its normal)"
        elif ratio > 1.2:
            tag = f"pricier than usual versus its own history ({ratio:.1f}x its normal)"
        else:
            tag = "about in line with its own history"
        points.append(f"Price: you pay about ₹{cpe:.0f} for every ₹1 of yearly profit "
                      f"(P/E {cpe:.0f}); historically you'd have paid about ₹{mpe:.0f} for that "
                      f"same ₹1 of profit — {tag}.")

    np_, ocf = v.get("net_profit"), v.get("operating_cash_flow")
    if not is_bank and np_ and ocf and np_ > 0:
        r = ocf / np_
        # WHY (real money, honesty; adversarial-review-style regression, same class as the debt
        # word below): derive the word from earnings_quality's OWN tier, the SAME computation the
        # Verdict's tier/concern flag is built from -- a prior version recomputed an independent
        # ratio/word here, which silently softened a genuinely NEGATIVE operating cash flow (cash
        # consumed while reporting a profit) into the same "only partly backed" wording as a
        # merely-thin-but-positive ratio, so the always-visible summary understated exactly the
        # pattern the collapsed evidence panel now calls a red flag.
        word = _OCF_WORD.get(earnings_quality(ocf, np_).verdict, "reasonably backed by cash")
        # WHY (clarity): a NEGATIVE ratio means cash left the business; "collected a net outflow of
        # cash" is a self-contradiction ("collected" an outflow), so phrase the negative case as
        # cash flowing OUT. The positive case is unchanged ("collected ₹X of cash").
        if r >= 0:
            cash_clause = f"it actually collected ₹{r:.2f} of cash"
        else:
            cash_clause = f"₹{abs(r):.2f} of cash actually flowed OUT of the business"
        points.append(f"Cash quality: for every ₹1 of reported profit {cash_clause} — "
                      f"profits are {word}.")

    debt, eq = v.get("total_debt"), v.get("equity")
    if not is_bank and debt is not None and eq and eq > 0:
        de = debt / eq
        ebit, interest = v.get("ebit"), v.get("interest_expense")
        has_interest = interest is not None and interest > 0
        # Only show a coverage multiple with a POSITIVE operating profit; a negative EBIT can't
        # cover interest, so "covers its interest bill about -2x over" is a confusing negative.
        cover = (ebit / interest) if (has_interest and ebit is not None and ebit > 0) else None
        operating_loss = has_interest and ebit is not None and ebit < 0
        # WHY (real money, honesty; adversarial-review regression): derive the word from
        # leverage_health's OWN tier (D/E AND interest coverage), the SAME computation the
        # Verdict's tier/concern flag and REAL_ESTATE_LEVERAGE_CAVEAT gating are built from --
        # a prior version recomputed an independent, D/E-only word here, which could disagree
        # with leverage_health when weak coverage (not D/E) was the actual stretched signal, so
        # the always-visible summary and the collapsed evidence panel called the same company
        # two different things.
        word = _LEVERAGE_WORD.get(leverage_health(debt, eq, ebit, interest).verdict, "moderate")
        s = f"Debt: it owes ₹{de:.2f} for every ₹1 of owners' money (D/E {de:.2f}) — {word}"
        if cover is not None:
            s += f", and operating profit covers its interest bill about {cover:.0f}x over"
        elif operating_loss:
            s += (", and its operating profit is negative, so it isn't covering its interest "
                  "bill from operations")
        s += "."
        if is_real_estate and word == "high, worth watching":
            s += " " + REAL_ESTATE_LEVERAGE_CAVEAT
        points.append(s)

    dy = v.get("dividend_yield_pct")
    if dy is not None and dy >= 0:
        # WHY (honest, non-judgmental): dividend yield is context-dependent, not a clean good/bad
        # signal. A 0% yield can be a legitimate reinvesting growth company; a high yield can be
        # generous returns OR a distressed, falling price. Never claim a direction is automatically
        # better; just report the cross-verified number and leave the weighing to the reader.
        if dy == 0:
            points.append("Dividend: it currently pays no dividend; profits are being reinvested "
                          "rather than distributed, common for growth-focused businesses, not "
                          "automatically a red flag.")
        else:
            band = "high" if dy >= 3.0 else "moderate" if dy >= 1.0 else "modest"
            line = (f"Dividend: a {band} {dy:.1f}% dividend yield at the current price. "
                    "Neither high nor low is automatically good or bad on its own, weigh "
                    "it against whether the business is reinvesting for growth instead.")
            # WHY (real money, honesty for income-seeking parents): yield = dividend / price, so an
            # UNUSUALLY high yield often reflects a fallen share price as much as a generous
            # dividend, and can flag a payout the market expects to be cut -- the classic yield trap.
            # Non-alarmist ("check"): stays true even for a legitimate high-yielder whose payout is
            # covered (the check simply passes), while warning off a distressed value trap.
            if dy >= _DIVIDEND_YIELD_UNUSUAL:
                line += (" A yield this high often reflects a fallen share price as much as a "
                         "generous dividend, and can flag a payout the market expects to be cut -- "
                         "check it's covered by earnings and cash flow before relying on it.")
            points.append(line)

    for m in deep:
        if m.known:
            points.append(m.detail)
    return points
