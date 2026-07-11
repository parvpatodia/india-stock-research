"""SIP (Systematic Investment Plan) projection math. Pure, no I/O.

This is compound-interest arithmetic on an assumed constant return. It is NOT a prediction
and NOT a promise: real fund returns vary year to year and can be negative. The UI must say
so. The math uses annuity-due (contributions at the start of each month), which is how SIP
installments are typically invested and how standard SIP calculators compute it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SIPProjection:
    monthly: float
    years: int
    annual_return_pct: float
    invested: float
    projected_value: float

    @property
    def gain(self) -> float:
        return self.projected_value - self.invested


def sip_future_value(monthly: float, annual_return_pct: float, years: int) -> SIPProjection:
    """Projected corpus of a monthly SIP under a constant assumed annual return.

    FV = P * [((1+i)^n - 1) / i] * (1+i)   (annuity-due; i = monthly rate, n = months)
    """
    months = years * 12
    invested = monthly * months
    i = annual_return_pct / 100.0 / 12.0
    if i == 0:
        projected = float(invested)
    else:
        projected = monthly * (((1 + i) ** months - 1) / i) * (1 + i)
    return SIPProjection(
        monthly=monthly, years=years, annual_return_pct=annual_return_pct,
        invested=float(invested), projected_value=projected,
    )


# India's rough long-run CPI average. An ASSUMPTION for real-value context (the caller discloses
# it), never a prediction -- real inflation varies year to year.
DEFAULT_INFLATION_PCT = 6.0


def real_value(nominal: float, years: int,
               inflation_pct: float = DEFAULT_INFLATION_PCT) -> float:
    """Purchasing power of a future NOMINAL amount expressed in TODAY's rupees, discounting at an
    assumed annual inflation rate.

    WHY (real money, honesty): a multi-decade SIP projects a large nominal corpus, but inflation
    erodes what it buys -- ₹1 crore in 30 years does not buy what ₹1 crore buys today. Showing the
    real value stops a non-expert planning for retirement from over-reading the bare nominal number
    as its actual worth. Deterministic arithmetic on a disclosed assumption, not a forecast.
    """
    if years <= 0 or inflation_pct <= 0:
        return float(nominal)
    return nominal / ((1 + inflation_pct / 100.0) ** years)


def sip_return_context(assumed_return_pct: float,
                       benchmark: tuple[float, float] | None) -> str:
    """The one-line context caption shown under a SIP projection.

    WHY (real money, honesty; enforces this module's own docstring mandate that the UI must say
    real returns can be negative): the DOWNSIDE disclosure -- returns vary year to year and can be
    negative, so a SIP can LOSE money -- is ALWAYS included, never gated on the live market
    benchmark. The benchmark is a live SENSEX fetch that can fail (network / rate-limit on the
    hosted app); when it does, the projection must STILL carry the "not a promise, can lose money"
    honesty, or a parent sees a rosy projected gain with no downside stated. When the benchmark IS
    available it ADDS how the assumption compares to SENSEX's own long-run price return, so the
    reader can judge how aggressive it is. `benchmark` is (cagr_pct, years) or None when unavailable.
    """
    downside = ("Real fund returns vary year to year and can be negative, so a SIP can lose money, "
                "not only gain.")
    if benchmark is None:
        return downside
    cagr, years = benchmark
    note = ("well above" if assumed_return_pct > cagr + 3
            else "well below" if assumed_return_pct < cagr - 3
            else "in line with")
    return (f"For context: SENSEX's own price return over the last {years:.0f} years (live data, "
            f"price only, excludes dividends) works out to about {cagr:.1f}%/yr. Your "
            f"{assumed_return_pct:.1f}% assumption is {note} that. {downside}")
