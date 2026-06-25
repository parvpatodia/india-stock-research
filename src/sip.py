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
