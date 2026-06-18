"""Instrument taxonomy. The research engine is instrument-agnostic; this enum is how the
rest of the system knows what kind of thing it is analyzing so it can pick the right data
adapter and the right plain-English explanation.
"""
from __future__ import annotations

from enum import Enum


class InstrumentType(str, Enum):
    STOCK = "stock"            # listed equity (NSE/BSE)
    MUTUAL_FUND = "mutual_fund"  # MF scheme (tracked via AMFI NAV)
    SIP = "sip"               # a recurring plan into a mutual fund
    IPO = "ipo"               # upcoming/ongoing public issue (analyzed from DRHP/RHP)
    OTHER = "other"           # ETFs, bonds, REITs, NPS, etc. - research-backed only

    @property
    def label(self) -> str:
        return {
            InstrumentType.STOCK: "Stock",
            InstrumentType.MUTUAL_FUND: "Mutual fund",
            InstrumentType.SIP: "SIP",
            InstrumentType.IPO: "IPO",
            InstrumentType.OTHER: "Other",
        }[self]
