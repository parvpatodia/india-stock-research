"""Mutual fund NAVs from AMFI (the Association of Mutual Funds in India).

AMFI publishes every scheme's NAV daily as a free public text file, no key needed. This is
a Tier-1 primary source for fund NAVs. The HTTP fetch is injectable so the parser can be
tested against a sample without the network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


@dataclass(frozen=True)
class MFScheme:
    scheme_code: str
    name: str
    nav: float
    date: str
    isin: str | None = None


def parse_navall(text: str) -> list[MFScheme]:
    """Parse the AMFI NAVAll dump. Data rows are ';'-separated with a numeric scheme code;
    fund-house and category header lines (and the column header) are skipped automatically."""
    schemes: list[MFScheme] = []
    for line in text.splitlines():
        if ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue  # skips the "Scheme Code;..." header and any non-data line
        try:
            nav = float(parts[4])
        except ValueError:
            continue  # NAV reported as 'N.A.' -> skip rather than guess
        # WHY: float('nan')/float('inf') parse silently; a 0 NAV is a not-yet-priced scheme.
        # Any of these would show as a Tier-1 fact, so reject rather than display a bad NAV.
        if not math.isfinite(nav) or nav <= 0:
            continue
        schemes.append(MFScheme(
            scheme_code=parts[0],
            name=parts[3],
            nav=nav,
            date=parts[5],
            isin=parts[1] or None,
        ))
    return schemes


class AMFIProvider:
    def __init__(self, fetcher: Callable[[], str] | None = None):
        self._fetcher = fetcher or self._http_fetch
        self._schemes: list[MFScheme] = []

    @staticmethod
    def _http_fetch() -> str:
        import urllib.request
        with urllib.request.urlopen(NAVALL_URL, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def load(self, text: str | None = None) -> int:
        raw = text if text is not None else self._fetcher()
        self._schemes = parse_navall(raw)
        return len(self._schemes)

    def get_by_code(self, scheme_code: str) -> MFScheme | None:
        return next((s for s in self._schemes if s.scheme_code == str(scheme_code)), None)

    def search(self, query: str, limit: int = 10) -> list[MFScheme]:
        """Match schemes whose name contains EVERY word of the query (order-independent).

        WHY (real money workflow, live-verified): a parent types a fund's words naturally -- in a
        different order, or with the plan/option words ("direct", "growth") that the AMFI scheme
        name splits apart with " - " separators. A single contiguous-substring test returned NOTHING
        for common, correct queries like "hdfc small cap direct growth" (the real name is "HDFC Small
        Cap Fund - Direct Plan - Growth", so the typed words are never contiguous). Requiring each
        query WORD to appear -- the standard search behaviour -- surfaces the fund the parent means.
        """
        # WHY replace("-", " "): a parent commonly hyphenates cap categories ("large-cap", "mid-cap",
        # "flexi-cap") or joins words with a hyphen; splitting only on whitespace left the hyphen in
        # the token so it never matched a name that separates the words with a space (or vice versa).
        # Treat a hyphen as a space. An "&" (e.g. L&T) is deliberately NOT split, so it still matches a
        # name that contains it rather than over-broadening into two single-character tokens.
        terms = query.strip().lower().replace("-", " ").split()
        if not terms:
            return []
        out: list[MFScheme] = []
        for s in self._schemes:
            name = s.name.lower()
            if all(t in name for t in terms):
                out.append(s)
                if len(out) >= limit:
                    break
        return out
