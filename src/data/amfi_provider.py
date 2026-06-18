"""Mutual fund NAVs from AMFI (the Association of Mutual Funds in India).

AMFI publishes every scheme's NAV daily as a free public text file, no key needed. This is
a Tier-1 primary source for fund NAVs. The HTTP fetch is injectable so the parser can be
tested against a sample without the network.
"""
from __future__ import annotations

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
        q = query.strip().lower()
        if not q:
            return []
        return [s for s in self._schemes if q in s.name.lower()][:limit]
