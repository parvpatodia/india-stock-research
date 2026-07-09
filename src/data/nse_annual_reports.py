"""Resolve a company's latest annual-report PDF URL from NSE, so the annual-report source can
run automatically per symbol (no manually pasted URL).

NSE's annual-reports listing endpoint needs a primed session cookie (a plain request is
blocked). We prime cookies from the NSE home page, then read the JSON listing and pick the
latest year's PDF. The network call is injectable so the selection logic is tested offline.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from ..sources.adapters import HttpDocumentAdapter
from .annual_report_source import AnnualReportFigureSource
from .figure_sources import FigureSource

_HOME = "https://www.nseindia.com/"
_LISTING = "https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
_HEADERS = [
    ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
    ("Accept", "text/html,application/json,*/*"),
    ("Accept-Language", "en-US,en;q=0.9"),
]
_YEAR = re.compile(r"(\d{4})")


def _year(value) -> int:
    m = _YEAR.search(str(value or ""))
    return int(m.group(1)) if m else -1


class NseAnnualReportResolver:
    def __init__(self, fetcher: Callable[[str], str | None] | None = None):
        self._fetcher = fetcher or self._http_fetch

    @staticmethod
    def _http_fetch(symbol: str) -> str | None:
        import http.cookiejar
        import urllib.request
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = list(_HEADERS)
        try:
            opener.open(_HOME, timeout=20)  # prime session cookies
            resp = opener.open(_LISTING.format(symbol=symbol), timeout=25)
            return resp.read().decode("utf-8", "replace")
        except Exception:
            return None

    def latest_report_url(self, symbol: str) -> str | None:
        raw = self._fetcher(symbol.strip().upper())
        if not raw:
            return None
        try:
            records = json.loads(raw).get("data", [])
        except (json.JSONDecodeError, ValueError, AttributeError):
            return None
        best_url, best_year = None, -1
        for rec in records:
            url = rec.get("fileName")
            year = _year(rec.get("toYr"))
            if url and str(url).lower().endswith(".pdf") and year > best_year:
                best_url, best_year = url, year
        return best_url


def nse_annual_report_source(client=None,
                             resolver: NseAnnualReportResolver | None = None,
                             adapter: HttpDocumentAdapter | None = None) -> FigureSource:
    """An AnnualReportFigureSource whose text is auto-fetched: resolve the symbol's latest AR
    URL from NSE, then download and extract. Fetched text is memoized per symbol."""
    resolver = resolver or NseAnnualReportResolver()
    adapter = adapter or HttpDocumentAdapter("annual_report")
    cache: dict[str, str | None] = {}

    def text_provider(symbol: str) -> str | None:
        key = symbol.strip().upper()
        if key in cache:
            return cache[key]
        url = resolver.latest_report_url(key)
        text = None
        if url:
            docs = adapter.fetch(url)
            text = docs[0].text if docs else None
        cache[key] = text
        return text

    return AnnualReportFigureSource(text_provider, client=client)


def fetch_annual_report_text(symbol: str, url: str = "",
                             resolver: "NseAnnualReportResolver | None" = None,
                             adapter: HttpDocumentAdapter | None = None) -> str | None:
    """Raw annual-report text for grounded reading: from `url` if given, else resolve the latest
    from NSE. Returns None if unavailable (e.g. NSE blocks the server IP), so the caller abstains."""
    adapter = adapter or HttpDocumentAdapter("annual_report")
    target = url.strip()
    if not target:
        resolver = resolver or NseAnnualReportResolver()
        target = resolver.latest_report_url(symbol.strip().upper()) or ""
    if not target:
        return None
    # WHY (real money): the docstring promises None on failure so the caller abstains, but the
    # fetch (urlopen + pypdf) can raise on a timeout / 403 / unparseable PDF. Honor the contract
    # so the "Read the annual report" button degrades to "couldn't fetch", never a stack trace.
    try:
        docs = adapter.fetch(target)
    except Exception:
        return None
    return docs[0].text if docs else None
