"""NSE-symbol -> BSE-scrip-code resolution, so BSE announcements can join the freshness ingest.

BseAnnouncementSource is keyed by a numeric BSE scrip code (e.g. 500325), but the ingest run works
in NSE symbols (RELIANCE). This resolver bridges them:

  * SEED_SCRIP_CODES -- a static, live-verified map of the current holdings (resolved 2026-07-26 via
    BSE's own scrip search and confirmed by exact ticker). Instant, offline, and avoids a live
    scrip lookup on every scheduled run.
  * a LIVE fallback -- for any symbol not in the seed (a newly added holding), query BSE's
    PeerSmartSearch and take the scrip ONLY when a result's ticker EXACTLY equals the NSE symbol.

The load-bearing rule for a real-money tool: NEVER return a wrong scrip. A wrong code would ingest
another company's filings as this company's. So the live path demands an exact ticker match and
otherwise returns None -- BSE is simply skipped for that symbol and NSE announcements still cover
it. Every failure (blocked IP, junk HTML, no match) degrades to None; the fetcher is injectable so
tests never touch the network.
"""
from __future__ import annotations

import html
import re
from typing import Callable

# Live-verified NSE symbol -> BSE scrip code for the tracked holdings (validated 2026-07-26 against
# BSE's scrip search + exact ticker). BSE Ltd ("BSE") is seeded because its own scrip search returns
# BSELALGO first, so the live exact-ticker path can't resolve it. A symbol not on BSE (e.g. an
# NSE-only SME) is intentionally absent -> resolves to None, and BSE is skipped for it.
SEED_SCRIP_CODES: dict[str, str] = {
    "ADANIPOWER": "533096", "AKUMS": "544222", "ASTRAL": "532830", "BLS": "540073",
    "BLUEJET": "544009", "BRIGADE": "532929", "BSE": "543233", "CREDITACC": "541770",
    "ENGINERSIN": "532178", "FDC": "531599", "GRSE": "542011", "GODFRYPHLP": "500163",
    "ICICIBANK": "532174", "INDOBORAX": "524342", "JSLL": "544476", "JIOFIN": "543940",
    "JSWSTEEL": "500228", "LT": "500510", "OIL": "533106", "POWERGRID": "532898",
    "RELIANCE": "500325", "SHAKTIPUMP": "531431", "SHARDACROP": "538666", "SCILAL": "544142",
    "SBIN": "500112", "TMCV": "544569", "TMPV": "500570", "TATAPOWER": "500400",
    "VOLTAMP": "532757", "WPIL": "505872", "YESBANK": "532648",
}

_LI = re.compile(r"liclick\('(\d+)','[^']*'\).*?<span>(.*?)</span>", re.S)
_TAG = re.compile(r"<[^>]+>")


def parse_scrip_search(raw: str | bytes, symbol: str) -> str | None:
    """Parse a BSE PeerSmartSearch payload and return the scrip code whose ticker EXACTLY matches
    `symbol`, else None.

    Each result is a `<li>` carrying `liclick('<scrip>','<name>')` and a `<span>` of
    "TICKER   ISIN   SCRIP" (the searched prefix `<strong>`-wrapped). The ticker is the span's first
    token once tags/entities are stripped. Only an exact (case-insensitive) ticker match resolves --
    so searching "BSE" (which returns "BSELALGO" first) yields None, never a wrong scrip.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if not raw:
        return None
    want = (symbol or "").strip().upper()
    if not want:
        return None
    for scrip, span in _LI.findall(raw):
        text = html.unescape(_TAG.sub("", span)).replace("\xa0", " ")
        tokens = text.split()
        if tokens and tokens[0].strip().upper() == want:
            return scrip
    return None


class BseScripResolver:
    """Resolve an NSE symbol to a BSE scrip code: seed first, then a degrade-safe live lookup.

    `static` overrides/augments the built-in seed (defaults to SEED_SCRIP_CODES). `fetcher` is the
    live scrip-search fetcher (symbol -> raw HTML/JSON, or None); it is injected in tests so nothing
    hits the network. Both hits and misses are cached for the resolver's lifetime, so a scheduled
    run resolves each symbol at most once and a persistent miss never triggers a re-fetch storm.
    """

    def __init__(self, static: dict[str, str] | None = None,
                 fetcher: Callable[[str], str | None] | None = None):
        self._static = {k.strip().upper(): v for k, v in
                        (SEED_SCRIP_CODES if static is None else static).items()}
        self._fetcher = fetcher or self._http_search
        self._cache: dict[str, str | None] = {}

    def resolve(self, symbol: str) -> str | None:
        """Return the BSE scrip code for `symbol`, or None if it can't be resolved safely."""
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        if sym in self._static:
            return self._static[sym]
        if sym in self._cache:
            return self._cache[sym]
        code = self._resolve_live(sym)
        self._cache[sym] = code           # cache the miss too
        return code

    def _resolve_live(self, sym: str) -> str | None:
        try:
            raw = self._fetcher(sym)
        except Exception:
            return None
        if not raw:
            return None
        return parse_scrip_search(raw, sym)

    @staticmethod
    def _http_search(symbol: str) -> str | None:
        """Personal-use BSE scrip search (same cookie-prime + Referer pattern as the announcement
        fetcher). Returns the raw PeerSmartSearch payload, or None on any failure. A licensed feed
        or a static seed replaces this via the injected fetcher / the seed map."""
        import http.cookiejar
        import urllib.parse
        import urllib.request
        url = ("https://api.bseindia.com/BseIndiaAPI/api/PeerSmartSearch/w?Type=SS&text="
               + urllib.parse.quote(symbol))
        headers = [
            ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
            ("Accept", "application/json,text/html,*/*"),
            ("Referer", "https://www.bseindia.com/"),
            ("Origin", "https://www.bseindia.com"),
        ]
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = list(headers)
        try:
            opener.open("https://www.bseindia.com/", timeout=20)     # prime session cookies
            resp = opener.open(url, timeout=25)
            return resp.read().decode("utf-8", "replace")
        except Exception:
            return None
