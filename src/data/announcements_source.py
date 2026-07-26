"""Corporate-announcement / filing source (NSE/BSE).

Exchange corporate announcements (results, board meetings, dividends, allotments, AGM notices,
rating actions) are the FRESHEST free primary-provenance feed: the exchange is the authoritative
publisher of "company X disclosed Y on date Z". Each announcement is recorded dated and
attributed; the freshness log stores this provenance, it does not promote an announcement's text
to a verified number (that still needs the grounding + cross-verification path).

Tier: PRIMARY. An exchange disclosure is official filing provenance (SPEC v4 registry:
"SEBI filings, exchange/AMFI data"), not analyst or creator context.

ToS reality (SPEC v4 §5): the exchange JSON endpoints (api.nseindia.com / bseindia.com) are
personal-use, non-redistributable, and block datacenter IPs; hitting them at volume is against
ToS. This is fine for a personal/parents tool run from a residential IP, and WRONG for a public
multi-user product, which needs a licensed feed (GDFL/TrueData/indianapi). So the fetcher is an
INJECTABLE seam: the default hits NSE lightly for personal use, and a licensed feed swaps in
without touching the parser or the ingestion path. Tests never touch the network -- the fetcher
is injected against a fixture that mirrors the real announcement JSON shape.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from ..sources.registry import CredibilityTier, Source

# Registered feed ids. PRIMARY tier: official exchange disclosure.
NSE_ANNOUNCE_SOURCE_ID = "nse_announcements"
BSE_ANNOUNCE_SOURCE_ID = "bse_announcements"

ANNOUNCEMENT_SOURCES: tuple[Source, ...] = (
    Source(id=NSE_ANNOUNCE_SOURCE_ID, name="NSE corporate announcements",
           tier=CredibilityTier.PRIMARY, url="https://www.nseindia.com",
           notes="Official exchange disclosure feed (results, board meetings, dividends, "
                 "allotments, AGM notices). Personal-use / non-redistributable per exchange ToS; "
                 "a licensed feed swaps in behind the injectable fetcher for a public product."),
    Source(id=BSE_ANNOUNCE_SOURCE_ID, name="BSE corporate announcements",
           tier=CredibilityTier.PRIMARY, url="https://www.bseindia.com",
           notes="Official exchange disclosure feed. Same personal-use ToS reality as NSE; "
                 "same injectable-fetcher seam for a licensed feed."),
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_KEY = re.compile(r"[^a-z0-9]+")
# NSE announcement datetimes seen in the wild: '2024-07-19 18:32:00' (sort_date, ISO-ish) and
# '19-Jul-2024 18:32:00' (an_dt). Parse both; anything else degrades to "" (undated, never fresh).
_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%d-%m-%Y %H:%M:%S")


def _clean(text: str | None) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _to_iso_date(value: str | None) -> str:
    """Return YYYY-MM-DD, or "" if the date can't be parsed."""
    if not value:
        return ""
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return ""


@dataclass(frozen=True)
class Announcement:
    symbol: str
    title: str             # the announcement subject text (or its category if the subject is blank)
    as_of: str             # ISO date (YYYY-MM-DD) it was disclosed, or "" if undated
    source_id: str = NSE_ANNOUNCE_SOURCE_ID
    ref: str = ""          # attachment / filing URL locator
    category: str = ""     # e.g. "Financial Results", "Board Meeting", "Dividend"

    @property
    def item_key(self) -> str:
        """A stable logical identity. The attachment URL is unique per filing, so it's the key;
        when a filing carries no attachment, fall back to a composite of symbol + date + a
        normalized title so distinct announcements coexist and an identical re-fetch dedups. An
        announcement with neither a ref nor a title has no usable identity -> "" (the caller
        counts it as an error and skips it, per the degrade-per-item rule)."""
        ref = (self.ref or "").strip()
        if ref:
            return ref[:200]
        title_key = _KEY.sub("-", self.title.lower()).strip("-")[:120]
        if not title_key:
            return ""
        return f"{self.symbol.strip().upper()}|{self.as_of}|{title_key}"[:200]

    @property
    def as_text(self) -> str:
        """Document text ingested for provenance: symbol + date + category up front so any answer
        drawn from it carries the source and 'as of' date."""
        sym = self.symbol.strip().upper() or "unknown"
        when = self.as_of or "undated"
        head = f"[NSE/BSE announcement - {sym}, {when}]"
        body = f"{self.category}: {self.title}" if self.category else self.title
        return f"{head} {body}".strip()


def parse_nse_announcements(raw: str | bytes,
                           source_id: str = NSE_ANNOUNCE_SOURCE_ID) -> list[Announcement]:
    """Parse the NSE corporate-announcements JSON into dated, attributed Announcements.

    Accepts either a bare JSON array (the live shape) or a ``{"data": [...]}`` wrapper (defensive).
    A record with no subject text AND no category is skipped (nothing usable to record). Bad JSON
    (an HTML block page) parses to [] so the caller abstains, never crashes.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    out: list[Announcement] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        category = _clean(rec.get("desc") or rec.get("sm_desc"))
        title = _clean(rec.get("attchmntText") or rec.get("smText")) or category
        if not title:
            continue
        as_of = _to_iso_date(rec.get("sort_date") or rec.get("an_dt") or rec.get("dt"))
        ref = str(rec.get("attchmntFile") or rec.get("attchmnt") or "").strip()
        out.append(Announcement(symbol=_clean(rec.get("symbol")), title=title, as_of=as_of,
                                source_id=source_id, ref=ref, category=category))
    return out


# BSE returns ATTACHMENTNAME as a bare filename (a GUID.pdf), not a URL; live attachments hang off
# this public path. Building the full locator makes `ref` a usable, unique-per-filing item_key,
# mirroring the full-URL ref the NSE parser stores.
_BSE_ATTACH_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def _bse_attachment_ref(name: str | None) -> str:
    """Turn BSE's bare ATTACHMENTNAME into the public attachment URL. A value that is already a URL
    is passed through; a blank stays blank (item_key then falls back to the composite)."""
    text = str(name or "").strip()
    if not text:
        return ""
    if text.lower().startswith(("http://", "https://")):
        return text
    return _BSE_ATTACH_BASE + text


def parse_bse_announcements(raw: str | bytes,
                           source_id: str = BSE_ANNOUNCE_SOURCE_ID) -> list[Announcement]:
    """Parse the BSE corporate-announcements JSON (bseindia.com AnnGetData shape) into the SAME
    dated, attributed Announcement dataclass, with the SAME tolerances as ``parse_nse_announcements``.

    Accepts BSE's ``{"Table": [...]}`` wrapper (the live shape) or a bare JSON array (defensive).
    Field mapping: subject = NEWSSUB or HEADLINE; category = CATEGORYNAME; disclosure date =
    NEWS_DT or News_submission_dt (ISO-with-T, handled by the shared date parser); attachment =
    ATTACHMENTNAME (a bare filename, built into a full URL); the record's numeric SCRIP_CD is
    echoed back as the symbol (BSE identity is a scrip code, not an NSE ticker). A record with no
    subject AND no category is skipped. Bad JSON (an HTML block page) parses to [] so the caller
    abstains, never crashes."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return []
    records = payload.get("Table") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    out: list[Announcement] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        category = _clean(rec.get("CATEGORYNAME"))
        title = _clean(rec.get("NEWSSUB") or rec.get("HEADLINE")) or category
        if not title:
            continue
        as_of = _to_iso_date(rec.get("NEWS_DT") or rec.get("News_submission_dt"))
        ref = _bse_attachment_ref(rec.get("ATTACHMENTNAME"))
        symbol = _clean(str(rec.get("SCRIP_CD") or ""))
        out.append(Announcement(symbol=symbol, title=title, as_of=as_of,
                                source_id=source_id, ref=ref, category=category))
    return out


class AnnouncementSource:
    """Fetch a symbol's recent corporate announcements as dated, attributed records.

    The fetcher is injected (offline in tests). It degrades to [] on any failure -- the exchange
    endpoint 403s / times out / block-pages a datacenter IP routinely, and a filing feed going
    dark must never crash the caller (LESSONS 2026-07-09: abstain at the fetch boundary).
    """

    def __init__(self, fetcher: Callable[[str], str | None] | None = None,
                 source_id: str = NSE_ANNOUNCE_SOURCE_ID):
        self._fetcher = fetcher or self._http_fetch
        self.source_id = source_id

    @staticmethod
    def _http_fetch(symbol: str) -> str | None:
        """Personal-use NSE fetch: prime session cookies from the home page (the API 403s a cold
        request), then read the announcements JSON. Same cookie-priming pattern the annual-report
        resolver uses. A licensed feed replaces this whole method via the injected fetcher."""
        import http.cookiejar
        import urllib.request
        home = "https://www.nseindia.com/"
        listing = ("https://www.nseindia.com/api/corporate-announcements"
                   f"?index=equities&symbol={symbol}")
        headers = [
            ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
            ("Accept", "application/json,text/html,*/*"),
            ("Accept-Language", "en-US,en;q=0.9"),
        ]
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = list(headers)
        try:
            opener.open(home, timeout=20)             # prime session cookies
            resp = opener.open(listing, timeout=25)
            return resp.read().decode("utf-8", "replace")
        except Exception:
            return None

    def fetch(self, symbol: str) -> list[Announcement]:
        try:
            raw = self._fetcher(symbol.strip().upper())
        except Exception:
            return []
        if not raw:
            return []
        return parse_nse_announcements(raw, source_id=self.source_id)


class BseAnnouncementSource(AnnouncementSource):
    """BSE variant of :class:`AnnouncementSource`: same injectable-fetcher / degrade-to-[] contract,
    but the caller supplies a numeric BSE SCRIP CODE (e.g. "500325" for Reliance), not an NSE
    symbol, and the payload is BSE's AnnGetData JSON parsed by ``parse_bse_announcements``. No
    symbol->scrip mapping lives here (out of scope): the fetcher takes whatever identifier the
    caller passes through, so a licensed feed or a resolver can supply it.
    """

    def __init__(self, fetcher: Callable[[str], str | None] | None = None,
                 source_id: str = BSE_ANNOUNCE_SOURCE_ID):
        # self._http_fetch resolves polymorphically to the BSE fetch below when no fetcher injected.
        super().__init__(fetcher=fetcher, source_id=source_id)

    @staticmethod
    def _http_fetch(scrip: str) -> str | None:
        """Personal-use BSE fetch: prime session cookies from the home page, then read the
        AnnGetData corporate-announcements JSON for a scrip code. BSE's api.bseindia.com 403s a
        cold request and requires a bseindia.com Referer + browser-like headers (the BSE analogue
        of the NSE cookie-priming). A licensed feed replaces this whole method via the injected
        fetcher.

        NOTE (unverified offline): the exact AnnGetData query params and JSON field names below are
        the publicly-documented BSE shape but are NOT exercised by the test suite (tests inject a
        fixture, never the network). The parser tolerates the wrapper/bare-list, multiple date
        fields, HTML, and missing fields, so drift in the live shape degrades to [] not a crash."""
        import http.cookiejar
        import urllib.request
        from datetime import date, timedelta

        home = "https://www.bseindia.com/"
        today = date.today()
        frm = (today - timedelta(days=30)).strftime("%Y%m%d")
        to = today.strftime("%Y%m%d")
        listing = ("https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
                   f"?pageno=1&strCat=-1&strPrevDate={frm}&strScrip={scrip}"
                   f"&strSearch=P&strToDate={to}&strType=C")
        headers = [
            ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
            ("Accept", "application/json,text/html,*/*"),
            ("Accept-Language", "en-US,en;q=0.9"),
            ("Referer", "https://www.bseindia.com/"),  # api.bseindia.com blocks a missing Referer
            ("Origin", "https://www.bseindia.com"),
        ]
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = list(headers)
        try:
            opener.open(home, timeout=20)             # prime session cookies
            resp = opener.open(listing, timeout=25)
            return resp.read().decode("utf-8", "replace")
        except Exception:
            return None

    def fetch(self, scrip: str) -> list[Announcement]:
        try:
            raw = self._fetcher(str(scrip).strip())   # scrip codes are numeric; no upper-casing
        except Exception:
            return []
        if not raw:
            return []
        return parse_bse_announcements(raw, source_id=self.source_id)
