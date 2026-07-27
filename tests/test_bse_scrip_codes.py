"""NSE-symbol -> BSE-scrip-code resolver (wires BSE announcements into the freshness ingest).

BseAnnouncementSource needs a numeric BSE scrip code, not an NSE ticker. This resolver bridges the
two: a verified static seed for the current holdings (instant, offline, no per-run network) plus a
live BSE scrip-search fallback for any new symbol. The load-bearing safety rule for a real-money
tool: NEVER return a wrong scrip (that would ingest another company's filings). Live resolution
requires an EXACT ticker match; anything else degrades to None (BSE skipped, NSE still covers it).
"""
from __future__ import annotations

from src.data.bse_scrip_codes import (
    BseScripResolver,
    SEED_SCRIP_CODES,
    parse_scrip_search,
)

# A realistic PeerSmartSearch payload: a JSON-quoted HTML string of <li> results, each with
# liclick('<scrip>','<name>') and a <span> holding "TICKER   ISIN   SCRIP". The searched prefix is
# <strong>-wrapped. This mirrors the live api.bseindia.com/.../PeerSmartSearch/w response shape.
_RELIANCE_SEARCH = (
    "\"<li class='quotemenu quotemenuselect' ng-click=\\\"liclick('500325','RELIANCE INDUSTRIES "
    "LTD')\\\"><a><strong>RELIANCE</strong> INDUSTRIES LTD<br /><span><strong>RELIANCE</strong>"
    "&nbsp;&nbsp;&nbsp;INE002A01018&nbsp;&nbsp;&nbsp;500325</span></a></li>"
    "<li class='quotemenu' ng-click=\\\"liclick('500390','RELIANCE INFRASTRUCTURE LTD')\\\">"
    "<a><strong>RELIANCE</strong> INFRASTRUCTURE LTD<br /><span>RELINFRA&nbsp;&nbsp;&nbsp;"
    "INE036A01016&nbsp;&nbsp;&nbsp;500390</span></a></li>\""
)

# Searching 'BSE' returns BSEL ALGO LTD first (ticker BSELALGO), NOT BSE Ltd -> the exact-ticker
# rule must reject it rather than pick the first result.
_BSE_SEARCH = (
    "\"<li class='quotemenu quotemenuselect' ng-click=\\\"liclick('532123','BSEL ALGO LTD')\\\">"
    "<a><strong>BSE</strong>L ALGO LTD<br /><span><strong>BSE</strong>LALGO&nbsp;&nbsp;&nbsp;"
    "INE395A01016&nbsp;&nbsp;&nbsp;532123</span></a></li>\""
)


def _never(_symbol):
    raise AssertionError("live fetcher must not be called for a seeded symbol")


def test_seed_covers_the_holdings_with_verified_codes():
    # spot-check a few of the live-verified seed entries (full set validated live 2026-07-26)
    assert SEED_SCRIP_CODES["RELIANCE"] == "500325"
    assert SEED_SCRIP_CODES["ICICIBANK"] == "532174"
    assert SEED_SCRIP_CODES["BSE"] == "543233"           # BSE Ltd: the live-search miss, seeded
    assert all(code.isdigit() for code in SEED_SCRIP_CODES.values())


def test_seeded_symbol_resolves_offline_and_normalized():
    r = BseScripResolver(fetcher=_never)
    assert r.resolve("RELIANCE") == "500325"
    assert r.resolve("reliance") == "500325"             # case-insensitive
    assert r.resolve("  ICICIBANK  ") == "532174"        # trimmed


def test_unseeded_symbol_resolves_via_live_exact_ticker():
    r = BseScripResolver(static={}, fetcher=lambda s: _RELIANCE_SEARCH)
    # exact ticker RELIANCE -> 500325, NOT the also-listed RELINFRA (500390)
    assert r.resolve("RELIANCE") == "500325"


def test_live_requires_exact_ticker_never_the_first_result():
    # 'BSE' search returns BSELALGO first; no exact 'BSE' ticker -> None, never a wrong scrip
    r = BseScripResolver(static={}, fetcher=lambda s: _BSE_SEARCH)
    assert r.resolve("BSE") is None


def test_resolver_degrades_to_none_on_fetch_failure_or_empty():
    def boom(_):
        raise RuntimeError("bseindia blocked")
    assert BseScripResolver(static={}, fetcher=boom).resolve("XYZ") is None
    assert BseScripResolver(static={}, fetcher=lambda s: None).resolve("XYZ") is None
    assert BseScripResolver(static={}, fetcher=lambda s: "garbage not html").resolve("XYZ") is None


def test_resolver_caches_live_lookups():
    calls: list[str] = []

    def f(s):
        calls.append(s)
        return _RELIANCE_SEARCH
    r = BseScripResolver(static={}, fetcher=f)
    assert r.resolve("RELIANCE") == "500325"
    assert r.resolve("RELIANCE") == "500325"
    assert len(calls) == 1                                # second lookup served from cache


def test_unresolvable_symbol_is_cached_as_a_miss():
    calls: list[str] = []

    def f(s):
        calls.append(s)
        return _BSE_SEARCH                                # never an exact match for "BSE"
    r = BseScripResolver(static={}, fetcher=f)
    assert r.resolve("BSE") is None
    assert r.resolve("BSE") is None
    assert len(calls) == 1                                # a miss is cached too (no re-fetch storm)


def test_parse_scrip_search_picks_exact_ticker():
    assert parse_scrip_search(_RELIANCE_SEARCH, "RELIANCE") == "500325"
    assert parse_scrip_search(_RELIANCE_SEARCH, "RELINFRA") == "500390"   # the 2nd li
    assert parse_scrip_search(_BSE_SEARCH, "BSE") is None                 # BSELALGO != BSE
    assert parse_scrip_search("", "RELIANCE") is None
    assert parse_scrip_search("<html>blocked</html>", "RELIANCE") is None


def test_parse_scrip_search_never_mispairs_across_a_span_less_row():
    # SAFETY (Finding 1): a span-less <li> must NOT let its scrip pair with the NEXT row's span.
    # Here BAR's ticker sits in the 2nd <li>; a naive non-greedy regex would return FOO's 111 for
    # a "BAR" search. Per-<li> isolation + the scrip cross-check must yield BAR's own 222.
    payload = (
        "\"<li class='quotemenu' ng-click=\\\"liclick('111','FOO LTD')\\\"><a><strong>FOO</strong>"
        " LTD</a></li>"                                                    # <-- NO span in this row
        "<li class='quotemenu' ng-click=\\\"liclick('222','BAR LTD')\\\"><a>BAR LTD<br />"
        "<span>BAR&nbsp;&nbsp;&nbsp;INE9X&nbsp;&nbsp;&nbsp;222</span></a></li>\""
    )
    assert parse_scrip_search(payload, "BAR") == "222"                    # its own code, not 111
    assert parse_scrip_search(payload, "FOO") is None                    # span-less row -> no guess


def test_parse_scrip_search_rejects_a_scrip_span_mismatch():
    # defense-in-depth: if the liclick scrip disagrees with the span's trailing scrip, skip the row.
    payload = ("\"<li ng-click=\\\"liclick('111','FOO LTD')\\\"><a><span>FOO&nbsp;&nbsp;&nbsp;"
               "INE9X&nbsp;&nbsp;&nbsp;999</span></a></li>\"")            # liclick 111 vs span 999
    assert parse_scrip_search(payload, "FOO") is None
