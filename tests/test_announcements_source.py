"""Corporate-announcement source: parse the NSE/BSE announcement feed into dated, attributed
records. Fetcher is injected so every test is offline against a fixture that mirrors the real
NSE corporate-announcements JSON shape."""
import json

from src.data.announcements_source import (
    ANNOUNCEMENT_SOURCES,
    BSE_ANNOUNCE_SOURCE_ID,
    NSE_ANNOUNCE_SOURCE_ID,
    Announcement,
    AnnouncementSource,
    BseAnnouncementSource,
    parse_bse_announcements,
    parse_nse_announcements,
)
from src.sources.registry import CredibilityTier

# Mirrors the real NSE /api/corporate-announcements shape: a JSON array of records carrying the
# subject text (attchmntText), category (desc), attachment URL (attchmntFile) and datetime.
FIXTURE = json.dumps([
    {"symbol": "RELIANCE", "desc": "Financial Results",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/RELIANCE_19072024.pdf",
     "attchmntText": "Board Meeting Outcome - audited financial results for the year ended "
                     "March 31, 2024",
     "an_dt": "2024-07-19 18:32:00", "sort_date": "2024-07-19 18:32:00"},
    {"symbol": "RELIANCE", "desc": "Board Meeting",
     "attchmntFile": "https://nsearchives.nseindia.com/corporate/RELIANCE_BM.pdf",
     "attchmntText": "Intimation of Board Meeting to be held on July 19, 2024",
     "an_dt": "20-Jul-2024 09:00:00"},          # d-b-Y style date, no sort_date -> robustness
    {"symbol": "RELIANCE", "desc": "Dividend",
     "attchmntFile": "",                          # no attachment -> item_key falls back to composite
     "attchmntText": "Recommendation of final dividend of Rs 10 per equity share",
     "an_dt": "2024-07-19 18:40:00"},
])


def test_parse_extracts_announcement_fields():
    anns = parse_nse_announcements(FIXTURE)
    assert len(anns) == 3
    first = anns[0]
    assert isinstance(first, Announcement)
    assert first.symbol == "RELIANCE"
    assert "audited financial results" in first.title
    assert first.category == "Financial Results"
    assert first.as_of == "2024-07-19"                 # dated from the announcement datetime
    assert first.ref.endswith("RELIANCE_19072024.pdf")
    assert first.source_id == NSE_ANNOUNCE_SOURCE_ID


def test_parse_handles_alternate_date_format():
    anns = parse_nse_announcements(FIXTURE)
    # the second record uses '20-Jul-2024 09:00:00' with no sort_date
    assert anns[1].as_of == "2024-07-20"


def test_parse_tolerates_data_wrapper():
    wrapped = json.dumps({"data": json.loads(FIXTURE)})
    assert len(parse_nse_announcements(wrapped)) == 3


def test_parse_skips_titleless_and_non_dict_records():
    raw = json.dumps([
        {"symbol": "X", "desc": "", "attchmntText": "", "an_dt": "2024-07-19 10:00:00"},  # no title
        "not-a-dict",
        {"symbol": "X", "desc": "Analyst Meet", "attchmntText": "",
         "an_dt": "2024-07-19 10:00:00"},          # title falls back to category 'Analyst Meet'
    ])
    anns = parse_nse_announcements(raw)
    assert len(anns) == 1
    assert anns[0].title == "Analyst Meet"


def test_parse_bad_json_returns_empty():
    assert parse_nse_announcements("<html>blocked</html>") == []
    assert parse_nse_announcements("") == []


def test_item_key_prefers_attachment_url_then_composite():
    anns = parse_nse_announcements(FIXTURE)
    with_file = anns[0]
    without_file = anns[2]                            # the dividend record, no attachment
    assert with_file.item_key.endswith("RELIANCE_19072024.pdf")
    # no attachment -> a stable composite of symbol + date + normalized title, never empty
    assert without_file.item_key != ""
    assert "reliance" in without_file.item_key.lower()
    # a fully-empty announcement has no usable identity
    assert Announcement(symbol="", title="", as_of="", source_id=NSE_ANNOUNCE_SOURCE_ID).item_key == ""


def test_as_text_carries_symbol_and_date_for_grounding():
    ann = parse_nse_announcements(FIXTURE)[0]
    text = ann.as_text
    assert "RELIANCE" in text and "2024-07-19" in text
    assert "audited financial results" in text


def test_source_fetch_parses_via_injected_fetcher():
    src = AnnouncementSource(fetcher=lambda s: FIXTURE)
    anns = src.fetch("RELIANCE")
    assert len(anns) == 3


def test_source_degrades_on_fetcher_failure_or_empty():
    def boom(_):
        raise RuntimeError("api.nseindia blocks datacenter IPs")
    assert AnnouncementSource(fetcher=boom).fetch("RELIANCE") == []
    assert AnnouncementSource(fetcher=lambda s: None).fetch("RELIANCE") == []


def test_announcement_sources_are_primary_tier():
    # WHY (load-bearing trust): an exchange corporate announcement is official PRIMARY provenance
    # (SPEC registry: 'SEBI filings, exchange data'), not analyst/creator context.
    assert len(ANNOUNCEMENT_SOURCES) >= 1
    for source in ANNOUNCEMENT_SOURCES:
        assert source.tier == CredibilityTier.PRIMARY


# ---------------------------------------------------------------------------------------------
# BSE (H4). Mirrors the real BSE api.bseindia.com AnnGetData shape: a {"Table":[...]} wrapper of
# records carrying subject (NEWSSUB/HEADLINE), category (CATEGORYNAME), a numeric scrip code
# (SCRIP_CD), the disclosure datetime (NEWS_DT / News_submission_dt, ISO-with-T), and a bare
# attachment filename (ATTACHMENTNAME). Same Announcement dataclass, same tolerances as NSE.
BSE_FIXTURE = json.dumps({
    "Table": [
        {"SCRIP_CD": 500325, "CATEGORYNAME": "Result",
         "NEWSSUB": "Reliance Industries Ltd - 500325 - Board Meeting Outcome: audited financial "
                    "results for the year ended March 31, 2024",
         "HEADLINE": "Outcome of Board Meeting",
         "NEWS_DT": "2024-07-19T18:32:00",
         "News_submission_dt": "2024-07-19T18:32:15.123",
         "ATTACHMENTNAME": "e1f2a3b4-1111-2222-3333-abcdef012345.pdf"},
        {"SCRIP_CD": 500325, "CATEGORYNAME": "Company Update",
         "NEWSSUB": "<b>Press Release</b> - Reliance announces <i>new</i> investment",
         "HEADLINE": "",
         "NEWS_DT": "2024-07-18T10:00:00",
         "ATTACHMENTNAME": ""},                     # HTML subject + no attachment -> composite key
        {"SCRIP_CD": 500325, "CATEGORYNAME": "AGM/EGM",
         "NEWSSUB": "", "HEADLINE": "Notice of 47th Annual General Meeting",
         "NEWS_DT": "", "News_submission_dt": "",   # undated -> as_of "" (kept, never fresh)
         "ATTACHMENTNAME": "agm-notice.pdf"},        # HEADLINE fallback for the subject
    ],
})


def test_bse_parse_extracts_announcement_fields():
    anns = parse_bse_announcements(BSE_FIXTURE)
    assert len(anns) == 3
    first = anns[0]
    assert isinstance(first, Announcement)
    assert first.symbol == "500325"                  # BSE identity is the numeric scrip code
    assert "audited financial results" in first.title
    assert first.category == "Result"
    assert first.as_of == "2024-07-19"               # dated from NEWS_DT (ISO-with-T)
    assert first.ref.endswith("e1f2a3b4-1111-2222-3333-abcdef012345.pdf")
    assert first.ref.startswith("https://")          # bare ATTACHMENTNAME -> full BSE locator
    assert first.source_id == BSE_ANNOUNCE_SOURCE_ID


def test_bse_parse_strips_html_and_falls_back_to_composite_key():
    ann = parse_bse_announcements(BSE_FIXTURE)[1]
    assert "<b>" not in ann.title and "<i>" not in ann.title
    assert "Press Release" in ann.title and "new investment" in ann.title
    assert ann.ref == ""                             # no attachment
    assert ann.item_key != "" and "500325" in ann.item_key


def test_bse_parse_headline_fallback_and_undated():
    ann = parse_bse_announcements(BSE_FIXTURE)[2]
    assert ann.title == "Notice of 47th Annual General Meeting"   # NEWSSUB blank -> HEADLINE
    assert ann.as_of == ""                                        # no date -> undated, still kept


def test_bse_parse_tolerates_bare_list():
    bare = json.dumps(json.loads(BSE_FIXTURE)["Table"])
    assert len(parse_bse_announcements(bare)) == 3


def test_bse_parse_skips_subjectless_and_non_dict_records():
    raw = json.dumps({"Table": [
        {"SCRIP_CD": 1, "CATEGORYNAME": "", "NEWSSUB": "", "HEADLINE": "",
         "NEWS_DT": "2024-07-19T10:00:00"},          # no subject AND no category -> skipped
        "not-a-dict",
        {"SCRIP_CD": 1, "CATEGORYNAME": "Analyst Meet", "NEWSSUB": "", "HEADLINE": "",
         "NEWS_DT": "2024-07-19T10:00:00"},          # subject falls back to category
    ]})
    anns = parse_bse_announcements(raw)
    assert len(anns) == 1
    assert anns[0].title == "Analyst Meet"


def test_bse_parse_bad_json_returns_empty():
    assert parse_bse_announcements("<html>blocked</html>") == []
    assert parse_bse_announcements("") == []


def test_bse_http_fetch_hits_the_working_annsubcategory_endpoint(monkeypatch):
    # WHY (regression, no mistake twice): the old AnnGetData/w query returned "No Record Found!" for
    # every scrip/window (live-verified 2026-07-26). The working shape is AnnSubCategoryGetData/w
    # with subcategory=-1 and a LOWERCASE strscrip. This locks that URL without touching the network
    # (the request is captured), so a future edit can't silently revert to the dead endpoint.
    import urllib.request

    opened: list[str] = []

    class _Resp:
        def read(self):
            return b'{"Table": []}'

    class _Opener:
        addheaders: list = []

        def open(self, url, timeout=None):
            opened.append(url)
            return _Resp()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: _Opener())
    BseAnnouncementSource._http_fetch("500325")

    listing = opened[-1]                                  # the API call (after the home cookie-prime)
    assert "/AnnSubCategoryGetData/w" in listing
    assert "AnnGetData/w?" not in listing                 # never the dead endpoint
    assert "subcategory=-1" in listing
    assert "strscrip=500325" in listing                   # lowercase, the shape that returns records


def test_bse_parse_builds_source_id_and_carries_grounding_text():
    ann = parse_bse_announcements(BSE_FIXTURE, source_id="bse_custom")[0]
    assert ann.source_id == "bse_custom"
    text = ann.as_text
    assert "500325" in text and "2024-07-19" in text and "audited financial results" in text


def test_bse_source_fetch_parses_via_injected_fetcher():
    src = BseAnnouncementSource(fetcher=lambda scrip: BSE_FIXTURE)
    anns = src.fetch("500325")
    assert len(anns) == 3
    assert src.source_id == BSE_ANNOUNCE_SOURCE_ID   # defaults to the BSE feed id


def test_bse_source_degrades_on_fetcher_failure_or_empty():
    def boom(_):
        raise RuntimeError("api.bseindia blocks / 403s a cold request")
    assert BseAnnouncementSource(fetcher=boom).fetch("500325") == []
    assert BseAnnouncementSource(fetcher=lambda s: None).fetch("500325") == []
