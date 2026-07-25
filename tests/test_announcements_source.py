"""Corporate-announcement source: parse the NSE/BSE announcement feed into dated, attributed
records. Fetcher is injected so every test is offline against a fixture that mirrors the real
NSE corporate-announcements JSON shape."""
import json

from src.data.announcements_source import (
    ANNOUNCEMENT_SOURCES,
    NSE_ANNOUNCE_SOURCE_ID,
    Announcement,
    AnnouncementSource,
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
