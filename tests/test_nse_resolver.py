import json

from src.data.nse_annual_reports import AnnualReportRef, NseAnnualReportResolver

LISTING = json.dumps({"data": [
    {"toYr": "2024", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_2_X_2024.pdf"},
    {"toYr": "2026", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_1_X_2026.pdf"},
    {"toYr": "2025", "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_3_X_2025.pdf"},
]})


def test_resolver_picks_latest_year():
    r = NseAnnualReportResolver(fetcher=lambda s: LISTING)
    assert r.latest_report_url("X").endswith("AR_1_X_2026.pdf")


def test_resolver_blocked_returns_none():
    assert NseAnnualReportResolver(fetcher=lambda s: None).latest_report_url("X") is None


def test_resolver_ignores_non_pdf():
    listing = json.dumps({"data": [{"toYr": "2026", "fileName": "https://x/notapdf.html"}]})
    assert NseAnnualReportResolver(fetcher=lambda s: listing).latest_report_url("X") is None


def test_resolver_bad_json_returns_none():
    assert NseAnnualReportResolver(fetcher=lambda s: "<html>blocked</html>").latest_report_url("X") is None


def test_latest_report_returns_ref_with_fy_end_as_of():
    # The freshness log needs the report's coverage date, not just its URL. Indian fiscal years
    # end 31 March, so a report tagged toYr=2026 is dated 2026-03-31 (conservative as-of: the
    # report cannot be effective before its financial year ends).
    ref = NseAnnualReportResolver(fetcher=lambda s: LISTING).latest_report("X")
    assert isinstance(ref, AnnualReportRef)
    assert ref.symbol == "X"
    assert ref.fiscal_year == 2026
    assert ref.url.endswith("AR_1_X_2026.pdf")
    assert ref.as_of == "2026-03-31"


def test_latest_report_none_when_blocked():
    assert NseAnnualReportResolver(fetcher=lambda s: None).latest_report("X") is None


def test_latest_report_url_still_delegates_to_latest_report():
    r = NseAnnualReportResolver(fetcher=lambda s: LISTING)
    assert r.latest_report_url("X") == r.latest_report("X").url
