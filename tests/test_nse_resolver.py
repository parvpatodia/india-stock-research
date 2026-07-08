import json

from src.data.nse_annual_reports import NseAnnualReportResolver

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
