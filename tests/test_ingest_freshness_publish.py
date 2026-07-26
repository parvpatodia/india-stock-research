"""H7: the publish step of the scheduled ingest entrypoint.

After ingesting news + announcements + AR per symbol, the Mac-side run projects a snapshot and
writes it to the Sheets backend's Freshness tab. These tests exercise that wiring offline with an
in-memory gateway (no live NSE/Google, no real Sheet): projection from run results, a degrade-safe
publish, env-driven gateway resolution, and that the --publish flag is stripped from symbol args.
"""
from __future__ import annotations

import scripts.ingest_freshness as ing
from src.data.sheets_backend import AppsScriptGateway, InMemoryGateway
from src.freshness.filings_ingest import AnnualReportIngestResult, FilingsIngestSummary
from src.freshness.news_ingest import IngestSummary
from src.freshness.snapshot import FRESHNESS_TAB, parse_snapshot


def _result(symbol):
    return ing.SymbolFreshness(
        symbol=symbol,
        news=IngestSummary(new=3, superseded=1, skipped=2),
        announcements=FilingsIngestSummary(new=2, skipped=4),
        annual_report=AnnualReportIngestResult(found=True, fiscal_year=2026, as_of="2026-03-31"),
    )


def test_build_snapshot_maps_every_result():
    results = [_result("RELIANCE"), _result("INFY")]
    snaps = ing.build_snapshot(results, checked_at="2026-07-26", window_days=120)
    assert [s.symbol for s in snaps] == ["RELIANCE", "INFY"]
    assert snaps[0].news_recent == 3 + 1 + 2
    assert snaps[0].annual_report_fy == 2026


def test_publish_snapshot_writes_a_readable_freshness_tab():
    gw = InMemoryGateway()
    snaps = ing.build_snapshot([_result("RELIANCE")], checked_at="2026-07-26", window_days=120)
    assert ing.publish_snapshot(gw, snaps) is True
    parsed = parse_snapshot(gw.read(FRESHNESS_TAB))
    assert "RELIANCE" in parsed
    assert parsed["RELIANCE"].annual_report_as_of == "2026-03-31"


def test_publish_snapshot_degrades_when_the_gateway_raises():
    class _Boom(InMemoryGateway):
        def write(self, tab, header, rows):
            raise RuntimeError("sheet unreachable")

    snaps = ing.build_snapshot([_result("RELIANCE")], checked_at="2026-07-26", window_days=120)
    # a publish failure is best-effort: the ingest already succeeded, so this must not raise
    assert ing.publish_snapshot(_Boom(), snaps) is False


def test_gateway_from_env_builds_apps_script_when_configured():
    gw = ing.gateway_from_env({"APPS_SCRIPT_URL": "https://x/exec", "APPS_SCRIPT_TOKEN": "secret"})
    assert isinstance(gw, AppsScriptGateway)


def test_gateway_from_env_none_when_unconfigured():
    assert ing.gateway_from_env({}) is None
    assert ing.gateway_from_env({"APPS_SCRIPT_URL": "https://x/exec"}) is None  # token missing


def test_publish_flag_is_stripped_from_symbol_args():
    remaining, publish = ing.resolve_publish(["--publish", "RELIANCE", "INFY"])
    assert publish is True
    assert remaining == ["RELIANCE", "INFY"]
    remaining2, publish2 = ing.resolve_publish(["RELIANCE"])
    assert publish2 is False
    assert remaining2 == ["RELIANCE"]
