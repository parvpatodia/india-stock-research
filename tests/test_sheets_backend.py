import pytest

from src.analysis.sizing import Stance
from src.data.sheets_backend import (
    AppsScriptGateway,
    GspreadGateway,
    InMemoryGateway,
    LocalJsonGateway,
    ReportRecord,
    append_log,
    read_holdings,
    read_reports,
    record_from_report,
    resolve_approved_stances,
    save_report,
)
from src.research.report import (
    Confidence,
    Leaning,
    QualityTier,
    Report,
    ValuationTier,
    Verdict,
)
from src.research.verification import SourcedValue, verify_figure


def _approved_report(company="BLS International") -> Report:
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                Confidence.MEDIUM, reasons=("P/E 14.6 vs median 32.7 reads cheap.",))
    fig = verify_figure("net_profit", [SourcedValue(100, "yfinance"), SourcedValue(101, "screener")])
    return Report(company=company, figures=(fig,), verdict=v).approve("parv", note="looks right")


def _persisted_approval(symbol="BLS", stance="favorable") -> ReportRecord:
    return ReportRecord(symbol=symbol, company="BLS International", valuation="cheap",
                        quality="strong", leaning="constructive", confidence="medium",
                        stance=stance, status="approved", reviewer="parv",
                        updated_at="2026-01-01T00:00:00Z")


def _draft_report(company="BLS International") -> Report:
    # WHY: a fresh re-research this session that has NOT (yet) been reviewed -- DRAFT, untrusted.
    v = Verdict(ValuationTier.FAIR, QualityTier.MIXED, Leaning.NEUTRAL, Confidence.MEDIUM)
    return Report(company=company, verdict=v)


def test_resolve_approved_stances_persisted_only():
    result = resolve_approved_stances([_persisted_approval()], {})
    assert result == {"BLS": Stance.FAVORABLE}


def test_resolve_approved_stances_fresh_trusted_session_report_supersedes_persisted():
    persisted = [_persisted_approval(stance="unfavorable")]
    fresh = _approved_report()   # FAVORABLE, trusted (see helper above: CONSTRUCTIVE leaning)
    result = resolve_approved_stances(persisted, {"BLS (live/x)": fresh})
    assert result == {"BLS": Stance.FAVORABLE}


def test_resolve_approved_stances_fresh_untrusted_report_clears_stale_persisted_approval():
    # WHY (real money, the actual bug): a symbol approved in a PAST session, re-researched THIS
    # session into a fresh, not-yet-reviewed DRAFT, must not keep feeding suggest_allocation with
    # the old approval -- the current (unreviewed) analysis supersedes it, dropping it entirely.
    persisted = [_persisted_approval()]
    result = resolve_approved_stances(persisted, {"BLS (live/x)": _draft_report()})
    assert result == {}


def test_resolve_approved_stances_untouched_symbol_keeps_persisted_approval():
    persisted = [_persisted_approval("BLS"), _persisted_approval("HDFCBANK", stance="neutral")]
    result = resolve_approved_stances(persisted, {"TCS (live/x)": _approved_report("TCS")})
    assert result["BLS"] == Stance.FAVORABLE
    assert result["HDFCBANK"] == Stance.NEUTRAL
    assert result["TCS"] == Stance.FAVORABLE


def test_record_from_report_captures_verdict_stance_reviewer():
    rec = record_from_report(_approved_report(), "bls", Stance.FAVORABLE.value)
    assert rec.symbol == "BLS"                 # normalized upper
    assert rec.company == "BLS International"
    assert rec.valuation == "cheap" and rec.quality == "strong"
    assert rec.stance == "favorable" and rec.status == "approved"
    assert rec.reviewer == "parv" and rec.note == "looks right"


def test_report_record_row_round_trip():
    rec = record_from_report(_approved_report(), "BLS", Stance.FAVORABLE.value)
    assert ReportRecord.from_row(rec.as_row()) == rec


def test_save_report_upserts_by_symbol():
    gw = InMemoryGateway()
    save_report(gw, record_from_report(_approved_report(), "BLS", Stance.FAVORABLE.value))
    save_report(gw, record_from_report(_approved_report("Brigade"), "BRIGADE", Stance.FAVORABLE.value))
    assert len(read_reports(gw)) == 2

    # re-save BLS with a different stance -> replaces, does not duplicate
    v = Verdict(ValuationTier.FAIR, QualityTier.MIXED, Leaning.NEUTRAL, Confidence.MEDIUM)
    fig = verify_figure("net_profit", [SourcedValue(100, "yfinance"), SourcedValue(101, "screener")])
    save_report(gw, record_from_report(
        Report(company="BLS International", figures=(fig,), verdict=v).approve("parv"),
        "BLS", Stance.NEUTRAL.value))
    records = {r.symbol: r for r in read_reports(gw)}
    assert len(records) == 2
    assert records["BLS"].stance == "neutral"       # updated in place


def test_read_holdings_parses_sheet_records():
    gw = InMemoryGateway({"Holdings": [
        {"Symbol": "RELIANCE", "Quantity": 10, "Avg Cost": 2400, "Sector": "Energy"},
        {"Symbol": "SBIN", "Quantity": 5, "Avg Cost": 600, "Sector": ""},
    ]})
    holdings = read_holdings(gw)
    assert [h.symbol for h in holdings] == ["RELIANCE", "SBIN"]
    assert holdings[0].sector == "Energy"
    assert holdings[1].sector == "Unknown"          # blank -> loader default


def test_read_holdings_empty_is_empty():
    assert read_holdings(InMemoryGateway()) == []


def test_append_log_records_action_and_timestamp():
    gw = InMemoryGateway()
    append_log(gw, "approved", "BLS", "parv", note="looks right")
    rows = gw.read("Log")
    assert len(rows) == 1
    assert rows[0]["action"] == "approved" and rows[0]["symbol"] == "BLS"
    assert rows[0]["reviewer"] == "parv" and rows[0]["timestamp"]


def test_apps_script_gateway_maps_actions_to_transport():
    # Fake the web app with an in-memory store; assert read/write/append map to the right calls.
    backing = InMemoryGateway()
    tokens_seen = []

    def getter(params):
        tokens_seen.append(params.get("token"))
        assert params["action"] == "read"
        return backing.read(params["tab"])

    def poster(payload):
        tokens_seen.append(payload.get("token"))
        if payload["action"] == "write":
            backing.write(payload["tab"], payload["header"], payload["rows"])
        elif payload["action"] == "append":
            backing.append(payload["tab"], payload["header"], payload["row"])
        return {"ok": True}

    gw = AppsScriptGateway("https://script/exec", "sekret", getter=getter, poster=poster)
    save_report(gw, record_from_report(_approved_report(), "BLS", Stance.FAVORABLE.value))
    append_log(gw, "approved", "BLS", "parv")
    assert [r.symbol for r in read_reports(gw)] == ["BLS"]
    assert gw.read("Log")[0]["symbol"] == "BLS"
    assert set(tokens_seen) == {"sekret"}          # every call carried the shared token


def test_apps_script_retries_transient_failure_then_succeeds():
    # WHY (regression 2026-07-09): Apps Script web apps cold-start slowly; the 09:49 daily run
    # died on a single 20s ReadTimeout. A transient blip must be retried, not fatal.
    calls = {"n": 0}
    slept = []

    def flaky_getter(params):
        calls["n"] += 1
        if calls["n"] < 3:                       # fail twice, succeed on the 3rd
            raise TimeoutError("cold start")
        return [{"symbol": "BLS"}]

    gw = AppsScriptGateway("https://script/exec", "sekret", getter=flaky_getter,
                           attempts=3, backoff=0.0, sleeper=slept.append)
    assert gw.read("Today") == [{"symbol": "BLS"}]
    assert calls["n"] == 3
    assert len(slept) == 2                        # backed off before each retry, not after success


def test_apps_script_gives_up_after_attempts_and_raises():
    def always_fails(params):
        raise TimeoutError("still cold")

    gw = AppsScriptGateway("https://script/exec", "sekret", getter=always_fails,
                           attempts=3, backoff=0.0, sleeper=lambda s: None)
    import pytest
    with pytest.raises(TimeoutError):
        gw.read("Today")


class _FakeWorksheet:
    """Stands in for a gspread Worksheet so GspreadGateway.write() is testable without a real
    Google Sheet connection (GspreadGateway.__init__ makes a live network call)."""

    def __init__(self, update_should_fail: bool = False):
        self.cleared = False
        self.updated_grid: list | None = None
        self.resized_to: int | None = None
        self._update_should_fail = update_should_fail

    def clear(self):
        self.cleared = True

    def update(self, grid):
        if self._update_should_fail:
            raise TimeoutError("simulated transient Google Sheets API failure")
        self.updated_grid = grid

    def resize(self, rows=None, cols=None):
        self.resized_to = rows


def _gateway_with_fake_worksheet(ws: _FakeWorksheet) -> GspreadGateway:
    # WHY: GspreadGateway.__init__ opens a real Sheet over the network, so it can't be
    # constructed normally in a test; bypass __init__ and inject a fake worksheet directly.
    gw = object.__new__(GspreadGateway)
    gw._worksheet = lambda tab, header=None: ws
    return gw


def test_gspread_write_does_not_wipe_the_tab_if_the_update_fails():
    # WHY (data-loss resilience): the old clear()-then-update() sequence would leave the tab
    # PERMANENTLY EMPTY if update() failed partway (network blip, quota, transient Google API
    # error) -- clear() had already wiped it with no way to recover. This is the parents'
    # Reports history / daily-picks tab; a transient API hiccup must not silently destroy it.
    ws = _FakeWorksheet(update_should_fail=True)
    gw = _gateway_with_fake_worksheet(ws)
    with pytest.raises(TimeoutError):
        gw.write("Reports", ["symbol", "stance"], [{"symbol": "BLS", "stance": "favorable"}])
    assert ws.cleared is False   # must NOT have wiped the tab before the failed write landed


def test_gspread_write_succeeds_and_trims_stale_rows_after():
    # The new data must land via a plain overwrite (no preceding clear), and stale trailing rows
    # from a previously-larger tab are trimmed AFTER the write succeeds, via resize -- a much
    # lower-stakes secondary step than clearing before the write.
    ws = _FakeWorksheet()
    gw = _gateway_with_fake_worksheet(ws)
    gw.write("Reports", ["symbol", "stance"], [{"symbol": "BLS", "stance": "favorable"}])
    assert ws.updated_grid == [["symbol", "stance"], ["BLS", "favorable"]]
    assert ws.resized_to == 2   # header + 1 data row
    assert ws.cleared is False


def test_local_json_gateway_persists_across_instances(tmp_path):
    path = tmp_path / "store.json"
    gw1 = LocalJsonGateway(path)
    save_report(gw1, record_from_report(_approved_report(), "BLS", Stance.FAVORABLE.value))
    append_log(gw1, "approved", "BLS", "parv")

    gw2 = LocalJsonGateway(path)                    # fresh instance, same file
    assert [r.symbol for r in read_reports(gw2)] == ["BLS"]
    assert gw2.read("Log")[0]["symbol"] == "BLS"
