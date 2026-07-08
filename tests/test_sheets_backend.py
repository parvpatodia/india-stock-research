from src.analysis.sizing import Stance
from src.data.sheets_backend import (
    AppsScriptGateway,
    InMemoryGateway,
    LocalJsonGateway,
    ReportRecord,
    append_log,
    read_holdings,
    read_reports,
    record_from_report,
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


def _approved_report(company="BLS International") -> Report:
    v = Verdict(ValuationTier.CHEAP, QualityTier.STRONG, Leaning.CONSTRUCTIVE,
                Confidence.MEDIUM, reasons=("P/E 14.6 vs median 32.7 reads cheap.",))
    return Report(company=company, verdict=v).approve("parv", note="looks right")


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
    save_report(gw, record_from_report(Report(company="BLS International", verdict=v).approve("parv"),
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


def test_local_json_gateway_persists_across_instances(tmp_path):
    path = tmp_path / "store.json"
    gw1 = LocalJsonGateway(path)
    save_report(gw1, record_from_report(_approved_report(), "BLS", Stance.FAVORABLE.value))
    append_log(gw1, "approved", "BLS", "parv")

    gw2 = LocalJsonGateway(path)                    # fresh instance, same file
    assert [r.symbol for r in read_reports(gw2)] == ["BLS"]
    assert gw2.read("Log")[0]["symbol"] == "BLS"
