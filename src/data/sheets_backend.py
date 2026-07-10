"""Google Sheet persistence: the parents' Sheet is the app's memory.

The Sheet holds their Holdings, a log of expert-approved Reports, and an append-only audit Log.
Access is via a Google service account (the Sheet is shared with the service email once), so
there is no per-user OAuth. Everything goes through a small SheetGateway: the real one wraps
gspread, an in-memory one drives the tests, and a local-JSON one is the offline fallback when no
service account is configured. Nothing here decides trust or verdicts; it only stores and returns
records, so the persistence layer can be swapped without touching the analysis.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..analysis.sizing import Stance, stance_from_verdict
from ..portfolio.loader import load_holdings
from ..portfolio.models import Holding
from ..research.report import Report, most_recent_by_symbol

HOLDINGS_TAB = "Holdings"
REPORTS_TAB = "Reports"
LOG_TAB = "Log"

LOG_HEADER = ["timestamp", "action", "symbol", "reviewer", "note"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ReportRecord:
    """One row of the Reports tab: the approved (or draft) verdict + stance for a symbol."""
    symbol: str
    company: str
    valuation: str
    quality: str
    leaning: str
    confidence: str
    stance: str
    status: str
    reviewer: str
    updated_at: str
    note: str = ""

    def as_row(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_row(cls, row: dict) -> "ReportRecord":
        return cls(**{f.name: str(row.get(f.name, "")) for f in fields(cls)})


REPORT_HEADER = [f.name for f in fields(ReportRecord)]


def record_from_report(report: Report, symbol: str, stance: str) -> ReportRecord:
    """Build a persistable record from a Report + its stance string. Reviewer/note come from the
    latest audit event so the stored row reflects who acted and why. stance is passed as a string
    to keep this data layer decoupled from the analysis layer."""
    verdict = report.verdict
    last = report.audit[-1] if report.audit else None
    return ReportRecord(
        symbol=symbol.strip().upper(),
        company=report.company,
        valuation=verdict.valuation.value if verdict else "unknown",
        quality=verdict.quality.value if verdict else "unknown",
        leaning=verdict.leaning.value if verdict else "unknown",
        confidence=verdict.confidence.value if verdict else "low",
        stance=stance,
        status=report.status.value,
        reviewer=last.reviewer if last else "",
        updated_at=_now(),
        note=last.note if last else "",
    )


class SheetGateway(ABC):
    """Thin tab-level store. Rows are dicts keyed by header. write overwrites a tab (fine at this
    scale); append adds one row."""

    @abstractmethod
    def read(self, tab: str) -> list[dict]: ...

    @abstractmethod
    def write(self, tab: str, header: list[str], rows: list[dict]) -> None: ...

    @abstractmethod
    def append(self, tab: str, header: list[str], row: dict) -> None: ...


class InMemoryGateway(SheetGateway):
    """For tests. Seed with {tab: [row, ...]}."""

    def __init__(self, data: dict[str, list[dict]] | None = None):
        self._data: dict[str, list[dict]] = {k: [dict(r) for r in v] for k, v in (data or {}).items()}

    def read(self, tab: str) -> list[dict]:
        return [dict(r) for r in self._data.get(tab, [])]

    def write(self, tab: str, header: list[str], rows: list[dict]) -> None:
        self._data[tab] = [{h: r.get(h, "") for h in header} for r in rows]

    def append(self, tab: str, header: list[str], row: dict) -> None:
        self._data.setdefault(tab, []).append({h: row.get(h, "") for h in header})


class LocalJsonGateway(SheetGateway):
    """Offline fallback: persist tabs to a JSON file (under data/, gitignored). Same interface as
    the real Sheet, so local dev and tests behave identically to production."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read(self, tab: str) -> list[dict]:
        return [dict(r) for r in self._load().get(tab, [])]

    def write(self, tab: str, header: list[str], rows: list[dict]) -> None:
        data = self._load()
        data[tab] = [{h: r.get(h, "") for h in header} for r in rows]
        self._save(data)

    def append(self, tab: str, header: list[str], row: dict) -> None:
        data = self._load()
        data.setdefault(tab, []).append({h: row.get(h, "") for h in header})
        self._save(data)


class GspreadGateway(SheetGateway):
    """Real Google Sheet via a service account. gspread is lazy-imported so the module (and the
    test suite) load without it; only this path needs it."""

    def __init__(self, creds_dict: dict, sheet_key: str):
        import gspread
        self._gspread = gspread
        self._sheet = gspread.service_account_from_dict(creds_dict).open_by_key(sheet_key)

    def _worksheet(self, tab: str, header: list[str] | None = None):
        try:
            return self._sheet.worksheet(tab)
        except self._gspread.WorksheetNotFound:
            ws = self._sheet.add_worksheet(title=tab, rows=100, cols=max(len(header or []), 10))
            if header:
                ws.update([header])
            return ws

    def read(self, tab: str) -> list[dict]:
        try:
            return self._worksheet(tab).get_all_records()
        except Exception:
            return []

    def write(self, tab: str, header: list[str], rows: list[dict]) -> None:
        ws = self._worksheet(tab, header)
        grid = [header] + [[str(r.get(h, "")) for h in header] for r in rows]
        # WHY (data-loss resilience): clear()-then-update() would leave the tab PERMANENTLY
        # EMPTY if update() failed partway (network blip, quota, transient Google API error) --
        # clear() had already wiped it with no way to recover. This is the parents' Reports
        # history / daily-picks tab. Write the new data FIRST (a plain overwrite from A1 that
        # leaves the OLD data untouched if it fails), THEN trim any stale trailing rows beyond
        # the new grid's extent via resize -- a much lower-stakes secondary step: if THAT fails,
        # the important data already landed safely; worst case is a few leftover stale rows/cols,
        # never a wiped tab. WHY both rows AND cols (found by adversarial review): a header can
        # shrink too (REPORT_HEADER/TODAY_HEADER are plain code-derived lists that have changed
        # before) -- trimming only rows would leave stale data in trailing columns from a wider
        # previous write.
        ws.update(grid)   # gspread 6.x: update(values, range_name=None) -> writes from A1
        ws.resize(rows=len(grid), cols=len(header))

    def append(self, tab: str, header: list[str], row: dict) -> None:
        ws = self._worksheet(tab, header)
        if not ws.get_all_values():
            ws.update([header])
        ws.append_row([str(row.get(h, "")) for h in header])


class AppsScriptGateway(SheetGateway):
    """Talk to a Google Apps Script web app bound to the owner's Sheet copy. Keyless: no service
    account. The web app is deployed 'execute as me / anyone', and every request carries a shared
    secret token the script checks, so the endpoint is bearer-token private (not public like a
    published CSV). getter/poster are injectable so the mapping is tested offline; the defaults use
    `requests` (installed via gspread) which follows the Apps Script 302 redirect transparently."""

    def __init__(self, url: str, token: str, getter=None, poster=None,
                 attempts: int = 3, backoff: float = 2.0, sleeper=None):
        self._url = url
        self._token = token
        self._getter = getter or self._http_get
        self._poster = poster or self._http_post
        # WHY (regression 2026-07-09): Apps Script web apps cold-start slowly and the daily 09:49
        # run died on one 20s ReadTimeout. This same bridge serves the parents' picks read, so a
        # transient blip must retry with backoff, not be fatal. connect 10s / read 45s covers a
        # cold start that also does Sheet I/O.
        self._attempts = max(1, attempts)
        self._backoff = backoff
        import time
        self._sleep = sleeper or time.sleep

    def _http_get(self, params: dict):
        import requests
        resp = requests.get(self._url, params=params, timeout=(10, 45))  # token already in params
        resp.raise_for_status()
        return resp.json()

    def _http_post(self, payload: dict):
        import requests
        resp = requests.post(self._url, json=payload, timeout=(10, 45))  # token already in payload
        resp.raise_for_status()
        return resp.json()

    def _retry(self, fn, arg):
        # WHY: reads and our writes are idempotent (write clears+overwrites a whole tab; save_report
        # upserts). The only non-idempotent op is the append-only Log, where a rare duplicate audit
        # line on a retried-after-success timeout is harmless and preferable to losing the entry.
        last: Exception | None = None
        for i in range(self._attempts):
            try:
                return fn(arg)
            except Exception as exc:  # transport-level: timeout, connection reset, 5xx, cold start
                last = exc
                if i < self._attempts - 1:
                    self._sleep(self._backoff * (i + 1))
        raise last  # type: ignore[misc]

    def read(self, tab: str) -> list[dict]:
        out = self._retry(self._getter, {"action": "read", "tab": tab, "token": self._token})
        if isinstance(out, list):
            return out
        return out.get("rows", []) if isinstance(out, dict) else []

    def write(self, tab: str, header: list[str], rows: list[dict]) -> None:
        self._retry(self._poster, {"action": "write", "tab": tab, "header": header, "rows": rows,
                                   "token": self._token})

    def append(self, tab: str, header: list[str], row: dict) -> None:
        self._retry(self._poster, {"action": "append", "tab": tab, "header": header, "row": row,
                                   "token": self._token})


def build_gateway(creds_dict: dict | None, sheet_key: str | None,
                  local_fallback_path: str | Path) -> SheetGateway:
    """Real Sheet if a service account + key are configured, else the local-JSON fallback. A
    failed Sheet connection falls back to local rather than crashing the app."""
    if creds_dict and sheet_key:
        try:
            return GspreadGateway(creds_dict, sheet_key)
        except Exception:
            pass
    return LocalJsonGateway(local_fallback_path)


# --- backend operations (gateway-agnostic) ---

def read_holdings(gateway: SheetGateway) -> list[Holding]:
    """Read the Holdings tab into Holding objects, reusing the CSV loader's column matching."""
    records = gateway.read(HOLDINGS_TAB)
    if not records:
        return []
    return load_holdings(pd.DataFrame(records))


def save_report(gateway: SheetGateway, record: ReportRecord) -> None:
    """Upsert a report row by symbol (one current row per name; history lives in the Log)."""
    existing = gateway.read(REPORTS_TAB)
    out: list[dict] = []
    replaced = False
    for row in existing:
        if str(row.get("symbol", "")).strip().upper() == record.symbol:
            out.append(record.as_row())
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(record.as_row())
    gateway.write(REPORTS_TAB, REPORT_HEADER, out)


def read_reports(gateway: SheetGateway) -> list[ReportRecord]:
    return [ReportRecord.from_row(r) for r in gateway.read(REPORTS_TAB)]


def resolve_approved_stances(persisted: list[ReportRecord],
                             session_reports: dict[str, Report]) -> dict[str, Stance]:
    """Merge persisted (Sheet) approvals with this session's fresher research into one
    {symbol: Stance} map for the Invest tab's real-money allocation.

    WHY (real money): a symbol approved in a PAST session is durable via the Sheet, but if the
    user re-researches it THIS session, that fresh report is the current state of the analysis.
    If the fresh report is trusted, its stance supersedes the old one (already correct before this
    fix). If the fresh report is NOT (yet) trusted -- a new DRAFT the user hasn't re-approved --
    the OLD persisted approval must not silently keep feeding suggest_allocation: the current
    analysis is what's now unreviewed, so the approval is dropped, not left stale. A symbol never
    re-researched this session is untouched (nothing contradicts its persisted approval). Uses
    most_recent_by_symbol so a re-run under a different key (e.g. an AR-URL override) is picked by
    actual recency, not dict-iteration order.
    """
    approved: dict[str, Stance] = {}
    for r in persisted:
        if r.status == "approved" and r.stance:
            try:
                approved[r.symbol] = Stance(r.stance)
            except ValueError:
                pass
    for sym_key in {key.split(" ")[0] for key in session_reports}:
        latest = most_recent_by_symbol(session_reports, sym_key)
        if latest is None:
            continue
        if latest.is_trusted and latest.verdict is not None:
            approved[sym_key] = stance_from_verdict(latest.verdict)
        else:
            approved.pop(sym_key, None)
    return approved


def append_log(gateway: SheetGateway, action: str, symbol: str,
               reviewer: str, note: str = "") -> None:
    gateway.append(LOG_TAB, LOG_HEADER,
                   {"timestamp": _now(), "action": action, "symbol": symbol,
                    "reviewer": reviewer, "note": note})
