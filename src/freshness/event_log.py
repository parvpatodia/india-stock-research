"""Append-only ingestion event log (JSONL), event-sourced.

Every ingested item is recorded as an immutable event: a stable content hash, the source id, the
date the content is effective (as-of) and the timestamp it was observed. The log is the source of
truth for "what do we know, from where, and how fresh".

Two derived facts fall out of the event stream:
  * DEDUP -- re-ingesting byte-identical content (same hash) for the same logical item is a no-op.
  * SUPERSESSION -- new content for an existing item appends a new event; the prior one is
    superseded. Nothing is ever mutated in place (append-only); "current vs superseded" is a
    projection over the stream (per item key, the last appended event is current).

Path and clock are injectable so tests write to a temp dir with a fixed clock. Bad inputs (empty
content or ids, an unparseable as-of date) are rejected HARD at ingest -- the money-math lesson:
validate at the boundary, don't let a junk record poison the store. A corrupt line on load is
skipped, not fatal (append-only writes are not atomic).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .staleness import Freshness, freshness, parse_iso_date

_WS = re.compile(r"\s+")


def content_hash(content: str) -> str:
    """Stable SHA-256 of whitespace-normalized content. Whitespace-insensitive so a trivially
    reflowed copy of the same text does not read as a new version; case/character preserving so a
    genuine edit does. 64-hex-char digest."""
    normalized = _WS.sub(" ", content or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestionEvent:
    item_key: str          # logical identity of the item (stable across versions)
    source_id: str         # registered/tiered source this came from
    content_hash: str
    as_of: str             # ISO date the content is effective, or "" if undated
    observed_at: str       # ISO-8601 UTC timestamp the item was ingested
    kind: str = "news"
    ref: str = ""          # locator / URL
    title: str = ""
    cluster_id: str = ""   # near-duplicate cluster this item belongs to (empty if not clustered)
    superseded: bool = False   # DERIVED on read: a newer event exists for this item_key

    # Fields persisted to JSONL. `superseded` is intentionally NOT persisted: it is a projection
    # over the whole stream, recomputed on load, so an old line is never rewritten.
    _PERSIST_FIELDS = ("item_key", "source_id", "content_hash", "as_of", "observed_at",
                       "kind", "ref", "title", "cluster_id")

    def to_record(self) -> dict:
        return {f: getattr(self, f) for f in self._PERSIST_FIELDS}

    @classmethod
    def from_record(cls, data: dict) -> "IngestionEvent":
        # Only known fields; a schema drift (extra/missing key) is tolerated by from_record's
        # caller (load skips a line that raises).
        return cls(
            item_key=str(data["item_key"]),
            source_id=str(data["source_id"]),
            content_hash=str(data["content_hash"]),
            as_of=str(data.get("as_of", "")),
            observed_at=str(data["observed_at"]),
            kind=str(data.get("kind", "news")),
            ref=str(data.get("ref", "")),
            title=str(data.get("title", "")),
            cluster_id=str(data.get("cluster_id", "")),
        )


@dataclass(frozen=True)
class IngestResult:
    action: str            # "new" | "superseded" | "skipped"
    event: IngestionEvent  # the current event for the item after this call
    wrote: bool            # whether a line was appended


class IngestionLog:
    """Append-only JSONL event log with content-hash dedup and version supersession."""

    def __init__(self, path: str | Path,
                 clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: list[IngestionEvent] = []            # raw events, file order
        self._current_idx_by_key: dict[str, int] = {}      # item_key -> index of current event
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # WHY skip, don't crash: append-only writes are not atomic, so a crash mid-write (or a
            # manual edit) can leave a partial/corrupt line. Same resilience as EvalStore/AMFI.
            try:
                event = IngestionEvent.from_record(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
            self._current_idx_by_key[event.item_key] = len(self._events)
            self._events.append(event)

    def _observed_now(self) -> str:
        dt = self._clock()
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def ingest(self, *, item_key: str, source_id: str, content: str, as_of: str = "",
               kind: str = "news", ref: str = "", title: str = "",
               cluster_id: str = "") -> IngestResult:
        """Record an ingested item. Returns an IngestResult saying whether it was new, a new
        version of an existing item (prior superseded), or a dedup no-op (identical content)."""
        item_key = (item_key or "").strip()
        source_id = (source_id or "").strip()
        if not item_key:
            raise ValueError("item_key must be a non-empty string")
        if not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        as_of_norm = ""
        if as_of:
            parsed = parse_iso_date(as_of)
            if parsed is None:
                raise ValueError(f"unparseable as_of date: {as_of!r}")
            as_of_norm = parsed.isoformat()

        digest = content_hash(content)
        cur_idx = self._current_idx_by_key.get(item_key)
        if cur_idx is not None and self._events[cur_idx].content_hash == digest:
            # Identical content for the same item -> dedup no-op, nothing appended.
            return IngestResult(action="skipped", event=self._events[cur_idx], wrote=False)

        event = IngestionEvent(
            item_key=item_key, source_id=source_id, content_hash=digest, as_of=as_of_norm,
            observed_at=self._observed_now(), kind=kind, ref=ref, title=title,
            cluster_id=cluster_id,
        )
        self._append(event)
        action = "superseded" if cur_idx is not None else "new"
        return IngestResult(action=action, event=event, wrote=True)

    def _append(self, event: IngestionEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_record()) + "\n")
        self._current_idx_by_key[event.item_key] = len(self._events)
        self._events.append(event)

    def _is_current(self, index: int) -> bool:
        event = self._events[index]
        return self._current_idx_by_key.get(event.item_key) == index

    def events(self) -> list[IngestionEvent]:
        """All events in ingest order, each with `superseded` set from the current projection."""
        return [replace(e, superseded=not self._is_current(i))
                for i, e in enumerate(self._events)]

    def current(self) -> list[IngestionEvent]:
        """Only the current (non-superseded) event per item key, in ingest order."""
        return [replace(e, superseded=False)
                for i, e in enumerate(self._events) if self._is_current(i)]

    def latest_for(self, item_key: str) -> IngestionEvent | None:
        idx = self._current_idx_by_key.get((item_key or "").strip())
        return replace(self._events[idx], superseded=False) if idx is not None else None

    def freshness_for(self, item_key: str, today, threshold_days: int) -> Freshness | None:
        """Freshness verdict for an item's current version, or None if the item is unknown. The
        helper the app calls to show 'latest filing: dated ... (stale/fresh)'."""
        event = self.latest_for(item_key)
        if event is None:
            return None
        return freshness(event.as_of, today, threshold_days)
