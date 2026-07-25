from datetime import date, datetime, timezone

import pytest

from src.freshness.event_log import (
    IngestionEvent,
    IngestionLog,
    content_hash,
)


def _fixed_clock(*instants):
    """A clock returning the given instants in order, then repeating the last one."""
    seq = list(instants)

    def clock():
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return clock


_T0 = datetime(2026, 7, 9, 10, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 7, 9, 11, 0, 0, tzinfo=timezone.utc)


def test_content_hash_is_stable_and_whitespace_insensitive():
    a = content_hash("Reliance profit was 100 crore")
    b = content_hash("Reliance   profit was 100 crore\n")   # extra whitespace only
    c = content_hash("Reliance profit was 200 crore")       # different content
    assert a == b            # deterministic + whitespace-normalized
    assert a != c
    assert len(a) == 64      # sha256 hex


def test_ingest_rejects_empty_content_and_ids(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0))
    with pytest.raises(ValueError):
        log.ingest(item_key="k", source_id="news_google", content="   ", as_of="2026-07-08")
    with pytest.raises(ValueError):
        log.ingest(item_key="", source_id="news_google", content="hi", as_of="2026-07-08")
    with pytest.raises(ValueError):
        log.ingest(item_key="k", source_id="", content="hi", as_of="2026-07-08")


def test_ingest_rejects_unparseable_as_of(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0))
    with pytest.raises(ValueError):
        log.ingest(item_key="k", source_id="news_google", content="hi", as_of="last tuesday")


def test_first_ingest_creates_a_current_event(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0))
    res = log.ingest(item_key="story-1", source_id="news_google",
                     content="Reliance profit rose", as_of="2026-07-08")
    assert res.action == "new"
    assert res.wrote is True
    ev = res.event
    assert isinstance(ev, IngestionEvent)
    assert ev.item_key == "story-1"
    assert ev.source_id == "news_google"
    assert ev.as_of == "2026-07-08"
    assert ev.observed_at == "2026-07-09T10:00:00Z"
    assert ev.superseded is False
    assert len(log.current()) == 1
    assert len(log.events()) == 1


def test_identical_content_reingest_is_a_noop(tmp_path):
    path = tmp_path / "events.jsonl"
    log = IngestionLog(path, clock=_fixed_clock(_T0, _T1))
    log.ingest(item_key="story-1", source_id="news_google", content="same text", as_of="2026-07-08")
    res = log.ingest(item_key="story-1", source_id="news_google", content="same  text",
                     as_of="2026-07-08")   # identical after whitespace-normalization
    assert res.action == "skipped"
    assert res.wrote is False
    assert len(log.events()) == 1          # nothing appended
    # the file itself must have exactly one line
    assert path.read_text().strip().count("\n") == 0


def test_changed_content_supersedes_prior_version(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0, _T1))
    log.ingest(item_key="story-1", source_id="news_google", content="v1 text", as_of="2026-07-08")
    res = log.ingest(item_key="story-1", source_id="news_google", content="v2 text UPDATED",
                     as_of="2026-07-09")
    assert res.action == "superseded"
    assert res.wrote is True
    events = log.events()
    assert len(events) == 2
    prior = [e for e in events if e.observed_at == "2026-07-09T10:00:00Z"][0]
    latest = [e for e in events if e.observed_at == "2026-07-09T11:00:00Z"][0]
    assert prior.superseded is True        # old version flagged superseded
    assert latest.superseded is False      # new version is current
    current = log.current()
    assert len(current) == 1
    assert current[0].content_hash == content_hash("v2 text UPDATED")


def test_different_item_keys_coexist(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0))
    log.ingest(item_key="story-1", source_id="news_google", content="a", as_of="2026-07-08")
    log.ingest(item_key="story-2", source_id="news_google", content="b", as_of="2026-07-08")
    assert len(log.current()) == 2


def test_log_persists_and_reloads_from_disk(tmp_path):
    path = tmp_path / "events.jsonl"
    log = IngestionLog(path, clock=_fixed_clock(_T0, _T1))
    log.ingest(item_key="story-1", source_id="news_google", content="v1", as_of="2026-07-08")
    log.ingest(item_key="story-1", source_id="news_google", content="v2", as_of="2026-07-09")

    reloaded = IngestionLog(path)                     # fresh instance reads the same file
    assert len(reloaded.events()) == 2
    assert len(reloaded.current()) == 1
    assert reloaded.current()[0].content_hash == content_hash("v2")
    # a further no-op still holds across the reload boundary
    res = reloaded.ingest(item_key="story-1", source_id="news_google", content="v2",
                          as_of="2026-07-09")
    assert res.action == "skipped"


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    # WHY (resilience, real money): append-only writes are not atomic; a crash mid-write can leave
    # a partial line. Loading must skip it and keep the valid events, like EvalStore/AMFI parsing.
    path = tmp_path / "events.jsonl"
    log = IngestionLog(path, clock=_fixed_clock(_T0))
    log.ingest(item_key="story-1", source_id="news_google", content="good", as_of="2026-07-08")
    with path.open("a", encoding="utf-8") as f:
        f.write("{ this is not valid json\n")          # corrupt trailing line
    reloaded = IngestionLog(path)
    assert len(reloaded.events()) == 1                  # bad line skipped, good one kept


def test_freshness_for_uses_the_as_of_date(tmp_path):
    log = IngestionLog(tmp_path / "events.jsonl", clock=_fixed_clock(_T0))
    log.ingest(item_key="story-1", source_id="news_google", content="old news",
               as_of="2023-01-01")
    f = log.freshness_for("story-1", today=date(2026, 7, 9), threshold_days=30)
    assert f is not None
    assert f.stale is True
    assert log.freshness_for("missing-key", today=date(2026, 7, 9), threshold_days=30) is None
