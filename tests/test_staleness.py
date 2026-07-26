from datetime import date

import pytest

from src.freshness.staleness import (
    DEFAULT_RECENCY_WINDOW_DAYS,
    Freshness,
    describe_freshness,
    freshness,
    is_recent,
    parse_iso_date,
)

_TODAY = date(2026, 7, 9)


def test_parse_iso_date_accepts_date_str_and_timestamp():
    assert parse_iso_date("2024-06-30") == date(2024, 6, 30)
    assert parse_iso_date("2024-06-30T05:09:16Z") == date(2024, 6, 30)  # full ISO timestamp
    assert parse_iso_date(date(2024, 6, 30)) == date(2024, 6, 30)
    assert parse_iso_date("") is None
    assert parse_iso_date(None) is None
    assert parse_iso_date("not a date") is None


def test_fresh_item_within_threshold_is_not_stale():
    f = freshness("2026-06-30", _TODAY, threshold_days=30)   # 9 days old
    assert isinstance(f, Freshness)
    assert f.known is True
    assert f.stale is False
    assert f.age_days == 9


def test_stale_item_beyond_threshold_is_flagged():
    f = freshness("2023-01-01", _TODAY, threshold_days=30)   # ~3.5 years old
    assert f.known is True
    assert f.stale is True
    assert f.age_days > 1000


def test_boundary_exactly_at_threshold_is_not_stale():
    # WHY: age == threshold is still within the window; only strictly older is stale.
    f = freshness("2026-06-09", _TODAY, threshold_days=30)   # exactly 30 days old
    assert f.age_days == 30
    assert f.stale is False
    f2 = freshness("2026-06-08", _TODAY, threshold_days=30)  # 31 days old -> stale
    assert f2.age_days == 31
    assert f2.stale is True


def test_undated_item_is_unknown_not_silently_fresh():
    # WHY (real money, honesty): an undated record must never read as "fresh". It is unknown.
    f = freshness("", _TODAY, threshold_days=30)
    assert f.known is False
    assert f.age_days is None
    assert f.stale is False           # not provably stale, but 'known' is False so the UI cautions


def test_future_dated_item_is_not_stale():
    f = freshness("2026-08-01", _TODAY, threshold_days=30)   # dated in the future
    assert f.stale is False
    assert f.age_days is not None and f.age_days < 0


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        freshness("2026-06-30", _TODAY, threshold_days=0)
    with pytest.raises(ValueError):
        freshness("2026-06-30", _TODAY, threshold_days=-5)


# --- recency window (H1: bound the ingestion log to recent filings) ---------------------------

def test_is_recent_boundary_exactly_window_days_is_included():
    # WHY: a filing exactly `window_days` old is still RECENT (included at ingest); only strictly
    # older is excluded. Mirrors the freshness threshold's inclusive boundary.
    assert is_recent("2026-05-10", _TODAY, window_days=60) is True   # exactly 60 days old
    assert is_recent("2026-05-09", _TODAY, window_days=60) is False  # 61 days old -> excluded


def test_is_recent_recent_item_is_true_old_item_is_false():
    assert is_recent("2026-07-01", _TODAY, window_days=120) is True   # 8 days old
    assert is_recent("2024-07-19", _TODAY, window_days=120) is False  # ~2 years old


def test_is_recent_undated_returns_none_never_dropped():
    # WHY (real money, honesty): an undated filing must not vanish through the window. None signals
    # the caller to KEEP-and-count it separately, never a silent drop.
    assert is_recent("", _TODAY, window_days=120) is None
    assert is_recent(None, _TODAY, window_days=120) is None
    assert is_recent("not a date", _TODAY, window_days=120) is None


def test_is_recent_future_dated_is_recent():
    assert is_recent("2026-08-01", _TODAY, window_days=120) is True   # future-dated -> recent


def test_is_recent_window_days_must_be_positive_int():
    for bad in (0, -5, True):
        with pytest.raises(ValueError):
            is_recent("2026-07-01", _TODAY, window_days=bad)


def test_is_recent_rejects_unparseable_today():
    with pytest.raises(ValueError):
        is_recent("2026-07-01", "not a date", window_days=120)


def test_default_recency_window_is_120_days():
    assert DEFAULT_RECENCY_WINDOW_DAYS == 120


def test_describe_freshness_reads_for_a_human():
    fresh = describe_freshness("2024-06-30", date(2024, 7, 10), threshold_days=90,
                               subject="latest filing")
    assert "latest filing" in fresh
    assert "2024-06-30" in fresh
    assert "fresh" in fresh.lower()

    stale = describe_freshness("2020-01-01", _TODAY, threshold_days=90, subject="latest filing")
    assert "stale" in stale.lower()
    assert "2020-01-01" in stale

    unknown = describe_freshness("", _TODAY, threshold_days=90, subject="latest filing")
    assert "unknown" in unknown.lower()
