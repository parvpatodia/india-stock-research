"""As-of dating and staleness flags.

Every ingested record carries the date its content is effective (the "as-of" date). Freshness
is a pure function of that date, a reference "today", and a configurable threshold: a record
older than the threshold is STALE. An undated record is UNKNOWN, never silently "fresh" -- for a
real-money research tool, "we don't know how old this is" must read differently from "this is
current". The date parsing lives here as the single source of truth, reused by the event log.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


def parse_iso_date(value) -> date | None:
    """Parse a date, datetime, or ISO string ('YYYY-MM-DD' or a full timestamp) to a date.

    Returns None for empty or unparseable input rather than raising, so callers can treat
    "undated" as a first-class state. The single date parser used across the freshness package.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        # Accept a full ISO timestamp too (news feeds carry 'YYYY-MM-DDTHH:MM:SSZ').
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class Freshness:
    """The freshness verdict for one as-of date against a reference day + threshold."""
    as_of: str            # ISO date the content is effective, or "" if unknown
    today: str            # ISO reference date the verdict was computed against
    threshold_days: int
    age_days: int | None  # today - as_of in days; None if the date is unknown
    stale: bool           # provably older than the threshold
    known: bool           # False if the as-of date is missing/unparseable

    @property
    def label(self) -> str:
        if not self.known:
            return "date unknown"
        if self.stale:
            return f"stale, {self.age_days} days old"
        if self.age_days is not None and self.age_days < 0:
            return "fresh (future-dated)"
        return f"fresh, {self.age_days} days old"


def freshness(as_of, today, threshold_days: int) -> Freshness:
    """Compute the freshness verdict for an as-of date.

    `today` accepts a date or an ISO string. `threshold_days` must be a positive integer: a
    non-positive window is a caller bug (it would flag everything or nothing), rejected hard
    rather than silently coerced -- the loader/AMFI money-math lesson applied to freshness config.
    """
    if not isinstance(threshold_days, int) or isinstance(threshold_days, bool) or threshold_days <= 0:
        raise ValueError(f"threshold_days must be a positive int, got {threshold_days!r}")
    ref = parse_iso_date(today)
    if ref is None:
        raise ValueError(f"unparseable 'today' reference date: {today!r}")
    as_of_date = parse_iso_date(as_of)
    if as_of_date is None:
        return Freshness(as_of="", today=ref.isoformat(), threshold_days=threshold_days,
                         age_days=None, stale=False, known=False)
    age = (ref - as_of_date).days
    return Freshness(as_of=as_of_date.isoformat(), today=ref.isoformat(),
                     threshold_days=threshold_days, age_days=age,
                     stale=age > threshold_days, known=True)


def describe_freshness(as_of, today, threshold_days: int, subject: str = "latest update") -> str:
    """A human-readable freshness line, e.g. 'latest filing: dated 2024-06-30 (fresh, 20 days old)'
    or 'latest filing: date unknown'. What the app surfaces next to a figure or a news item."""
    f = freshness(as_of, today, threshold_days)
    if not f.known:
        return f"{subject}: date unknown"
    return f"{subject}: dated {f.as_of} ({f.label})"
