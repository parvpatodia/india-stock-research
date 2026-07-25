"""Typed numeric records with provenance.

Given a piece of document text, extract each figure as a structured record: the value as
written, the exact raw string, its unit (rupees / percent / ratio / bps), its Indian scale
(crore / lakh / million / none), currency, fiscal period, company, source document, and a
locator back to the chunk it came from. The scale is normalized to an ABSOLUTE value
deterministically (crore -> 1e7, lakh -> 1e5, million -> 1e6). This is the seam the
compute-don't-generate layer (W3) builds on: a stated number can be resolved to its exact
record and its absolute magnitude computed, instead of being re-typed by the model.

This COMPLEMENTS src/data/figure_sources.py (which cross-verifies provider figures). It does
NOT cross-verify; it turns raw citable numbers in document text into typed records with
provenance. Bad inputs are rejected hard (non-finite / overflow-on-normalization are refused
at construction, mirroring the money-math discipline in figure_sources._num); garbage in the
text is skipped rather than crashing extraction. Dependency-light: stdlib only, deterministic,
no network.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Unit labels.
RUPEES = "rupees"
PERCENT = "percent"
RATIO = "ratio"
BPS = "bps"

# Scale -> absolute multiplier. "none" means the value is already in its final unit (a percent,
# a ratio, basis points, or absolute rupees), so normalization is the identity.
SCALE_FACTORS: dict[str, float] = {
    "crore": 1e7,
    "lakh": 1e5,
    "million": 1e6,
    "none": 1.0,
}

# Word -> canonical scale. Covers the spellings Indian filings/press actually use.
_SCALE_OF: dict[str, str] = {
    "crore": "crore", "crores": "crore", "cr": "crore",
    "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
    "million": "million", "mn": "million",
}


@dataclass(frozen=True)
class NumericRecord:
    """One figure extracted from document text, with provenance and a deterministic
    normalization to absolute units. `value` is the magnitude AS WRITTEN (sign applied);
    `absolute_value` applies the scale factor. Percent/ratio/bps carry scale 'none' and pass
    through unchanged (a percent is never multiplied by a rupee scale)."""
    value: float
    raw_string: str
    unit: str
    scale: str
    currency: str | None
    period: str | None
    company: str | None
    source_doc: str | None
    locator: str | None

    def __post_init__(self) -> None:
        # WHY (real money, mirrors figure_sources._num): a figure that is NaN / +-inf must never
        # become a record -- it would later render or compute as a fabricated number. Reject at
        # construction so no downstream layer has to re-check. A string/None value also fails here.
        try:
            finite = math.isfinite(self.value)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"NumericRecord.value must be a finite number, got {self.value!r}")
        if self.scale not in SCALE_FACTORS:
            raise ValueError(f"unknown scale {self.scale!r}; expected one of {list(SCALE_FACTORS)}")
        # A finite value can still overflow to inf once scaled (e.g. a garbage 1e308 crore);
        # reject the normalized non-finite too so the seam never yields an infinite magnitude.
        if not math.isfinite(self.value * SCALE_FACTORS[self.scale]):
            raise ValueError("normalized absolute value is non-finite (overflow)")

    @property
    def absolute_value(self) -> float:
        """Deterministic normalization to absolute units (crore->1e7, lakh->1e5, million->1e6)."""
        return self.value * SCALE_FACTORS[self.scale]


def number_key(s: str) -> str:
    """Canonical match key for a figure: drop thousands-separator commas and any decoration
    (currency mark, scale word, parentheses, sign) but KEEP the decimal point. Mirrors
    grounded_analyst._num_key so a record resolves the same way numbers_grounded checks: '73,670'
    and '73670' collide, but 12.34 stays distinct from 1234 (a 100x mismatch must not match)."""
    return re.sub(r"[^\d.]", "", s or "")


def find_record(raw_number: str, records) -> NumericRecord | None:
    """Resolve a stated number to the exact record it came from, or None if no record carries it.
    This is the "no record -> no numeric claim" primitive: a number absent from the retrieved
    records cannot be cited. Matches on number_key (commas dropped, decimal kept)."""
    key = number_key(str(raw_number))
    if not key or key == ".":
        return None
    # Match on the as-written form (raw_string always contains the figure's digits). A plain
    # numeric key (commas dropped, decimal kept) avoids the scientific-notation and trailing-zero
    # ambiguity a value-format round-trip would introduce for large rupee magnitudes.
    for r in records:
        if number_key(r.raw_string) == key:
            return r
    return None


# --- extraction ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Metadata that carries digits but is NOT a citable figure: ISO dates, FY tags, and fiscal-year
# ranges ("2024-25"). Blanked (to equal-length spaces, preserving offsets) before extraction so a
# year can never be mined as a figure nor a range's second half be read as a negative.
_DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z?)?")
_FY_TAG = re.compile(r"\bFY\s?\d{2,4}(?:\s?[-/]\s?\d{2,4})?\b", re.IGNORECASE)
_YEAR_RANGE = re.compile(r"\b(?:19|20)\d{2}\s?[-/]\s?\d{2,4}\b")

_CURRENCY_AT_END = re.compile(r"(?:₹|inr|rs\.?)\s*$", re.IGNORECASE)
_PERCENT_AT_START = re.compile(r"\s*(%|per\s*cent|percent)", re.IGNORECASE)
_BPS_AT_START = re.compile(r"\s*(bps|basis\s*points?)\b", re.IGNORECASE)
# A scale word may sit just past a closing parenthesis: "(1,234) crore".
_SCALE_AT_START = re.compile(r"\s*\)?\s*(crores?|lakhs?|lacs?|million|mn|cr)\b", re.IGNORECASE)


def _blank_metadata(text: str) -> str:
    """Replace date/FY/year-range spans with equal-length spaces so extraction offsets stay
    aligned with the original text while those digits are removed from consideration."""
    def blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    out = _DATE_LIKE.sub(blank, text)
    out = _FY_TAG.sub(blank, out)
    out = _YEAR_RANGE.sub(blank, out)
    return out


def _classify_suffix(right: str) -> tuple[str | None, str | None, int]:
    """Given the text immediately after a number, return (unit, scale, consumed_len) for the
    inline unit, or (None, None, 0) if there is none. Order: percent, bps, rupee scale, ratio."""
    m = _PERCENT_AT_START.match(right)
    if m:
        return PERCENT, "none", m.end()
    m = _BPS_AT_START.match(right)
    if m:
        return BPS, "none", m.end()
    m = _SCALE_AT_START.match(right)
    if m:
        return RUPEES, _SCALE_OF[m.group(1).lower()], m.end()
    # A ratio 'x' must be appended directly to the number (e.g. "18.2x"), never after a space,
    # so an incidental "5 x 3" is not read as a P/E-style ratio.
    if right[:1].lower() == "x" and (len(right) < 2 or not right[1].isalpha()):
        return RATIO, "none", 1
    return None, None, 0


def _left_flags(left: str) -> tuple[str | None, bool]:
    """Return (currency, is_negative) from the text immediately before a number. Negative is the
    accounting parenthesis convention "(1,234)" or a directly-attached minus "-100"; a spaced
    hyphen (a separator/range) is deliberately NOT treated as a sign."""
    currency = "INR" if _CURRENCY_AT_END.search(left) else None
    stripped = left.rstrip()
    negative = stripped.endswith("(") or left.endswith("-")
    return currency, negative


def _is_figure_like(numstr: str) -> bool:
    """A unit-less number under a caption scale is only recorded if it reads like a magnitude:
    grouped (a comma), fractional (a decimal), or >=5 digits. This skips bare years/small counts
    (e.g. '2024', '31') that would otherwise be mislabeled as figures in a scaled table."""
    digits = re.sub(r"\D", "", numstr)
    return ("," in numstr) or ("." in numstr) or len(digits) >= 5


def extract_records(text: str, *, default_scale: str | None = None,
                    currency: str | None = None, period: str | None = None,
                    company: str | None = None, source_doc: str | None = None,
                    locator: str | None = None) -> list[NumericRecord]:
    """Extract typed numeric records from `text`.

    A number is recorded when it carries an inline unit (a rupee scale word / 'cr', a percent, a
    ratio 'x', or basis points) OR a currency prefix (absolute rupees). When `default_scale` is
    a money scale (from a table's caption/units line: 'crore'/'lakh'/'million', or 'none' for a
    plain-rupee table), figure-like bare cells inherit it -- this is how an intact scaled table's
    columns become typed records. Provenance (period/company/source_doc/locator) is attached to
    every record. Garbage is skipped; a value that overflows on normalization is skipped (its
    NumericRecord construction refuses it) rather than crashing extraction."""
    if not text or not text.strip():
        return []
    work = _blank_metadata(text)
    records: list[NumericRecord] = []
    for m in _NUMBER_RE.finditer(work):
        num_start, num_end = m.start(), m.end()
        numstr = text[num_start:num_end]
        left, right = work[:num_start], work[num_end:]
        cur, negative = _left_flags(left)
        unit, scale, consumed = _classify_suffix(right)

        if unit is None:
            if cur is not None:
                # currency prefix, no scale word -> absolute rupees
                unit, scale = RUPEES, "none"
            elif default_scale is not None and _is_figure_like(numstr):
                # a bare cell in a scaled table inherits the table's declared scale
                unit, scale = RUPEES, default_scale
            else:
                continue  # a bare number with no unit and no caption scale is not a figure

        # currency: an explicit mark wins; else a rupee figure inherits the chunk-level currency
        # (a table's caption currency). Percent/ratio/bps never take a currency.
        if unit == RUPEES and cur is None:
            cur = currency
        rec_currency = cur if unit == RUPEES else None
        magnitude = _to_float(numstr, negative)
        if magnitude is None:
            continue
        raw_string = text[num_start:num_end + consumed].strip()
        try:
            records.append(NumericRecord(
                value=magnitude, raw_string=raw_string, unit=unit, scale=scale,
                currency=rec_currency, period=period, company=company,
                source_doc=source_doc, locator=locator))
        except ValueError:
            # non-finite / overflow-on-normalization: degrade by skipping this one figure
            continue
    return records


def _to_float(numstr: str, negative: bool) -> float | None:
    try:
        v = float(numstr.replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return -v if negative else v
