"""Compute-don't-generate: derive every figure in Python; hand the LLM only the RESULT to phrase.

SPEC v4 §2 decision #1 / §3 (the named failure mode this whole app guards against): an LLM must
never do arithmetic on financial numbers -- FinQA/ConvFinQA show LLM arithmetic fails, and a
confident wrong computed figure is exactly the real-money mistake to avoid. The seam here: take
TYPED NumericRecords (src/research/numeric_records.py), whose magnitudes are already normalized to
absolute units deterministically, compute year-over-year growth / a margin / a CAGR in plain
Python, and return a ComputedFigure the model can only put into words. The model receives a
finished value, never two raw numbers to divide itself.

REUSE (not a second, drifting copy): CAGR delegates to src/analysis/trends.cagr, the one tested
multi-year growth implementation, so this seam can never diverge from the Research tab's trend
math. This module ADDS only the two primitives not already exposed as pure arithmetic anywhere:
year-over-year growth % and a ratio/margin % (deep_metrics bundles those with thresholds and
prose, which is not what the model should be handed to phrase).

Money-math discipline (mirrors figure_sources._num and NumericRecord.__post_init__): a non-finite
input, or a divide-through-(near-)zero denominator, yields None -- withhold, never a fabricated or
infinite number. A ComputedFigure only ever carries a finite value. Dependency-light: stdlib +
the existing analysis math, deterministic, no network, no LLM.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..analysis.trends import cagr
from .numeric_records import PERCENT, RUPEES, NumericRecord

# Divide-through guard: a denominator at or below this magnitude is treated as zero (growth off a
# ~zero base, or a margin on ~zero sales, is undefined -- never divide by it and fabricate a rate).
_EPS = 1e-9


@dataclass(frozen=True)
class ComputedFigure:
    """One value computed deterministically from source figures, for the LLM to PHRASE (never to
    compute). `label` says what it is, `value` is the finite result, `unit` its unit
    ('percent'/'x'), `inputs` the exact absolute source magnitudes used (so the phrasing can be
    checked against them), `formula` the human-readable method. The model is handed this whole
    object and asked only to describe it -- it never sees the raw operands to combine itself."""
    label: str
    value: float
    unit: str
    inputs: tuple[float, ...]
    formula: str


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in xs)


# --- pure primitives (operate on plain floats) -------------------------------------------------

def yoy_growth_pct(previous: float, current: float) -> float | None:
    """Year-over-year growth %, ((current - previous) / |previous|) * 100. None if an input is
    non-finite or `previous` is ~0 (growth off a zero base is undefined -- never divide by it).
    |previous| keeps the sign of the CHANGE correct when a figure recovers from a negative base."""
    if not _finite(previous, current) or abs(previous) <= _EPS:
        return None
    g = (current - previous) / abs(previous) * 100.0
    return g if math.isfinite(g) else None


def ratio_pct(part: float, whole: float) -> float | None:
    """A ratio/margin as a percent, (part / whole) * 100 (e.g. net profit / revenue). None if an
    input is non-finite or `whole` is ~0 (a margin on zero sales is undefined)."""
    if not _finite(part, whole) or abs(whole) <= _EPS:
        return None
    m = part / whole * 100.0
    return m if math.isfinite(m) else None


def cagr_pct(series: dict[int, float]) -> tuple[float, int] | None:
    """Compound annual growth rate (%) and span in years, DELEGATED to analysis.trends.cagr (the
    single tested implementation) so this seam and the Research tab's trend math never diverge.
    Inherits cagr's rules: needs >=3 years and positive endpoints, else None."""
    return cagr(series)


# --- typed-record adapters (the compute-don't-generate seam) -----------------------------------

def _same_unit(a: NumericRecord, b: NumericRecord, *require: str) -> bool:
    """True iff both records carry the SAME unit (and, if `require` is given, that unit is one of
    the required ones). Never compute across incompatible units -- a percent is not a rupee."""
    return a.unit == b.unit and (not require or a.unit in require)


def growth_between(previous: NumericRecord, current: NumericRecord) -> ComputedFigure | None:
    """Year-over-year growth between two records of the SAME unit, on their absolute (scale-
    normalized) magnitudes. None if the units differ (a rupee figure is never compared to a
    percent) or the growth is undefined (see yoy_growth_pct)."""
    if not _same_unit(previous, current):
        return None
    g = yoy_growth_pct(previous.absolute_value, current.absolute_value)
    if g is None:
        return None
    return ComputedFigure(
        label="year-over-year growth", value=g, unit=PERCENT,
        inputs=(previous.absolute_value, current.absolute_value),
        formula="(current - previous) / |previous| * 100")


def margin_between(part: NumericRecord, whole: NumericRecord) -> ComputedFigure | None:
    """A margin (part / whole, as a percent) from two RUPEE records on their absolute magnitudes
    (e.g. net profit / revenue). None unless BOTH are rupee figures (a margin is rupee/rupee) or
    the margin is undefined (see ratio_pct)."""
    if not _same_unit(part, whole, RUPEES):
        return None
    m = ratio_pct(part.absolute_value, whole.absolute_value)
    if m is None:
        return None
    return ComputedFigure(
        label="margin", value=m, unit=PERCENT,
        inputs=(part.absolute_value, whole.absolute_value),
        formula="part / whole * 100")


def _period_year(period: str | None) -> int | None:
    """The 4-digit fiscal year carried on a record's period tag (e.g. 'FY2024' -> 2024), or None."""
    if not period:
        return None
    m = re.search(r"(\d{4})", period)
    return int(m.group(1)) if m else None


def series_from_records(records) -> dict[int, float]:
    """Build a {fiscal_year: absolute_value} series from records of ONE figure across years, so a
    CAGR can be computed. Only records sharing the FIRST record's unit are used (never mix a
    percent series with a rupee one); a record with no parseable FY period is skipped; if two
    records map to the same year the FIRST is kept (a figure is listed once per period)."""
    materialized = list(records)
    if not materialized:
        return {}
    unit = materialized[0].unit
    series: dict[int, float] = {}
    for r in materialized:
        if r.unit != unit:
            continue
        y = _period_year(r.period)
        if y is None or y in series:
            continue
        series[y] = r.absolute_value
    return series


def cagr_from_records(records) -> ComputedFigure | None:
    """CAGR (%) across the fiscal years covered by one figure's records, reusing analysis.trends.cagr
    via cagr_pct. None unless there are >=3 years with positive endpoints (cagr's own rule)."""
    series = series_from_records(records)
    result = cagr_pct(series)
    if result is None:
        return None
    rate, span = result
    return ComputedFigure(
        label=f"compound annual growth rate over {span} years", value=rate, unit=PERCENT,
        inputs=tuple(series[y] for y in sorted(series)),
        formula="(last / first) ** (1 / years) - 1, as a percent")
