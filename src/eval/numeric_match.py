"""Unit-normalized numeric-exact-match metric (SPEC v4 §4 W5, the India golden set).

Over a small golden set of Indian annual-report/table fixtures with KNOWN ground-truth figures,
assert every extracted NumericRecord's scale-normalized absolute value matches ground truth
EXACTLY (crore->1e7, lakh->1e5, million->1e6). The point is unit discipline: a figure the source
states in CRORE, if extracted or matched as MILLION, is a 10x error and MUST fail the metric --
the crore/lakh/million trap this app exists to prevent, made a measurable gate.

Deterministic, offline: fixtures are inline text ingested through the real DocumentStore +
numeric_records extractor (no network, no LLM). Period-keyed: a golden figure is only satisfied by
a record carrying BOTH the same digits AND (when specified) the same fiscal period, so an FY23
value presented under FY24 does not match -- the period-mixing guard at the record layer.

This EXTENDS src/eval; it reuses the DocumentStore ingestion + numeric_records seam (W2/W3) rather
than re-implementing extraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..research.grounding import DocumentStore, numeric_records
from ..research.numeric_records import number_key
from ..sources.registry import CredibilityTier, Source, SourceRegistry

# Exact means exact: these ground-truth magnitudes are integer * a power-of-ten scale factor, all
# well within 2^53, so equality is representable in float. A tiny tolerance only absorbs the
# floating multiply, never a real scale error (crore vs million differ 10x, far outside this).
_REL_TOL = 1e-9


def _exact(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


@dataclass(frozen=True)
class GoldenFigure:
    """One known figure in a fixture: the digit form as written, its ground-truth ABSOLUTE
    magnitude (already unit-normalized), and optionally the fiscal period it belongs to."""
    label: str
    raw_number: str
    expected_absolute: float
    period: str | None = None


@dataclass(frozen=True)
class GoldenFixture:
    """A fixture document with known figures. `structured=True` ingests element-aware (tables kept
    intact); `fiscal_period` tags the whole document (used for single-period fixtures)."""
    name: str
    text: str
    company: str
    figures: tuple[GoldenFigure, ...]
    structured: bool = False
    fiscal_period: str | None = None
    source_id: str = "annual_report"


@dataclass(frozen=True)
class FigureMatch:
    fixture: str
    figure: GoldenFigure
    matched: bool
    found_absolute: float | None
    detail: str


@dataclass(frozen=True)
class NumericMatchResult:
    matches: tuple[FigureMatch, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.matches)

    @property
    def matched(self) -> int:
        return sum(1 for m in self.matches if m.matched)

    @property
    def mismatches(self) -> list[FigureMatch]:
        return [m for m in self.matches if not m.matched]

    @property
    def exact_match_rate(self) -> float:
        """matched / total; 1.0 vacuously for an empty set (nothing failed)."""
        return self.matched / self.total if self.total else 1.0


def _store_for(fixture: GoldenFixture) -> DocumentStore:
    registry = SourceRegistry([Source(fixture.source_id, fixture.source_id,
                                       CredibilityTier.PRIMARY)])
    store = DocumentStore(registry=registry)
    store.add_document(fixture.source_id, fixture.text, structured=fixture.structured,
                       company=fixture.company, doc_type="annual_report",
                       fiscal_period=fixture.fiscal_period)
    return store


def _records_of(fixture: GoldenFixture):
    store = _store_for(fixture)
    # min_score=0.0 with a large k returns every chunk, so the metric sees all extracted records
    # regardless of TF-IDF overlap (the fixture is small and self-contained).
    hits = store.retrieve(fixture.text, k=100, min_score=0.0)
    return numeric_records(hits)


def _resolve(figure: GoldenFigure, records) -> FigureMatch:
    key = number_key(figure.raw_number)
    candidates = [r for r in records
                  if number_key(r.raw_string) == key
                  and (figure.period is None or r.period == figure.period)]
    if not candidates:
        return FigureMatch("", figure, False, None,
                           f"no record for {figure.raw_number!r}"
                           + (f" in {figure.period}" if figure.period else ""))
    # A record satisfies the figure only if its normalized absolute matches EXACTLY. Report the
    # first candidate's magnitude as `found` so a scale error (crore read as million) is visible.
    for r in candidates:
        if _exact(r.absolute_value, figure.expected_absolute):
            return FigureMatch("", figure, True, r.absolute_value, "exact match")
    found = candidates[0].absolute_value
    return FigureMatch("", figure, False, found,
                       f"expected {figure.expected_absolute:,.0f} but record is {found:,.0f} "
                       f"(scale mismatch)")


def numeric_exact_match(fixtures) -> NumericMatchResult:
    """Ingest each fixture, then check every golden figure resolves to a typed record whose
    unit-normalized absolute value EXACTLY equals ground truth. A scale error (crore vs million) or
    a period swap surfaces as a mismatch."""
    matches: list[FigureMatch] = []
    for fixture in fixtures:
        records = _records_of(fixture)
        for figure in fixture.figures:
            m = _resolve(figure, records)
            matches.append(FigureMatch(fixture.name, figure, m.matched, m.found_absolute,
                                       m.detail))
    return NumericMatchResult(tuple(matches))


# --- the India golden set (small, realistic annual-report/table figures) ----------------------

_PL_TABLE = """Statement of Profit and Loss
(Rs in crore)
Particulars                 FY2024     FY2023
Revenue from operations    9,00,000   7,92,000
Net profit for the year      73,670     66,700
"""

GOLDEN_FIXTURES: tuple[GoldenFixture, ...] = (
    GoldenFixture(
        name="pl_table_crore",
        text=_PL_TABLE,
        company="RELIANCE",
        structured=True,
        figures=(
            GoldenFigure("revenue_fy24", "9,00,000", 900000.0 * 1e7),
            GoldenFigure("revenue_fy23", "7,92,000", 792000.0 * 1e7),
            GoldenFigure("net_profit_fy24", "73,670", 73670.0 * 1e7),
            GoldenFigure("net_profit_fy23", "66,700", 66700.0 * 1e7),
        ),
    ),
    GoldenFixture(
        name="inline_crore_and_percent",
        text="Net profit was Rs 500 crore and ROE improved to 22.5% for the year.",
        company="ACME",
        figures=(
            GoldenFigure("net_profit", "500", 500.0 * 1e7),
        ),
    ),
    GoldenFixture(
        name="inline_lakh",
        text="Other income was Rs 4,50,000 lakh for the year.",
        company="ACME",
        figures=(
            GoldenFigure("other_income", "4,50,000", 450000.0 * 1e5),
        ),
    ),
    GoldenFixture(
        name="period_keyed_fy2023",
        text="Net profit for the year was Rs 620 crore.",
        company="ACME",
        fiscal_period="FY2023",
        figures=(
            GoldenFigure("net_profit_fy23", "620", 620.0 * 1e7, period="FY2023"),
        ),
    ),
)
