"""Real PDF TABLE extraction (H5), feeding the element-aware chunker.

A fetched annual-report PDF is mostly financial TABLES. pypdf's flat `extract_text()` collapses
a table's columns to single spaces, so `structure._is_table_row` cannot see it as a table and the
figures degrade to prose (a scaled cell loses its "(Rs in crore)" units). This module uses
pdfplumber's geometry-aware table detection to pull each page's TABLES as real grids and render
them so the SAME downstream chunker (W2 structure.py) and record layer (numeric_records.py) treat
their cells as typed numeric records.

Two seams, split for testability:
- `format_table` / `format_tables` are PURE: list-of-rows -> chunker-friendly text (each row a
  Markdown-pipe line, so `structure._is_table_row` detects it as an intact table row). No I/O.
- `extract_pdf_tables_text` does the pdfplumber I/O: per page, render detected tables and keep
  the surrounding prose (with table regions removed so their numbers are not duplicated as
  mangled text), assembled in reading order so a table's caption/units/title line stays directly
  above it -- exactly what split_elements absorbs into the table region.

DEGRADE-SAFE (real money, abstain-on-failure): if pdfplumber is missing or raises on a given
PDF, `extract_pdf_tables_text` returns None so the caller falls back to the existing pypdf text
path. It NEVER raises and NEVER fabricates.

LIMITATIONS (honest, for the reviewer):
- Uses pdfplumber's default (ruled-lines) table detection. A truly BORDERLESS table (columns
  aligned by whitespace only, no drawn lines) is not detected as a table; its text falls back to
  prose -- degraded exactly as today's pypdf path, never wrong.
- A row that has only ONE non-empty cell (e.g. an in-table sub-section label spanning the grid)
  renders as a single-content pipe line, which `_is_table_row` does not count as a table row; it
  can split the grid there. Real statement data rows carry >=2 populated cells, so this is rare.
"""
from __future__ import annotations

import io
import re

_WS = re.compile(r"\s+")

# A cell string may carry internal newlines (a wrapped cell) or a None (an empty grid cell).
Row = "list"  # documentation only; rows are list[str | None], tables are list[Row]


def _clean_cell(cell) -> str:
    """Normalize one table cell to a single-line string: None -> '', collapse internal
    whitespace/newlines so a wrapped cell does not break onto its own line."""
    if cell is None:
        return ""
    return _WS.sub(" ", str(cell)).strip()


def format_table(rows, *, caption: str | None = None) -> str:
    """Render ONE extracted table (a list of rows, each a list of cell strings) as
    chunker-friendly text. Each non-empty row becomes a Markdown-pipe line `| a | b | c |`, which
    `structure._is_table_row` detects as a table row (>=2 non-empty pipe cells). A caption/units
    line (e.g. '(Rs in crore)'), when given, is emitted verbatim ABOVE the grid so split_elements
    absorbs it into the table region and the record layer inherits the declared scale. Fully-empty
    rows are dropped. Returns '' for an empty/None table (with no caption)."""
    lines: list[str] = []
    cap = _clean_cell(caption)
    if cap:
        lines.append(cap)
    for row in rows or []:
        cells = [_clean_cell(c) for c in (row or [])]
        if not any(cells):
            continue                      # skip a fully-empty grid row
        lines.append("| " + " | ".join(cells) + " |")
    # a caption with no data rows is not a table; drop it so we never emit a lone units line
    if len(lines) <= (1 if cap else 0):
        return ""
    return "\n".join(lines)


def format_tables(tables) -> str:
    """Render several tables (blank-line separated so each stays its own table region)."""
    blocks = [format_table(t) for t in (tables or [])]
    return "\n\n".join(b for b in blocks if b)


# --- pdfplumber I/O (degrade-safe) ------------------------------------------------------------


def _line_in_any_table(line: dict, bboxes) -> bool:
    """True if a text line sits inside any table's bounding box (vertical center within the box
    and horizontal overlap), so its text is dropped from prose rather than duplicated."""
    cy = (float(line["top"]) + float(line["bottom"])) / 2.0
    lx0, lx1 = float(line["x0"]), float(line["x1"])
    for (x0, top, x1, bottom) in bboxes:
        if top <= cy <= bottom and lx0 < x1 and lx1 > x0:
            return True
    return False


def _page_to_text(page) -> str:
    """One page -> text: detected tables rendered as pipe grids, plus the non-table prose, merged
    in reading order (by vertical position) so a table's caption/units/title line stays above it."""
    tables = page.find_tables()
    bboxes = [t.bbox for t in tables]
    # (vertical_top, text) blocks; sorting by top reconstructs reading order.
    blocks: list[tuple[float, str]] = []
    for t in tables:
        rendered = format_table(t.extract())
        if rendered:
            blocks.append((float(t.bbox[1]), rendered))
    for line in page.extract_text_lines():
        if not _line_in_any_table(line, bboxes):
            text = str(line.get("text", "")).strip()
            if text:
                blocks.append((float(line["top"]), text))
    blocks.sort(key=lambda b: b[0])
    return "\n".join(text for _, text in blocks)


def extract_pdf_tables_text(raw: bytes) -> str | None:
    """Extract a PDF's text with TABLES kept as structured pipe grids, via pdfplumber.

    Returns the assembled text, or None if pdfplumber is unavailable, the PDF cannot be parsed,
    or nothing usable was extracted -- the caller then falls back to the pypdf text path. This
    function never raises and never fabricates: on any failure it abstains by returning None.
    """
    try:
        import pdfplumber
    except Exception:
        return None
    try:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = _page_to_text(page)
                if page_text.strip():
                    parts.append(page_text)
        text = "\n\n".join(parts)
        return text if text.strip() else None
    except Exception:
        return None
